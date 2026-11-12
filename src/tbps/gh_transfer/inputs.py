"""独立CPU逐图/逐Query复核历史输入，不调用G/H或目标模型。"""
from pathlib import Path
from dataclasses import asdict
import time,resource
from .common import *

def audit_inputs(path,method):
    import torch
    from PIL import Image
    from torchvision.transforms.functional import resize,to_tensor
    from tbps.alternating.parents import cpu_restore_only
    from tbps.alternating.native_sources import native_session
    from tbps.alternating.fresh_audit import validate_action
    from tbps.alternating.state import tree_sha
    torch.set_num_threads(2)
    c=config(path);h,f,test,session=historical(c,method);m=manifest(c)
    dest=ArtifactStore(Path(c['output_base'])/'inputs'/method,new_base=c['output_base'])
    need(not dest.root.exists(),'input audit Attempt consumed; no overwrite/retry')
    dest.publish('claim',{'configuration.json':json_bytes(asdict(ref(path)))},{'method':method,'round':60})
    start=time.monotonic();attack=ArtifactStore(test/'attack',new_base=h['output_base'])
    images=[];texts=[];protection={};maximum=0.;max_edits=0
    def protect(p):
        p=Path(p);v=p.stat();protection[str(p)]={'size':v.st_size,'mtime_ns':v.st_mtime_ns,'inode':v.st_ino,'sha256':sha_file(p)}
    allowed=[test/'attack'/f'image_{i:04d}/data.pt' for i in range(3074)]+[h[s]['path'] for s in ('G','H')]
    try:
        with cpu_restore_only(allowed),native_session(session) as ns,torch.no_grad():
            contracts=ns.import_module('contracts');constraints=ns.import_module('constraints')
            cp=read_json(Path(h['output_base'])/'train/checkpoints/round_60/manifest.json')
            for side in ('G','H'):
                v=torch.load(h[side]['path'],map_location='cpu',weights_only=True)
                need(v['round']==60 and v['model']==side and v['configuration_sha256']==h['configuration']['sha256'],'checkpoint envelope')
                need(tree_sha(v['state_dict'])==cp['provenance'][side+'_state_sha'],'checkpoint state digest');del v
            for i,g in enumerate(m.gallery):
                key=f'image_{i:04d}';md=attack.verify(key);data=test/'attack'/key/'data.pt'
                item={'id':g.image_id,'data':asdict(ref(data)),'manifest':asdict(ref(data.parent/'manifest.json')),
                      'original_sha256':g.image_sha256,'adversarial_sha256':md['provenance']['adversarial_rgb']}
                need(md['provenance']['id']==g.image_id and md['provenance']['original_file']==g.image_sha256,'image row identity')
                v=image_value(item);source=Path(c['image_root'])/g.image_rel_path;checked(FileRef(str(source),g.image_sha256))
                with Image.open(source) as im:original=to_tensor(resize(im.convert('RGB'),[384,128])).unsqueeze(0)
                need(torch.equal(original,v['original']),'stored clean pixels equal original IRRA preprocessing')
                maximum=max(maximum,float(v['effective_delta'].double().abs().max()));images.append(item)
                for q in (data,data.parent/'manifest.json',data.parent/'completed.json'):protect(q)
                if (i+1)%64==0 or i==3073:print('TRANSFER_INPUT_IMAGE',method,i+1,3074,flush=True)
            for i,q in enumerate(m.queries):
                key=f'text_{i:04d}';md=attack.verify(key);data=test/'attack'/key/'action.json';a=read_json(data)
                need(a['original']==a['native_action']['original']==q.caption and a['adversarial']==a['native_action']['adversarial'],'caption/native original binding')
                n=validate_action(a['native_action'],contracts,constraints);max_edits=max(max_edits,n)
                need(md['provenance']=={'id':q.query_id,'original_utf8':utf8(q.caption),'adversarial_utf8':utf8(a['adversarial'])},'text row identity')
                texts.append({'id':q.query_id,'data':asdict(ref(data)),'manifest':asdict(ref(data.parent/'manifest.json')),
                              'original_sha256':utf8(q.caption),'adversarial_sha256':utf8(a['adversarial'])})
                for z in (data,data.parent/'manifest.json',data.parent/'completed.json'):protect(z)
                if (i+1)%128==0 or i==6155:print('TRANSFER_INPUT_TEXT',method,i+1,6156,flush=True)
        for z in [h['configuration']['path'],h['G']['path'],h['H']['path'],test/'completed/summary.json',
                  test/'cpu_audit/completed/report.json',Path(h['output_base'])/'train_cpu_audit/round_60/completed/report.json']:protect(z)
        for z in h['proofs'].values():protect(z['path'])
        value={'schema':'gh-transfer-collection-1','historical':h,'manifest':asdict(m),'images':images,'texts':texts}
        report={'status':'GH_TRANSFER_INPUT_CPU_AUDIT_PASS','method':method,'round':60,'gallery':3074,'query':6156,
                'collection_sha256':digest(value),'max_Linf':maximum,'maximum_edited_words':max_edits,
                'original_pixels_exact':True,'native_Unicode_replay':True,'G_H_forward':0,'optimizer':0,'JVP':0,
                'elapsed_seconds':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        dest.publish('completed',{'collection.json':json_bytes(value),'report.json':json_bytes(report),
                                 'protection.json':json_bytes(protection)},
                     {'method':method,'round':60},sources=(ref(path),FileRef(**h['configuration']),FileRef(**h['G']),FileRef(**h['H']),ref(test/'cpu_audit/completed/manifest.json')))
        print('TRANSFER_INPUT_AUDIT_PASS',method,flush=True);return report
    except BaseException:
        import traceback
        (dest.root/'FAIL').write_text(traceback.format_exc());raise
