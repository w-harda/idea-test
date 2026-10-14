"""同一已测caption的一步默认scale/transform形状检查；不扩数据。"""
import run as runner
from run import *

def main():
 hashes,rows,payloads,bases=runner.inputs_and_records();cid='cuhk:34:0';record=next(r for r in rows if r['row_id']==cid);payload=next(p for p in payloads if p['caption_id']==cid)
 sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v4_learned_reader_20261003'));import learned_reader as v4
 v4.setup();torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 victim=v4.load_victim();source=IRRASurrogate(victim);plan=prepared_plan(payload,record,bases,'cuda');versions={n:p._version for n,p in victim.named_parameters()}
 with Image.open(Path('/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs')/record['paired_image']) as im:image=victim.pixels(im).unsqueeze(0).cuda()
 transport=TextTransport(victim,record['caption']);v,mapping=transport.read(plan[0]);seed_all(42);t0=time.time();torch.cuda.reset_peak_memory_stats()
 engine=AttributeTTA(source,image,steps=1);adv=engine.run(plan,{plan[0].instance_id:v});torch.cuda.synchronize()
 assert all(not p.requires_grad and p.grad is None and p._version==versions[n] for n,p in victim.named_parameters());runner.verify(hashes)
 result={'caption_id':cid,'case_count':1,'additional_new_captions':0,'steps':1,'transforms':engine.attack.N_trans,'scales':engine.scales,'seconds':time.time()-t0,'peak_allocated_GB':torch.cuda.max_memory_allocated()/1024**3,'rounds':engine.events,'mapping':mapping,'frozen_source_SHA_unchanged':True,'IRRA_frozen':True,'source_sha256':digest(__file__),'no_formal_experiment':True}
 out=ROOT/'outputs/default_shape_smoke.json'
 if out.exists():raise FileExistsError(out)
 save(out,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
