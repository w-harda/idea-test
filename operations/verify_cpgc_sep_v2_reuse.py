import sys,json,importlib.util,pathlib,time
s=importlib.util.spec_from_file_location('prior_resume','/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-resume-20261029/operations/cpgc_sep_resume.py')
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'));m.frozen();c=m.read(m.CONFIG);c['sha256']=m.CONFIG_SHA
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore,DigestSession
from tbps.artifacts import sha_file
session=DigestSession()
def deep_verify(c,step):
 root=step.root;store=VerifiedArtifactStore(root,new_base=c['output_base'],session=session)
 assert not store.pending() and not (root/'FAIL.json').exists()
 for claim in root.glob('*.Claim.json'):assert (root/claim.name.removesuffix('.Claim.json')/'manifest.json').is_file()
 for p in root.glob('*/manifest.json'):store.verify(p.parent.name)
 return m.verify_step(c,step,deep=False)
out=[];start=time.monotonic()
for step in m.steps(c):
 if step.root.exists() and (step.root/'completed/manifest.json').exists():
  print('VERIFY',step.label,flush=True);out.append(deep_verify(c,step))
p=pathlib.Path('/home/lzf/ldx/tmp/cpgc_sep_throughput_prepare_v3_20261031/verified_v2_reuse.json')
m.atomic(p,{'status':'PASS','steps':out,'elapsed':time.monotonic()-start,'Full_audits':sum('_full_' in x['label'] and x['label'].endswith('_audit') for x in out)})
print('FIVE_FULL_REUSE_PASS',flush=True)
