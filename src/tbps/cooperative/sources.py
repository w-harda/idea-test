"""在原冻结来源核验上绑定协同目标和可计算的额外前向上界。"""
from pathlib import Path
import math
from tbps.artifacts import read_json,sha_file
from tbps.alternating import fresh_sources as old
from .objectives import need
Grant=old.Grant
KEYS={'g_synergy_weight','h_synergy_weight','edit_credit_weight','caption_set_weight','residual_mix','attribute_temperature','caption_temperature'}
def augment(f,c):
 cfg=f['cooperative'];need(set(cfg)==KEYS,'exact cooperative settings')
 need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in cfg.values()),'finite settings')
 need(0<=cfg['residual_mix']<=1 and cfg['attribute_temperature']>0 and cfg['caption_temperature']>0,'temperatures/mix')
 need(f['method']=='GH-Cooperative-v1','explicit new method identity')
 c['cooperative']=dict(cfg);c['training_configuration']['cooperative']=dict(cfg)
 entries={e['identity']['row_id']:e for e in read_json(Path(c['prototype']['history_root'])/'qualification/train.json')['entries']}
 n=len(c['prototype']['samples']);rounds=c['prototype']['rounds']
 c['smoke']['max_H_forwards']+=rounds*n
 c['smoke']['max_IRRA_texts']+=rounds*(2*n+sum(len(entries[x]['attributes']) for x in c['prototype']['samples']))
 return f,c
def load(path,scope):return augment(*old.load(path,scope))
def preflight(path,scope):
 f,c,b,g,p=old.preflight(path,scope);augment(f,c)
 p.update(status='GH_COOP_CPU_PREFLIGHT_PASS',method=f['method'],cooperative=f['cooperative'],forward_caps=c['smoke'])
 return f,c,b,g,p
