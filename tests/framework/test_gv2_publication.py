"""纯报告收尾门禁测试；没有实验调用或Git写入。"""
from pathlib import Path
import importlib.util
import pytest
p=Path(__file__).resolve().parents[2]/'operations/finalize_gv2_cross_dataset.py'
s=importlib.util.spec_from_file_location('report_finish',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
@pytest.mark.parametrize('phase,expected',[('RUNNING_ICFG_FULL','WAIT'),('ICFG_AUDIT','WAIT'),('PUBLISHING_REPORT','WAIT'),('FAIL','STOP'),('SUCCESS','PUBLISH')])
def test_only_completed_success_can_publish(phase,expected):assert m.decision({'phase':phase})==expected
@pytest.mark.parametrize('mutation',['valid','smoke','missing','metric'])
def test_exact_four_full_rows(mutation):
 audits={ds:{'groups':{mode:{k:12.345 for k in ('R@1','R@5','R@10','mAP')} for mode in ('Clean','Image-only')}} for ds in ('rstp','icfg')}
 rows=[dict(Dataset=ds,Victim='IRRA',Mode=mode,**{k:'12.345' for k in ('R@1','R@5','R@10','mAP')}) for ds in ('RSTPReid','ICFG-PEDES') for mode in ('Clean','Image-only (G-v2 E60)')]
 if mutation=='smoke':rows[0]['Mode']='Smoke'
 if mutation=='missing':rows.pop()
 if mutation=='metric':rows[2]['mAP']='9'
 if mutation=='valid':m.verify_rows(rows,audits)
 else:
  with pytest.raises(ValueError):m.verify_rows(rows,audits)
