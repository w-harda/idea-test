"""首次torchvision导入的真实CPU启动时序，CUDA调用全部Mock。"""
import os,subprocess,sys
from pathlib import Path

def test_real_cpu_preprocessing_import_before_gate_then_mock_cuda():
    root=Path(__file__).resolve().parents[2]
    code=r'''
import sys,importlib.util
from unittest.mock import patch
import torch
events=[]
import tbps.b3_eval as entry
from tbps.baselines.launch import main
def run():
    from tbps.materialization.irra_startup import CPUStartupGate,isolation_snapshot
    # launch已完成首次torchvision导入；原像素工具及真实CPU预处理不得再改变快照。
    gate=CPUStartupGate()
    before=isolation_snapshot()
    spec=importlib.util.spec_from_file_location("_cpu_b3_reader",sys.argv[1])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from PIL import Image
    x=module.native_pixels(Image.new("RGB",(24,30),(100,120,140))).unsqueeze(0)
    assert x.shape==(1,3,256,128) and not torch.cuda.is_initialized()
    assert isolation_snapshot()==before
    events.append("cpu_read_complete")
    gate.checked();gate.exit_checked();events.append("isolation_exited")
    gate.initialize_cuda()
    events.append("mock_model_entry")
with patch.object(torch.cuda,"reset_peak_memory_stats",side_effect=lambda:events.append("mock_cuda")),patch.object(entry,"main",side_effect=run):
    main()
assert events==["cpu_read_complete","isolation_exited","mock_cuda","mock_model_entry"]
assert not torch.cuda.is_initialized()
print("B3_CPU_STARTUP_ORDER_PASS")
'''
    env=dict(os.environ,PYTHONPATH=str(root/"src"),CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
    p=subprocess.run([sys.executable,"-B","-c",code,str(root/"src/attributes/ap_gallery_baseline.py")],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert "B3_CPU_STARTUP_ORDER_PASS" in p.stdout
