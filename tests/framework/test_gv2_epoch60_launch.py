"""只验证后台启动保护和训练标量汇总；不启动CUDA或真实训练。"""
import copy,csv,json,subprocess,tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from tbps.ablations import gv2_epoch60_results as r

class LaunchTests(unittest.TestCase):
    def rows(self):
        return [{"epoch":e,**{k:.01 for k in r.COLUMNS if k not in ("epoch","lr","seconds")},
                 "lr":.0002,"seconds":""} for e in range(1,61)]
    def test_complete_epoch_curve(self):
        rows=self.rows();self.assertEqual(r.validate_epochs(rows),rows)
    def test_missing_duplicate_or_wrong_epoch_refused(self):
        rows=self.rows()
        for values in (rows[:-1],rows+[rows[-1]],list(reversed(rows))):
            with self.assertRaises(ValueError):r.validate_epochs(values)
    def test_nonfinite_and_changed_lr_refused(self):
        for key,value in (("train_total",float("nan")),("validation_global",float("inf")),("lr",.001)):
            values=self.rows();values[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_epochs(values)
    def test_shell_syntax(self):
        p=Path(r.__file__).with_name("run_gv2_epoch60_train.sh")
        self.assertEqual(subprocess.run(["bash","-n",str(p)]).returncode,0)
    def test_unauthorized_launcher_does_not_start(self):
        p=Path(r.__file__).with_name("run_gv2_epoch60_train.sh")
        control=Path("/home/lzf/ldx/tmp/gv2_epoch60_train_v1")
        before=(control/"started").exists()
        result=subprocess.run(["bash",str(p)],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertEqual(before,(control/"started").exists())
    def test_no_results_for_not_started_training(self):
        with tempfile.TemporaryDirectory(dir="/home/lzf/ldx/tmp",prefix="tbps-framework-empty-results-") as tmp:
            base=Path(tmp);p=base/"config.json";out=base/"train"
            p.write_text(json.dumps({"formal_output":str(out),"output_base":str(base)}))
            with self.assertRaises((ValueError,FileNotFoundError)):r.collect(p)
            self.assertFalse(out.exists())
    def test_real_cpu_plotting_with_synthetic_curve(self):
        with tempfile.TemporaryDirectory(dir="/home/lzf/ldx/tmp",prefix="tbps-framework-curve-test-") as tmp:
            p=Path(tmp)
            with (p/"loss_raw.csv").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=r.COLUMNS);w.writeheader();w.writerows(self.rows())
            r.plot(self.rows(),p)
            self.assertGreater((p/"loss_curves.png").stat().st_size,1000)
            self.assertEqual(ET.parse(p/"loss_curves.svg").getroot().tag,"{http://www.w3.org/2000/svg}svg")
if __name__=="__main__":unittest.main()
