"""仅发布绘图的真实CSV身份与异常输入门禁。"""
import unittest,tempfile,hashlib
from pathlib import Path
from gh_v2_joint_plot import verified_rows

class PublicationTests(unittest.TestCase):
    def valid(self):return 'Round,R@1,R@5,R@10,mAP\n'+''.join(f'{n},30.25,50.5,60.75,20.125\n' for n in (10,20,30,40,50,60))
    def parse(self,s,sha=None):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'metrics.csv';p.write_text(s)
            return verified_rows(p,sha or hashlib.sha256(p.read_bytes()).hexdigest())
    def test_exact_values_not_scaled(self):
        rows=self.parse(self.valid());self.assertEqual(rows[0],{'Round':10,'R@1':30.25,'R@5':50.5,'R@10':60.75,'mAP':20.125})
    def test_sha_mismatch(self):
        with self.assertRaisesRegex(ValueError,'SHA'):self.parse(self.valid(),'0'*64)
    def test_wrong_endpoints(self):
        with self.assertRaises(ValueError):self.parse(self.valid().replace('60,','50,'))
    def test_nonfinite_range_extra_column(self):
        for value in ('nan','inf','101','-1','30.25,100'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):self.parse(self.valid().replace('30.25',value,1))
if __name__=='__main__':unittest.main()
