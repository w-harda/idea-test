import unittest
import torch
from run import parser,centroids,geometries,validation
class FakeFeatures:
    def __init__(self,z):self.z=z
    def get(self,ix):return self.z[ix]
class ProtocolTests(unittest.TestCase):
    def fixture(self):
        split={'slots':['toy'],'values':[['a','b','null']],'train_indices':[0,1,2,3], 'records':[{'labels':[x]} for x in [0,0,0,1,0,1,2]]}
        z=torch.zeros(7,1,512);z[0,0,0]=0;z[1,0,0]=3;z[2,0,0]=6;z[3,0,0]=12;z[4,0,0]=3;z[5,0,0]=12
        z[:,:,1]=1
        return split,FakeFeatures(z)
    def test_default_does_not_run_full(self):
        self.assertEqual(parser().parse_args([]).mode,'smoke');self.assertEqual(parser().parse_args(['--mode','full']).mode,'full')
    def test_centroids_equal_value_and_no_validation_leak(self):
        s,f=self.fixture();c=centroids(s,s['train_indices'],f)[0]
        self.assertEqual(c['counts'],[3,1,0]);self.assertEqual(c['centroids'][:,0].tolist(),[3,12])
        self.assertEqual(float(c['centroids'].mean(0)[0]),7.5)
        with self.assertRaisesRegex(AssertionError,'leakage'):centroids(s,[0,4],f)
    def test_unseen_value_counted_not_dropped(self):
        s,f=self.fixture();s['values'][0]=['a','b','null','unseen'];s['records'][4]['labels']=[3]
        c=centroids(s,s['train_indices'],f);geo=geometries(c);v=validation(s,[4,5,6],f,c,geo)[0]
        self.assertEqual(v['total_non_null'],2);self.assertEqual(v['uncovered_train_values'],1)
        for e in v['methods'].values():self.assertLessEqual(e['correct'],1)
        self.assertEqual(v['full_euclidean_prediction_disagreements'],0)
if __name__=='__main__':unittest.main()
