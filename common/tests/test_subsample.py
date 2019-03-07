import unittest
from common import subsample, args

class TestSubsample(unittest.TestCase):
    args = args.Args().parse_args(args=[])
    
    # the mask ratio returned should have exact subsampling_ratio 
    def test_lf_focused(self):
        self.args.subsampling_type = 'lf_focused'
        mask_func = subsample.Mask(reuse_mask=True) 
        mask = mask_func(shape=(100,1,128,128,2), args=self.args)
        for m in mask:
            assert m.sum()/(128*128*2) == 1.0/self.args.subsampling_ratio

    def test_hf_focused(self):
        self.args.subsampling_type = 'hf_focused'
        mask_func = subsample.Mask(reuse_mask=True) 
        mask = mask_func(shape=(100,1,128,128,2), args=self.args)
        for m in mask:
            assert m.sum()/(128*128*2) == 1.0/self.args.subsampling_ratio
        # one batch
        mask = mask_func(shape=(1,128,128,2), args=self.args)
        for m in mask:
            assert m.sum()/(128*128*2) == 1.0/self.args.subsampling_ratio

    def test_random(self):
        self.args.subsampling_type = 'random'
        mask_func = subsample.Mask(reuse_mask=True) 
        mask = mask_func(shape=(100,1,128,128,2), args=self.args)
        for m in mask:
            assert m.sum()/(128*128*2) == 1.0/self.args.subsampling_ratio
    
    def test_consistent_mask(self):
        self.args.subsampling_type = 'lf_focused'        
        ref = None
        for _ in range(100):
            mask = subsample.Mask(reuse_mask=True)(shape=(1,128,128,2), args=self.args)
            if ref is None:
                ref = mask
            else:
                assert (ref - mask).sum() == 0

