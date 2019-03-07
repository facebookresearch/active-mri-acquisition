import unittest
import numpy as np

from common.args import Args
from common.subsample import Mask


class MaskTest(unittest.TestCase):
    def test_fixed_mask(self):
        args = Args().parse_args()
        shape = (32, 1, 128, 128, 2)
        mask_func = Mask(reuse_mask=True)
        mask1 = mask_func(shape, args)
        mask2 = mask_func(shape, args)
        mask3 = mask_func(shape, args)
        assert np.all(mask1 == mask2) and np.all(mask2 == mask3)

    def test_random_mask(self):
        args = Args().parse_args()
        shape = (32, 1, 128, 128, 2)
        mask_func = Mask(reuse_mask=False)
        mask1 = mask_func(shape, args)
        mask2 = mask_func(shape, args)
        mask3 = mask_func(shape, args)
        assert np.any(mask1 != mask2) and np.any(mask2 != mask3)
