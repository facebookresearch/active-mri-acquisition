import unittest
from common.admm2 import runtestadmm


class TestADMM2(unittest.TestCase):
    filename = 'data/shepp_logan_32x64.png'
    kwargs = {'mu': 1e6, 'seed': 1337}

    def test_baseline_fft_cpu(self):
        print()
        kwargs = self.kwargs.copy()
        kwargs['n_iter'] = 10
        l0 = runtestadmm('cs_baseline', True, self.filename, **kwargs)
        l1 = runtestadmm('cs_fft', True, self.filename, **kwargs)
        self.assertAlmostEqual(l0, l1)

    def test_baseline_fft_gpu(self):
        print()
        kwargs = self.kwargs.copy()
        kwargs['n_iter'] = 10
        l0 = runtestadmm('cs_baseline', True, self.filename, **kwargs)
        l1 = runtestadmm('cs_fft', False, self.filename, **kwargs)
        self.assertAlmostEqual(l0, l1)

    def test_fft_cpu_fft_gpu(self):
        print()
        kwargs = self.kwargs.copy()
        kwargs['n_iter'] = 100
        l0 = runtestadmm('cs_fft', True, self.filename, **kwargs)
        l1 = runtestadmm('cs_fft', False, self.filename, **kwargs)
        self.assertAlmostEqual(l0, l1)
