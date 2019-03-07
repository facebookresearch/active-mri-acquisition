import unittest
import numpy as np
import torch
from common.ctorch import ComplexTensor, fft, ifft, fft2, ifft2


class ComplexTest(unittest.TestCase):

    def random_z(self, size=(10, 10)):
        return ComplexTensor(torch.Tensor(*size).normal_(),
                             torch.Tensor(*size).normal_())

    def test_add(self):
        A, B = self.random_z(), self.random_z()
        np.testing.assert_allclose((A + B).numpy(), A.numpy() + B.numpy())

    def test_sub(self):
        A, B = self.random_z(), self.random_z()
        np.testing.assert_allclose((A - B).numpy(), A.numpy() - B.numpy())

    def test_matmul(self):
        A, B = self.random_z(), self.random_z()
        np.testing.assert_allclose((A @ B).numpy(), A.numpy() @ B.numpy(),
                                   rtol=1e-05, atol=1e-08)

    def test_t(self):
        A = self.random_z()
        np.testing.assert_allclose((A.t()).numpy(), A.numpy().T)

    def test_conjt(self):
        A = self.random_z()
        np.testing.assert_allclose((A.h()).numpy(), A.numpy().conj().T)

    def test_mul(self):
        A = self.random_z()
        np.testing.assert_allclose((A * 2).numpy(), A.numpy() * 2)

    def test_abs(self):
        A = self.random_z()
        np.testing.assert_allclose(abs(A), abs(A.numpy()), rtol=1e-05,
                                   atol=1e-08)

    def test_fft(self):
        for _ in range(3):
            for sz in [(100,), (10, 10), (2, 5, 10), (2, 3, 6, 7)]:
                X = self.random_z(size=sz).double().cuda()
                np.testing.assert_allclose(fft(X).cpu().numpy(),
                                           np.fft.fft(X.cpu().numpy()))
                np.testing.assert_allclose(ifft(X).cpu().numpy(),
                                           np.fft.ifft(X.cpu().numpy()))

    def test_fft2(self):
        for _ in range(3):
            for sz in [(10, 10), (1, 256, 256), (2, 5, 10), (2, 2, 5, 5)]:
                X = self.random_z(size=sz).double().cuda()
                np.testing.assert_allclose(fft2(X).cpu().numpy(),
                                           np.fft.fft2(X.cpu().numpy()))
                np.testing.assert_allclose(ifft2(X).cpu().numpy(),
                                           np.fft.ifft2(X.cpu().numpy()))
