"""Checks the gather/scatter sparse convolutions against dense conv3d.

Run: python3 -m unittest discover -s scripts/coda   (skipped without PyTorch)
"""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn.functional as F

    import detect_lidar_cpu as det
except ImportError:  # PyTorch / Shapely are optional for CI
    torch = None


def random_sparse(shape, count, channels, seed=0):
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randperm(shape[0] * shape[1] * shape[2], generator=generator)[:count]
    coords = torch.stack([flat // (shape[1] * shape[2]), (flat // shape[2]) % shape[1], flat % shape[2]], dim=1)
    feats = torch.randn(count, channels, generator=generator)
    return det.Sparse(coords, feats, shape)


def dense(x):
    grid = torch.zeros(x.feats.shape[1], *x.shape)
    grid[:, x.coords[:, 0], x.coords[:, 1], x.coords[:, 2]] = x.feats.T
    return grid


def spconv_weight_to_torch(weight):
    # spconv stores (out, kz, ky, kx, in); torch conv3d wants (out, in, kz, ky, kx).
    return weight.permute(0, 4, 1, 2, 3).contiguous()


@unittest.skipIf(torch is None, "PyTorch not installed")
class SparseConvTest(unittest.TestCase):
    def test_submanifold_matches_dense_conv_at_active_sites(self):
        x = random_sparse((7, 9, 11), 60, 3)
        weight = torch.randn(4, 3, 3, 3, 3)
        y = det.subm_conv(x, weight)
        reference = F.conv3d(dense(x)[None], spconv_weight_to_torch(weight), padding=1)[0]
        expected = reference[:, x.coords[:, 0], x.coords[:, 1], x.coords[:, 2]].T
        torch.testing.assert_close(y.feats, expected, rtol=1e-4, atol=1e-4)

    def test_strided_sparse_conv_matches_dense_conv(self):
        for kernel, stride, padding in (((3, 3, 3), (2, 2, 2), (1, 1, 1)), ((3, 3, 3), (2, 2, 2), (0, 1, 1)),
                                        ((3, 1, 1), (2, 1, 1), (0, 0, 0))):
            x = random_sparse((11, 10, 12), 80, 3, seed=1)
            weight = torch.randn(5, *kernel, 3)
            y = det.sparse_conv(x, weight, stride, padding)
            reference = F.conv3d(dense(x)[None], spconv_weight_to_torch(weight), stride=stride, padding=padding)[0]
            self.assertEqual(tuple(reference.shape[1:]), y.shape)
            torch.testing.assert_close(dense(y), reference.masked_fill(dense(y).abs().sum(0, keepdim=True) == 0, 0.0),
                                       rtol=1e-4, atol=1e-4)
            # Every output site the dense conv can reach from an active input is active.
            reach = F.conv3d((dense(x).abs().sum(0) > 0).float()[None, None], torch.ones(1, 1, *kernel),
                             stride=stride, padding=padding)[0, 0] > 0
            self.assertEqual(int(reach.sum()), len(y.coords))


if __name__ == "__main__":
    unittest.main()
