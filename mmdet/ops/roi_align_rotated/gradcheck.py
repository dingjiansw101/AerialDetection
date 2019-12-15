# TODO
import random
import numpy as np
import torch
from torch.autograd import gradcheck

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from roi_align_rotated import RoIAlignRotated  # noqa: E402
from mmdet.core.evaluation import RotBox2Polys
import unittest


def bilinear_interpolate(bottom, height, width, y, x):
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return 0.0, []
    x = max(0.0, x)
    y = max(0.0, y)
    x_low = int(x)
    y_low = int(y)
    if x_low >= width - 1:
        x_low = x_high = width - 1
        x = x_low
    else:
        x_high = x_low + 1

    if y_low >= height - 1:
        y_low = y_high = height - 1
        y = y_low
    else:
        y_high = y_low + 1

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    v1 = bottom[y_low, x_low]
    v2 = bottom[y_low, x_high]
    v3 = bottom[y_high, x_low]
    v4 = bottom[y_high, x_high]

    '''
    ----------->x
    |hx hy | lx hy
    |------+------
    |hx ly | lx ly
    V
    y
    v1|v2
    --+--
    v3|v4
    '''
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    grad = [(y_low, x_low, w1), (y_low, x_high, w2),
            (y_high, x_low, w3), (y_high, x_high, w4)
            ]
    return val, grad
class test_op_roi_align_rotated(unittest.TestCase):

    def test_roi_align_rotated_value(self):
        data = np.arange(16).reshape(1, 1, 4, 4).astype('float64')
        rois = np.array([[0, 1.0, 1.0, 2., 2., -np.pi/2.], [0, 1.0, 1.0, 2., 2., 0],
                           [0, 1.0, 1.0, 2., 2., np.pi/2.], [0, 1.0, 1.0, 2., 2., np.pi]], dtype='float64')
        data = torch.from_numpy(data).float().cuda()
        rois = torch.from_numpy(rois).float().cuda()
        expected_feat1 = np.array([[6.5, 2.5],
                                   [7.5, 3.5]])
        expected_feat2 = np.array([[2.5, 3.5],
                                   [6.5, 7.5]])
        expected_feat3 = np.array([[3.5, 7.5],
                                   [2.5, 6.5]])
        expected_feat4 = np.array([[7.5, 6.5],
                                   [3.5, 2.5]])
        expected_feat = np.concatenate((expected_feat1[np.newaxis, np.newaxis, :, :],
                                        expected_feat2[np.newaxis, np.newaxis, :, :],
                                        expected_feat3[np.newaxis, np.newaxis, :, :],
                                        expected_feat4[np.newaxis, np.newaxis, :, :]), axis=0)

        roialign_rotated = RoIAlignRotated(out_size=2, spatial_scale=1, sample_num=0)
        results = roialign_rotated(data, rois).cpu().numpy()
        np.testing.assert_almost_equal(results, expected_feat, decimal=6)

    def test_roi_align_rotated_autograd(self):
        # x1 = np.random.rand(4, 1, 12, 12).astype('float64')
        # x2_t = np.array([[0, 6.2, 6.3, 4.2, 4.4, np.pi / 4.], [2, 4.1, 4.2, 6.2, 6.0, -np.pi],
        #                  [1, 6.0, 6.3, 4.0, 4.1, 3 * np.pi / 4.]], dtype='float64')
        # polys2_t = RotBox2Polys(x2_t[:, 1:])
        x2 = np.array([[0, 6.2, 6.0, 4.0, 4.0, np.pi / 2.],
                       [0, 6.3, 6.0, 4.0, 4.0, -np.pi / 2.],
                       [0, 6.0, 6.0, 4.0, 4.0, -np.pi],
                       [0, 6.0, 6.0, 4.3, 4.0, np.pi],
                       [1, 6.0, 6.0, 4.0, 4.0, np.pi / 3.],
                       [2, 4.1, 4.2, 6.2, 6.0, -np.pi],
                       [1, 6.0, 6.3, 4.0, 4.1, 3 * np.pi / 4.],
                       [0, 6.2, 6.3, 4.2, 4.4, np.pi / 4.]
                       ], dtype='float64')
        x1 = torch.rand(4, 1, 12, 12, requires_grad=True, device='cuda:0')
        x2 = torch.from_numpy(x2).float().cuda()
        inputs = (x1, x2)
        print('Gradcheck for roi align...')
        spatial_scale = 1
        test = gradcheck(RoIAlignRotated(4, spatial_scale), inputs, atol=1e-3, eps=1e-3)
        print(test)
        test = gradcheck(RoIAlignRotated(4, spatial_scale, 2), inputs, atol=1e-3, eps=1e-3)
        print(test)

if __name__ == '__main__':
    unittest.main()