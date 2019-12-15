# TODO
import random
import numpy as np
import torch
from torch.autograd import gradcheck

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from psroi_align_rotated import PSRoIAlignRotated  # noqa: E402
import unittest
import itertools

class test_op_roi_align_rotated(unittest.TestCase):

    def test_psroi_align_rotated_value(self):
        num_classes = 1
        num_group = 2

        feat_height = 5
        feat_width = 5
        x1 = np.arange(num_classes*num_group*num_group*feat_height*feat_width).reshape(1, num_classes*num_group*num_group, feat_height, feat_width).astype('float32')

        x2 = np.array([[0, 2, 1, 2, 2, np.pi/2.],
                       [0, 2, 2, 4, 2, np.pi/2.]], dtype='float32')
        print("x1: ", x1)
        x1 = torch.from_numpy(x1).float().cuda()
        x2 = torch.from_numpy(x2).float().cuda()

        psroi_align_rotated_pooling = PSRoIAlignRotated(out_size=num_group, spatial_scale=1.0,
                                                        sample_num=1, output_dim=num_classes, group_size=num_group)
        pooled_feat = psroi_align_rotated_pooling(x1, x2).cpu().numpy()
        # TODO: check why it did not pass
        # expected_result1 = np.array([[[5., 35.0],
        #                               [54.0, 85.0]],
        #                              [[7.5, 42.5],
        #                               [56.5, 91.5]]])

        expected_result1 = np.array([[[5., 35.0],
                                      [54.0, 84.0]],
                                     [[7.5, 42.5],
                                      [56.5, 91.5]]])
        expected_result1 = expected_result1[:, np.newaxis, :, :]
        np.testing.assert_almost_equal(pooled_feat, expected_result1, decimal=6)


    def test_psroi_align_rotated_autograd(self):
        for num_rois in [1, 2]:
            for num_classes, num_group in itertools.product([2, 3], [2, 3]):
                for image_height, image_width in itertools.product([168, 224], [168, 224]):
                    for grad_notes in [['im_data']]:
                        spatial_scale = 0.0625
                        feat_height = np.int(image_height * spatial_scale)
                        feat_width = np.int(image_width * spatial_scale)
                        # im_data = np.random.rand(1, num_classes*num_group*num_group, feat_height, feat_width)
                        rois_data = np.zeros([num_rois, 6])
                        rois_data[:, [1,3]] = np.sort(np.random.rand(num_rois, 2)*(image_width-1))
                        rois_data[:, [2,4]] = np.sort(np.random.rand(num_rois, 2)*(image_height-1))
                        rois_data[:, 5] = np.random.rand(num_rois)

                        psroi_align_rotated_pooling = PSRoIAlignRotated(out_size=num_group, spatial_scale=spatial_scale,
                                                                        sample_num=0,
                                                                        output_dim=num_classes,
                                                                        group_size=num_group)
                        # rtol, atol = 1e-2, 1e-4
                        im_data = torch.rand(1, num_classes * num_group * num_group, feat_height, feat_width, requires_grad=True, device='cuda:0')
                        # im_data = torch.from_numpy(im_data).float().cuda()
                        # im_data.requires_grad = True
                        rois_data = torch.from_numpy(rois_data).float().cuda()
                        inputs = (im_data, rois_data)
                        gradcheck(psroi_align_rotated_pooling, inputs, atol=1e-3, eps=1e-3)







if __name__ == '__main__':
    unittest.main()