import unittest
import numpy as np
from transforms_rbbox import *
import torch
import copy

class Test_transforms_rbbox(unittest.TestCase):

    def test_dbbox2delta(self):
        """
        encoding format similar to RRPN, except the angle was restricted to [0, 2 pi], dangle was restricted to [0, 1]

        Must Test corner cases
        :return:
        """
        boxlist1 = torch.DoubleTensor([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 100, 60, 34, np.pi/2]
                                  ])

        boxlist2 = torch.DoubleTensor([[1, 1, 5, 8, np.pi/16],
                                  [1, 1, 5, 8, np.pi/16 + np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 90, 12, 45, np.pi/10]
                                  ])
        expected_targets = torch.DoubleTensor([[0.0000,  0.0000, -0.6931,  0.4700,  0.0312],
                                        [0.0000,  0.0000, -0.6931,  0.4700,  0.0313],
                                        [0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                                        [-0.1667,  0.0000, -1.6094,  0.2803, 0.8]])

        output = dbbox2delta(boxlist1, boxlist2)
        np.testing.assert_almost_equal(expected_targets.numpy(), output.numpy(), decimal=4)

    def test_delta2dbbox(self):
        """
            similar to light-head rcnn, different classes share the same bbox regression now
        :return:
        """
        boxlist1 = torch.FloatTensor([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 100, 60, 34, np.pi/2]
                                  ])
        # the boxlist2(ground truths) are restricted to (0, 2 * pi)
        boxlist2 = torch.FloatTensor([[1, 1, 5, 8, np.pi/16],
                                  [1, 1, 5, 8, np.pi/16 + np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 90, 12, 45, np.pi/10]
                                  ])
        expected_targets = dbbox2delta(boxlist1, boxlist2)
        expected_boxlist2 = delta2dbbox(boxlist1, expected_targets)

        np.testing.assert_almost_equal(expected_boxlist2.numpy(), boxlist2.numpy(), decimal=5)

    def test_dbbox_flip(self):
        """

        :return:
        """
        dbboxes = torch.DoubleTensor([[50, 60, 50, 30, np.pi/3, 23.5, 60, 50, 30, np.pi/3 + np.pi/10],
                                      [30, 20, 30, 30, np.pi/6, 10.33, 20, 30, 30, np.pi/6 - np.pi/11]])
        img_shape = (1024, 1024)

        expected_targets = torch.DoubleTensor([[973, 60, 50, 30, np.pi/3*2, 999.5, 60, 50, 30, np.pi*17/30],
                                               [993, 20, 30, 30, np.pi/6*5, 1012.67, 20, 30, 30, np.pi*61/66]])

        output = dbbox_flip(dbboxes, img_shape)

        np.testing.assert_almost_equal(expected_targets.numpy(), output.numpy(), decimal=6)

    # def dbbox_mapping(self):
    #
    #     dbboxes = torch.DoubleTensor([[50, 60, 50, 30, np.pi/3, 23.5, 60, 50, 30, np.pi/3 + np.pi/10],
    #                                   [30, 20, 30, 30, np.pi/6, 10.33, 20, 30, 30, np.pi/6 - np.pi/11]])
    #     img_shape

    def test_xy2wh(self):

        boxes = torch.DoubleTensor([[1, 3, 45, 10],
                          [24.4, 3., 44.5, 52.2]])
        outputs = xy2wh(boxes)
        expected_outputs = np.array([[23, 6.5, 45, 8],
                                     [34.45, 27.6, 21.1, 50.2]])
        np.testing.assert_almost_equal(expected_outputs, outputs.numpy())

    def test_dbbox2roi(self):

        dbbox_list = [torch.DoubleTensor([[2, 3, 39, 30, np.pi/2],
                                          [3.2, 3, 30, 20, np.pi/3]]),
                      torch.DoubleTensor([[1, 3, 39, 30, np.pi / 2],
                                          [5.2, 3, 30, 20, np.pi / 3]]) ]

        expected_targets = np.array([[0, 2, 3, 39, 30, np.pi/2],
                                     [0, 3.2, 3, 30, 20, np.pi/3],
                                     [1, 1, 3, 39, 30, np.pi / 2],
                                     [1, 5.2, 3, 30, 20, np.pi / 3]
                                     ])

        outputs = dbbox2roi(dbbox_list)

        np.testing.assert_almost_equal(expected_targets, outputs.numpy())

    def test_droi2dbbox(self):

        drois = np.array([[0, 2, 3, 39, 30, np.pi/2],
                                     [0, 3.2, 3, 30, 20, np.pi/3],
                                     [1, 1, 3, 39, 30, np.pi / 2],
                                     [1, 5.2, 3, 30, 20, np.pi / 3]
                                     ])
        drois = torch.from_numpy(drois)

        outputs = droi2dbbox(drois)

        expected_targets = [torch.DoubleTensor([[2, 3, 39, 30, np.pi/2],
                                          [3.2, 3, 30, 20, np.pi/3]]),
                      torch.DoubleTensor([[1, 3, 39, 30, np.pi / 2],
                                          [5.2, 3, 30, 20, np.pi / 3]])]
        np.testing.assert_equal(len(outputs), 2)
        np.testing.assert_equal(expected_targets[0].shape == outputs[0].shape, True)
        np.testing.assert_equal(expected_targets[1].shape == outputs[1].shape, True)

        # np.testing.assert_almost_equal(expected_targets, outputs.numpy())

    def test_polygonToRotRectangle_batch(self):
        polygons = np.array([[0, 0, 3, 0, 3, 3, 0, 3]])
        rotboxs = polygonToRotRectangle_batch(polygons)
        print('rotboxs:', rotboxs)

    def test_roi2droi(self):
        rois = torch.cuda.FloatTensor([
                             [0, -1, -2.5, 3, 4.5],
                             [0, 24.5, 68.0, 35.5, 112.0],
                             [1, 2, 4, 24.6, 8]
                             # [-9.8, 0.5, 13.8, 7.5],
                             ])
        expected_drois = torch.cuda.FloatTensor([
                            [0, 1, 1, 5.0, 8.0, -np.pi/2.],
                            [0, 30.0, 90.0, 12.0, 45.0, -np.pi/2.],
                            [1, 13.3, 6.0, 23.6, 5.0, -np.pi/2.]
                            ])
        drois = roi2droi(rois)

        np.testing.assert_almost_equal(drois.cpu().numpy(), expected_drois.cpu().numpy())


    def test_choose_best_Rroi_batch(self):
        Rrois = np.array([[3, 4, 2, 10, np.pi/6.],
                          [3, 4, 10, 2, np.pi/6. + np.pi/2.],
                          [3, 4, 2, 10, np.pi/6. + np.pi],
                          [3, 4, 10, 2, np.pi/6. + np.pi + np.pi/2.]])
        Rrois = torch.from_numpy(Rrois).float().cuda()
        results = choose_best_Rroi_batch(Rrois).cpu().numpy()
        expected_results = np.array([[3, 4, 10, 2, np.pi/6. + np.pi/2.],
                                     [3, 4, 10, 2, np.pi / 6. + np.pi / 2.],
                                     [3, 4, 10, 2, np.pi / 6. + np.pi / 2.],
                                     [3, 4, 10, 2, np.pi / 6. + np.pi / 2.]])

        np.testing.assert_almost_equal(results, expected_results, decimal=6)

    def test_best_match_dbbox2delta(self):
        boxlist1 = np.array([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi - np.pi/10.],
                                  [1, 1, 10, 5, np.pi - np.pi/10.]
                                  ])
        boxlist2 = np.array([[1, 1, 5, 10, -np.pi/10. + np.pi/2.],
                                  [1, 1, 10, 5, np.pi/10 + np.pi],
                                  [1, 1, 5, 10, np.pi - np.pi/10. - np.pi/20. - np.pi/2.],
                                  [1, 1, 10, 5, np.pi - np.pi/10. - np.pi/20. + 10 * np.pi]
                                  ])
        norm = np.pi / 2.
        expected_results = np.array([[0, 0, 0, 0, -np.pi/10./norm],
                                     [0, 0, 0, 0, np.pi/10./norm],
                                     [0, 0, 0, 0, -np.pi/20./norm],
                                     [0, 0, 0, 0, -np.pi/20./norm]])

        boxlist1 = torch.from_numpy(boxlist1).float().cuda()
        boxlist2 = torch.from_numpy(boxlist2).float().cuda()

        results = best_match_dbbox2delta(boxlist1, boxlist2)
        print('results: ', results)

        old_resutls = dbbox2delta(boxlist1, boxlist2)
        print('old_resutls: ', old_resutls)

        np.testing.assert_almost_equal(results.cpu().numpy(), expected_results, decimal=6)


        print('test decode')
        predict1 = delta2dbbox_v2(boxlist1, results)

        predict2 = delta2dbbox(boxlist1, old_resutls)

        diff = best_match_dbbox2delta(predict1, predict2)
        print('predict1:', predict1)
        print('predict2:', predict2)
        print('diff:', diff)

    def test_dbbox2delta_v3(self):
        boxlist1 = torch.FloatTensor([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 100, 60, 34, np.pi/2]
                                  ])
        # the boxlist2(ground truths) are restricted to (0, 2 * pi)
        boxlist2 = torch.FloatTensor([[1, 1, 5, 8, np.pi/16],
                                  [1, 1, 5, 8, np.pi/16 + np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 90, 12, 45, np.pi/10]
                                  ])
        expected_targets = dbbox2delta_v3(boxlist1, boxlist2)
        expected_boxlist2 = delta2dbbox_v3(boxlist1, expected_targets)

        np.testing.assert_almost_equal(expected_boxlist2.numpy(), boxlist2.numpy(), decimal=5)

    def test_RotBox2Polys_torch(self):
        boxlist1 = torch.FloatTensor([[1, 1, 10, 5, 0],
                                  [1, 1, 10, 5, np.pi/10],
                                  [1, 1, 10, 5, 0],
                                  [30, 100, 60, 34, np.pi/2]
                                  ])
        outs1 = RotBox2Polys(boxlist1.cpu().numpy())
        outs2 = RotBox2Polys_torch(boxlist1).cpu().numpy()

        np.testing.assert_almost_equal(outs1, outs2, decimal=6)


if __name__ == '__main__':
    unittest.main()