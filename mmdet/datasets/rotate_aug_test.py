import unittest
import numpy as np
from rotate_aug import *
import torch
import copy

class Test_rotate_aug(unittest.TestCase):

    @unittest.skip('skip')
    def test_poly2mask(self):
        polys = [[[77.8, 51.1, 82.4, 51.1, 82.4, 55.8, 77.8, 55.8]],
                          [[20, 21, 30, 21, 30, 30, 20, 30]],
                 [[490., 561., 490., 817., 234., 817., 234., 561.]] # TODO: figure out why this poly are transfered to np.zeros(1000, 1000)
                 ]
        h, w = 1000, 1000

        masks = poly2mask(polys, h, w)

        print('masks: ', masks)
        # print('sum masks: ', sum(masks))
        import pdb
        pdb.set_trace()
        # np.set_printoptions(threshold=np.inf)
        # print('masks: ', np.array(masks))
        polysdecode = mask2poly(np.array(masks))

        print('polysdecode: ', polysdecode)
        # np.testing.assert_almost_equal(np.array(polys), np.array(polysdecode))

    def test_poly2mask2(self):
        polys = [[[490., 561., 490., 817., 234., 817., 234., 561.]],
                [[533., 462., 533., 206., 789., 206., 789., 462.]],
                 [[561., 533., 817., 533., 817., 789., 561., 789.]]]
        h, w = 1000, 1000
        masks = poly2mask(polys, h, w)
        print('mask: ', masks)
        print('sum mask 1: ', sum(sum(masks[0])))
        print('sum mask 2: ', sum(sum(masks[1])))

        polysdecode = mask2poly(np.array(masks))
        print('polysdecode: ', polysdecode)

if __name__ == '__main__':
    unittest.main()

