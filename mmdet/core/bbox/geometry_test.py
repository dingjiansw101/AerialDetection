import unittest
import numpy as np
from geometry import *
import copy
import torch
import time
from DOTA_devkit.poly_nms_gpu.poly_overlaps import poly_overlaps
from shapely.geometry import Polygon
import shapely
from mmdet.core.evaluation import RotBox2Polys
import datetime
import time

def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            # union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 1
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

class Test_geometry(unittest.TestCase):
    def setUp(self):
        self.num_bboxes1 = 200
        self.num_bboxes2 = 20000
        self.image_width = 1024
        self.image_height = 1024

        self.bboxes1 = np.zeros([self.num_bboxes1, 4])
        self.bboxes1[:, [0, 2]] = np.sort(np.random.rand(self.num_bboxes1, 2) * (self.image_width - 1))
        self.bboxes1[:, [1, 3]] = np.sort(np.random.rand(self.num_bboxes1, 2) * (self.image_height - 1))

        self.bboxes2 = np.zeros([self.num_bboxes2, 4])
        self.bboxes2[:, [0, 2]] = np.sort(np.random.rand(self.num_bboxes2, 2) * (self.image_width - 1))
        self.bboxes2[:, [1, 3]] = np.sort(np.random.rand(self.num_bboxes2, 2) * (self.image_height - 1))

        self.bboxes1_tensor = torch.from_numpy(self.bboxes1)
        self.bboxes2_tensor = torch.from_numpy(self.bboxes2)

        start = time.perf_counter()
        self.ious = bbox_overlaps(self.bboxes1_tensor, self.bboxes2_tensor).numpy()
        elapsed = (time.perf_counter() - start)
        print('bbox_overlaps time: ', elapsed)

    @unittest.skip("have benn tested")
    def test_bbox_overlaps_cy(self):

        start = time.perf_counter()
        # ious_cy = bbox_overlaps_cy(self.bboxes1, self.bboxes2)
        ious_cy = bbox_overlaps_cy(self.bboxes1_tensor, self.bboxes2_tensor)
        self.assertTrue(type(ious_cy) == torch.Tensor)
        ious_cy = ious_cy.numpy()
        elapsed = (time.perf_counter() - start)
        print('cython time: ', elapsed)
        np.testing.assert_array_almost_equal(self.ious, ious_cy)

    @unittest.skip("have benn tested")
    def test_bbox_overlaps_cy2(self):

        start = time.perf_counter()
        ious_cy2 = bbox_overlaps_cy2(self.bboxes1_tensor, self.bboxes2_tensor).numpy()
        elapsed = (time.perf_counter() - start)
        print('cython 2 time: ', elapsed)
        np.testing.assert_array_almost_equal(self.ious, ious_cy2)

    @unittest.skip("have benn tested")
    def test_bbox_overlaps_np(self):

        start = time.perf_counter()
        ious_np = bbox_overlaps_np(self.bboxes1_tensor, self.bboxes2_tensor).numpy()
        elapsed = (time.perf_counter() - start)
        print('numpy time: ', elapsed)
        np.testing.assert_array_almost_equal(self.ious, ious_np)

    # def test_bbox_overlaps_np_v2(self):
    #
    #     start = time.perf_counter()
    #     ious_np_v2 = bbox_overlaps_np_v2(self.bboxes1, self.bboxes2)
    #     elapsed = (time.perf_counter() - start)
    #     print('numpy v2 time: ', elapsed)
    #     np.testing.assert_array_almost_equal(self.ious, ious_np_v2)
    @unittest.skip("have benn tested")
    def test_bbox_overlaps_np_v3(self):

        start = time.perf_counter()
        ious_np_v3 = bbox_overlaps_np_v3(self.bboxes1_tensor, self.bboxes2_tensor).numpy()
        elapsed = (time.perf_counter() - start)
        print('numpy v3 time: ', elapsed)
        np.testing.assert_array_almost_equal(self.ious, ious_np_v3)

    # def test_bbox_overlaps(self):
    #     pass

    # def test_bbox_overlaps_fp16(self):
    #
    #     ious_fp16 = bbox_overlaps_fp16(torch.from_numpy(self.bboxes1).to(0), torch.from_numpy(self.bboxes2).to(0)).numpy()
    #     np.testing.assert_array_almost_equal(self.ious, ious_fp16)
    @unittest.skip("have been tested")
    def test_rbbox_overlaps_cy(self):
        num_boxes=2000
        num_query_boxes=1000
        xs = np.random.rand(num_boxes) * 100
        ys = np.random.rand(num_boxes) * 100
        ws = np.random.rand(num_boxes) * 1000
        hs = np.random.rand(num_boxes) * 1000
        Theta = np.random.rand(num_boxes) * 2 * np.pi

        boxlist1 = np.concatenate((xs[:, np.newaxis],
                                   ys[:, np.newaxis],
                                   ws[:, np.newaxis],
                                   hs[:, np.newaxis],
                                   Theta[:, np.newaxis]), axis=1).astype(np.float32)

        xs2 = np.random.rand(num_query_boxes) * 100
        ys2 = np.random.rand(num_query_boxes) * 100
        ws2 = np.random.rand(num_query_boxes) * 1000
        hs2 = np.random.rand(num_query_boxes) * 1000
        Theta2 = np.random.rand(num_query_boxes) * 2 * np.pi

        boxlist2 = np.concatenate((xs2[:, np.newaxis],
                                   ys2[:, np.newaxis],
                                   ws2[:, np.newaxis],
                                   hs2[:, np.newaxis],
                                   Theta2[:, np.newaxis]), axis=1).astype(np.float32)

        polys1 = RotBox2Polys(boxlist1).astype(np.float32)
        polys2 = RotBox2Polys(boxlist2).astype(np.float32)

        expected_results = np.zeros((num_boxes, num_query_boxes))
        # expected_results = polygon_iou(polys1, polys2)

        # shapely library
        start_time = datetime.datetime.now()
        for i in range(num_boxes):
            for j in range(num_query_boxes):
                expected_results[i, j] = polygon_iou(polys1[i], polys2[j])

        end_time = datetime.datetime.now()
        interval = (end_time - start_time)
        print('shapely library poly overlaps time: ', interval)

        ## Test cpu overlaps
        boxlist1_ts = torch.from_numpy(boxlist1).cuda()
        boxlist2_ts = torch.from_numpy(boxlist2).cuda()
        start_time3 = datetime.datetime.now()
        ious_cy_warp = rbbox_overlaps_cy_warp(boxlist1_ts, boxlist2_ts)
        end_time3 = datetime.datetime.now()
        interval3 = (end_time3 - start_time3)
        print('cython poly overlaps warp time: ', interval3)

        np.testing.assert_allclose(expected_results, ious_cy_warp.cpu().numpy(), rtol=1e-3, atol=1e-2)

        ## Test cpu overlaps
        start_time4 = datetime.datetime.now()
        ious_cy = rbbox_overlaps_cy(boxlist1, boxlist2)
        end_time4 = datetime.datetime.now()
        interval4 = (end_time4 - start_time4)
        print('cython poly overlaps time: ', interval4)

        np.testing.assert_allclose(expected_results, ious_cy, rtol=1e-3, atol=1e-2)

        # Test gpu overlaps

        start_time2 = datetime.datetime.now()
        results = poly_overlaps(boxlist1, boxlist2)
        end_time2 = datetime.datetime.now()
        interval2 = (end_time2 - start_time2)
        print('gpu poly overlaps time: ', interval2)

        # np.testing.assert_allclose(expected_results, results, rtol=1e-2, atol=1e-2)
    def test_rbbox_overlaps_cy_compare_time(self):
        num_boxes=2000
        num_query_boxes=1000
        xs = np.random.rand(num_boxes) * 1000
        ys = np.random.rand(num_boxes) * 1000
        ws = np.random.rand(num_boxes) * 300
        hs = np.random.rand(num_boxes) * 300
        Theta = np.random.rand(num_boxes) * 2 * np.pi

        boxlist1 = np.concatenate((xs[:, np.newaxis],
                                   ys[:, np.newaxis],
                                   ws[:, np.newaxis],
                                   hs[:, np.newaxis],
                                   Theta[:, np.newaxis]), axis=1).astype(np.float32)

        xs2 = np.random.rand(num_query_boxes) * 1000
        ys2 = np.random.rand(num_query_boxes) * 1000
        ws2 = np.random.rand(num_query_boxes) * 300
        hs2 = np.random.rand(num_query_boxes) * 300
        Theta2 = np.random.rand(num_query_boxes) * 2 * np.pi

        boxlist2 = np.concatenate((xs2[:, np.newaxis],
                                   ys2[:, np.newaxis],
                                   ws2[:, np.newaxis],
                                   hs2[:, np.newaxis],
                                   Theta2[:, np.newaxis]), axis=1).astype(np.float32)

        ## Test cpu overlaps
        print('start convert to cuda')
        boxlist1_ts = torch.from_numpy(boxlist1).cuda()
        boxlist2_ts = torch.from_numpy(boxlist2).cuda()
        start_time3 = datetime.datetime.now()
        ious_cy_warp = rbbox_overlaps_cy_warp(boxlist1_ts, boxlist2_ts)
        end_time3 = datetime.datetime.now()
        interval3 = (end_time3 - start_time3)
        print('cython poly overlaps warp time: ', interval3)

        # np.testing.assert_allclose(expected_results, ious_cy_warp.cpu().numpy(), rtol=1e-3, atol=1e-2)

        ## Test cpu overlaps
        start_time4 = datetime.datetime.now()
        ious_cy = rbbox_overlaps_cy(boxlist1, boxlist2)
        end_time4 = datetime.datetime.now()
        interval4 = (end_time4 - start_time4)
        print('cython poly overlaps time: ', interval4)

        # np.testing.assert_allclose(expected_results, ious_cy, rtol=1e-3, atol=1e-2)

        # Test gpu overlaps

        start_time2 = datetime.datetime.now()
        results = poly_overlaps(boxlist1, boxlist2)
        end_time2 = datetime.datetime.now()
        interval2 = (end_time2 - start_time2)
        print('gpu poly overlaps time: ', interval2)
if __name__ == '__main__':
    unittest.main()
