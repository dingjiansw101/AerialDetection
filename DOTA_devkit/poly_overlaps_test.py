from poly_nms_gpu.poly_overlaps import poly_overlaps
import numpy as np

if __name__ == '__main__':
    ## TODO: improve the precision, the results seems like a little diffrerent from polyiou.cpp
    # , may caused by use float not double.
    anchors = np.array([
        [1, 1, 2, 10, 0],
        #                           [1, 30, 3, 1, np.pi/16],
        #                           [1000, 1000, 60, 60, 0],

                        ],
                       dtype=np.float32)
    anchors = np.repeat(anchors, 10000, axis=0)
    gt_boxes = np.array([
        [2, 1, 2, 10, 0],
        #                           [1, 30, 3, 1, np.pi/16 + np.pi/2],
        #                           [1010, 1010, 3, 3, 0],

                         ], dtype=np.float32)
    gt_boxes = np.repeat(gt_boxes, 10000, axis=0)
    # anchors = np.array([[1, 1, 200, 100, 0]],
    #                    dtype=np.float32)
    # gt_boxes = np.array([[2, 1, 200, 100, 0],
    #                      ], dtype=np.float32)
    # anchors = np.array([[1, 30, 3, 1, np.pi/16]],
    #                    dtype=np.float32)
    # gt_boxes = np.array([[1, 30, 3, 1, np.pi/16 + np.pi/2],
    #                      ], dtype=np.float32)
    overlaps = poly_overlaps(anchors, gt_boxes, 0)
    print(overlaps)

