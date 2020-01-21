import mmcv
import numpy as np
import pycocotools.mask as mask_utils
# from mmdet.datasets import get_dataset

import cv2

# get dataset

import os
# from xx import *
def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

# def RotBox2Polys(dboxes):
#     """
#     :param dboxes: (x_ctr, y_ctr, w, h, angle)
#         (numboxes, 5)
#     :return: quadranlges:
#         (numboxes, 8)
#     """
#     cs = np.cos(dboxes[:, 4])
#     ss = np.sin(dboxes[:, 4])
#     w = dboxes[:, 2] - 1
#     h = dboxes[:, 3] - 1
#
#     ## change the order to be the initial definition
#     x_ctr = dboxes[:, 0]
#     y_ctr = dboxes[:, 1]
#     x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
#     x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
#     x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
#     x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)
#
#     y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
#     y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
#     y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
#     y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)
#
#     x1 = x1[:, np.newaxis]
#     y1 = y1[:, np.newaxis]
#     x2 = x2[:, np.newaxis]
#     y2 = y2[:, np.newaxis]
#     x3 = x3[:, np.newaxis]
#     y3 = y3[:, np.newaxis]
#     x4 = x4[:, np.newaxis]
#     y4 = y4[:, np.newaxis]
#
#     polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)
#     return polys

def seg2poly_old(rle):
    # TODO: debug for this function
    """
    This function transform a single encoded RLE to a single poly
    :param seg: RlE
    :return: poly
    """
    # try:
    #     binary_mask = mask_utils.decode(rle)
    #     contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL)
    #     contour_lens = np.array(list(map(len, contours)))
    #     max_id = contour_lens.argmax()
    #     max_contour = contours[max_id]
    #     rect = cv2.minAreaRect(max_contour)
    #     poly = cv2.boxPoints(rect)
    #     return poly
    # except:
    #     return -1
    try:
        binary_mask = mask_utils.decode(rle)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # len is not appropriate
        # contour_lens = np.array(list(map(len, contours)))
        # max_id = contour_lens.argmax()
        contour_areas = np.array(list(map(cv2.contourArea, contours)))
        max_id = contour_areas.argmax()
        max_contour = contours[max_id]
        rect = cv2.minAreaRect(max_contour)
        poly = cv2.boxPoints(rect)
        poly = TuplePoly2Poly(poly)
        return poly
    except:
        return []

def seg2poly(rle):
    # TODO: debug for this function
    """
    This function transform a single encoded RLE to a single poly
    :param seg: RlE
    :return: poly
    """
    # try:
    #     binary_mask = mask_utils.decode(rle)
    #     contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL)
    #     contour_lens = np.array(list(map(len, contours)))
    #     max_id = contour_lens.argmax()
    #     max_contour = contours[max_id]
    #     rect = cv2.minAreaRect(max_contour)
    #     poly = cv2.boxPoints(rect)
    #     return poly
    # except:
    #     return -1
    try:
        binary_mask = mask_utils.decode(rle)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # len is not appropriate
        # contour_lens = np.array(list(map(len, contours)))
        # max_id = contour_lens.argmax()
        # contour_areas = np.array(list(map(cv2.contourArea, contours)))
        # max_id = contour_areas.argmax()
        # max_contour = contours[max_id]
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        poly = cv2.boxPoints(rect)
        poly = TuplePoly2Poly(poly)
        return poly
    except:
        return []
# def box2poly(boxes):
#     """
#     :param boxes: (x, y, w, h)  [n, 4]
#     :return: (x1, y1, ... x4, y4) [n, 8]
#     """
#     xs = boxes[:, 0]
#     ys = boxes[:, 1]
#     ws = boxes[:, 2]
#     hs = boxes[:, 3]
#     n = len(xs)
#     polys = np.zeros((n, 8))
#     polys[:, 0] = xs - ws/2.0
#     polys[:, 1] = ys - hs/2.0
#     polys[:, 2] = xs + ws/2.0
#     polys[:, 3] = ys - hs/2.0
#     polys[:, 4] = xs + ws/2.0
#     polys[:, 5] = ys + hs/2.0
#     polys[:, 6] = xs - ws/2.0
#     polys[:, 7] = ys + hs/2.0
#
#     return polys

# def xy2wh(boxes):
#     """
#     :param boxes: (xmin, ymin, xmax, ymax) (n,4)
#     :return: out_boxes: (x_ctr, y_ctr, w, h) (n, 4)
#     """
#     num_boxes = boxes.shape[0]
#
#     ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
#     ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
#     ex_ctr_x = boxes[:, 0] + 0.5 * (ex_widths - 1.0)
#     ex_ctr_y = boxes[:, 1] + 0.5 * (ex_heights - 1.0)
#
#     return np.concatenate((ex_ctr_x[:, np.newaxis], ex_ctr_y[:, np.newaxis], ex_widths[:, np.newaxis], ex_heights[:, np.newaxis]), axis=1)

def OBBDet2Comp4(dataset, results):
    results_dict = {}
    for idx in range(len(dataset)):
        filename = dataset.img_infos[idx]['filename']
        result = results[idx]
        for label in range(len(result)):
            rbboxes = result[label]
            cls_name = dataset.CLASSES[label]
            polys = RotBox2Polys(rbboxes[:, :-1])
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(rbboxes.shape[0]):
                poly = polys[i]
                score = float(rbboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                results_dict[cls_name].append(outline)
    return results_dict

def OBBDetComp4(dataset, results):
    results_dict = {}
    for idx in range(len(dataset)):
        filename = dataset.img_infos[idx]['filename']
        result = results[idx]
        for label in range(len(result)):
            rbboxes = result[label]
            # import pdb
            # pdb.set_trace()
            cls_name = dataset.CLASSES[label]
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(rbboxes.shape[0]):
                poly = rbboxes[i][:-1]
                score = float(rbboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                results_dict[cls_name].append(outline)
    return results_dict

def HBBDet2Comp4(dataset, results):
    results_dict = {}
    for idx in range(len(dataset)):
        # print('idx: ', idx, 'total: ', len(dataset))
        filename = dataset.img_infos[idx]['filename']
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]

            # try:
            cls_name = dataset.CLASSES[label]
            # except:
            #     import pdb
            #     pdb.set_trace()
            if cls_name not in results_dict:
                results_dict[cls_name] = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                score = float(bboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, bbox))
                results_dict[cls_name].append(outline)
    return results_dict


def HBBSeg2Comp4(dataset, results):
    hbb_results_dict = {}
    obb_results_dict = {}
    prog_bar = mmcv.ProgressBar(len(dataset))
    for idx in range(len(dataset)):
        # import pdb
        # pdb.set_trace()
        filename = dataset.img_infos[idx]['filename']
        # print('filename: ', filename)
        det, seg = results[idx]
        for label in range(len(det)):
            bboxes = det[label]
            segms = seg[label]
            cls_name = dataset.CLASSES[label]
            if cls_name not in hbb_results_dict:
                hbb_results_dict[cls_name] = []
            if cls_name not in obb_results_dict:
                obb_results_dict[cls_name] = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i].tolist()
                score = float(bboxes[i][-1])
                hbb_outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, bbox))
                hbb_results_dict[cls_name].append(hbb_outline)

                poly = seg2poly(segms[i])
                if poly != []:
                    score = float(bboxes[i][-1])
                    obb_outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                    obb_results_dict[cls_name].append(obb_outline)
        prog_bar.update()
    return hbb_results_dict, obb_results_dict

def HBBOBB2Comp4(dataset, results):
    hbb_results_dict = {}
    obb_results_dict = {}
    for idx in range(len(dataset)):
        filename = dataset.img_infos[idx]['filename']

        hbb_det, obb_det = results[idx]

        for label in range(len(hbb_det)):
            bboxes = hbb_det[label]
            rbboxes = obb_det[label]
            cls_name = dataset.CLASSES[label]
            if cls_name not in hbb_results_dict:
                hbb_results_dict[cls_name] = []
            if cls_name not in obb_results_dict:
                obb_results_dict[cls_name] = []
            # parse hbb results
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                score = float(bboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, bbox))
                hbb_results_dict[cls_name].append(outline)
            # parse obb results
            for i in range(rbboxes.shape[0]):
                poly = rbboxes[i]
                score = float(rbboxes[i][-1])
                outline = filename + ' ' + str(score) + ' ' + ' '.join(map(str, poly))
                obb_results_dict[cls_name].append(outline)
    return hbb_results_dict, obb_results_dict












