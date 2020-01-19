import torch
from bbox import bbox_overlaps_cython
# from bbox_v2 import bbox_overlaps_cython_v2
import numpy as np
import DOTA_devkit.polyiou as polyiou
from mmdet.core.bbox.transforms_rbbox import RotBox2Polys, poly2bbox, mask2poly, Tuplelist2Polylist

def bbox_overlaps_cy(boxes, query_boxes):
    box_device = boxes.device
    boxes_np = boxes.cpu().numpy().astype(np.float)
    query_boxes_np = query_boxes.cpu().numpy().astype(np.float)
    ious = bbox_overlaps_cython(boxes_np, query_boxes_np)
    return torch.from_numpy(ious).to(box_device)

# def bbox_overlaps_cy2(boxes, query_boxes):
#     box_device = boxes.device
#     boxes_np = boxes.cpu().numpy().astype(np.float)
#     query_boxes_np = query_boxes.cpu().numpy().astype(np.float)
#     ious = bbox_overlaps_cython_v2(boxes_np, query_boxes_np)
#     return torch.from_numpy(ious).to(box_device)

def bbox_overlaps_np(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    box_device = boxes.device
    boxes = boxes.cpu().numpy().astype(np.float)
    query_boxes = query_boxes.cpu().numpy().astype(np.float)

    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return torch.from_numpy(overlaps).to(box_device)

# def bbox_overlaps_torch_v2(anchors, gt_boxes):
#     """
#     anchors: (N, 4) ndarray of float
#     gt_boxes: (K, 4) ndarray of float
#
#     overlaps: (N, K) ndarray of overlap between boxes and query_boxes
#     """
#     N = anchors.size(0)
#     K = gt_boxes.size(0)
#
#     gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
#                 (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)
#
#     anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
#                 (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)
#
#     boxes = anchors.view(N, 1, 4).expand(N, K, 4)
#     query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)
#
#     iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
#         torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
#     iw[iw < 0] = 0
#
#     ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
#         torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
#     ih[ih < 0] = 0
#
#     ua = anchors_area + gt_boxes_area - (iw * ih)
#     overlaps = iw * ih / ua
#
#     return overlaps

def bbox_overlaps_np_v2(bboxes1, bboxes2):
    """
    :param bboxes1: (N, 4) ndarray of float
    :param bboxes2: (K, 4) ndarray of float
    :return: overlaps (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = bboxes1.shape[0]
    K = bboxes2.shape[0]

    area2 = ((bboxes2[:, 2] - bboxes2[:, 0] + 1) *
             (bboxes2[:, 3] - bboxes2[:, 1] + 1))[np.newaxis, :]

    area1 = ((bboxes1[:, 2] - bboxes1[:, 0] + 1) *
             (bboxes1[:, 3] - bboxes1[:, 1]))[:, np.newaxis]

    bboxes2 = bboxes2[np.newaxis, :, :]

    bboxes1 = bboxes1[:, np.newaxis, :]

    iw = np.minimum(bboxes1[:, :, 2], bboxes2[:, :, 2]) - \
         np.maximum(bboxes1[:, :, 0], bboxes2[:, :, 0]) + 1

    iw[iw < 0] = 0

    ih = np.minimum(bboxes1[:, :, 3], bboxes2[:, :, 3]) - \
         np.maximum(bboxes1[:, :, 1], bboxes2[:, :, 1]) + 1

    ih[ih < 0] = 0

    ua = area1 + area2 - (iw * ih)

    overlaps = iw * ih / ua

    return overlaps

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']
    # import pdb
    # pdb.set_trace()
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious

def bbox_overlaps_np_v3(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """

    :param bboxes1: (ndarray): shape (m, 4)
    :param bboxes2: (ndarray): shape (n, 4)
    :param mode: (str) : "iou" or "iof"
    :param is_aligned: (ndarray): shape (m, n) if is_aligned == False else shape (m, 1)
    :return:
    """

    assert mode in ['iou', 'iof']

    box_device = bboxes1.device
    bboxes1 = bboxes1.cpu().numpy().astype(np.float)
    bboxes2 = bboxes2.cpu().numpy().astype(np.float)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return np.random.rand(rows, 1).astype(bboxes1.dtype) if is_aligned \
            else np.random.rand(rows, cols).astype(bboxes1.dtype)

    if is_aligned:
        lt = np.maximum(bboxes1[:, :2], bboxes2[:, :2])
        rb = np.minimum(bboxes1[:, 2:], bboxes2[:, 2:])

        wh = np.clip(rb - lt + 1, a_min=0, a_max=None)
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = np.clip(rb - lt + 1, a_min=0, a_max=None) # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])
    ious = torch.from_numpy(ious).to(box_device)
    return ious








def bbox_overlaps_fp16(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """
    The fp16 version exist some bugs
    Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    bboxes1_fp16 = bboxes1.half()/100.
    bboxes2_fp16 = bboxes2.half()/100.

    rows = bboxes1_fp16.size(0)
    cols = bboxes2_fp16.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1_fp16.new(rows, 1) if is_aligned else bboxes1_fp16.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1_fp16[:, :2], bboxes2_fp16[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1_fp16[:, 2:], bboxes2_fp16[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1_fp16[:, 2] - bboxes1_fp16[:, 0] + 1) * (
            bboxes1_fp16[:, 3] - bboxes1_fp16[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2_fp16[:, 2] - bboxes2_fp16[:, 0] + 1) * (
                bboxes2_fp16[:, 3] - bboxes2_fp16[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1_fp16[:, None, :2], bboxes2_fp16[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1_fp16[:, None, 2:], bboxes2_fp16[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1_fp16[:, 2] - bboxes1_fp16[:, 0] + 1) * (
            bboxes1_fp16[:, 3] - bboxes1_fp16[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2_fp16[:, 2] - bboxes2_fp16[:, 0] + 1) * (
                bboxes2_fp16[:, 3] - bboxes2_fp16[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious.float()

def mask_overlaps():
    pass

# def bbox_overlaps_cy(boxes, query_boxes):
#     box_device = boxes.device
#     boxes_np = boxes.cpu().numpy().astype(np.float)
#     query_boxes_np = query_boxes.cpu().numpy().astype(np.float)
#     ious = bbox_overlaps_cython(boxes_np, query_boxes_np)
#     return torch.from_numpy(ious).to(box_device)

def rbbox_overlaps_cy_warp(rbboxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
    # import pdb
    # pdb.set_trace()
    box_device = query_boxes.device
    query_boxes_np = query_boxes.cpu().numpy().astype(np.float)

    # polys_np = RotBox2Polys(boxes_np)
    # TODO: change it to only use pos gt_masks
    # polys_np = mask2poly(gt_masks)
    # polys_np = np.array(Tuplelist2Polylist(polys_np)).astype(np.float)

    polys_np = RotBox2Polys(rbboxes).astype(np.float)
    query_polys_np = RotBox2Polys(query_boxes_np)

    h_bboxes_np = poly2bbox(polys_np)
    h_query_bboxes_np = poly2bbox(query_polys_np)

    # hious
    ious = bbox_overlaps_cython(h_bboxes_np, h_query_bboxes_np)
    import pdb
    # pdb.set_trace()
    inds = np.where(ious > 0)
    for index in range(len(inds[0])):
        box_index = inds[0][index]
        query_box_index = inds[1][index]

        box = polys_np[box_index]
        query_box = query_polys_np[query_box_index]

        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        ious[box_index][query_box_index] = overlap

    return torch.from_numpy(ious).to(box_device)

def rbbox_overlaps_hybrid(boxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, use the gpu_overlaps to calculate the obb overlaps
    # box_device = boxes.device
    pass

def rbbox_overlaps_cy(boxes_np, query_boxes_np):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps

    polys_np = RotBox2Polys(boxes_np).astype(np.float)
    query_polys_np = RotBox2Polys(query_boxes_np).astype(np.float)

    h_bboxes_np = poly2bbox(polys_np).astype(np.float)
    h_query_bboxes_np = poly2bbox(query_polys_np).astype(np.float)

    # hious
    ious = bbox_overlaps_cython(h_bboxes_np, h_query_bboxes_np)
    import pdb
    # pdb.set_trace()
    inds = np.where(ious > 0)
    for index in range(len(inds[0])):
        box_index = inds[0][index]
        query_box_index = inds[1][index]

        box = polys_np[box_index]
        query_box = query_polys_np[query_box_index]

        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        ious[box_index][query_box_index] = overlap

    return ious

