import torch
from mmdet.ops.nms.rnms_wrapper import py_cpu_nms_poly_fast
from mmdet.ops.nms import rnms_wrapper
from mmdet.ops.poly_nms import poly_nms_wrapper
from mmdet.core import RotBox2Polys, RotBox2Polys_torch
import time
# TODO: refator the code
# TODO: debug for testing cancel the nms
DEBUG = False
def multiclass_nms_rbbox(multi_bboxes,
                         multi_scores,
                         score_thr,
                         nms_cfg,
                         max_num=-1,
                         score_factors=None):
    """
    NMS for multi-class bboxes.
    :param multi_bboxes:
    :param multi_scores:
    :param score_thr:
    :param nms_cfg:
    :param max_num:
    :param score_factors:
    :return:
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    # nms_type = nms_cfg_.pop('type', 'nms')
    # nms_op = py_cpu_nms_poly_fast

    nms_type = nms_cfg_.pop('type', 'nms')
    # TODO: refactor it
    if nms_type == 'poly_nms':
        nms_op = getattr(poly_nms_wrapper, nms_type)
    else:
        nms_op = getattr(rnms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 5:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 5: (i + 1) * 5]
        _bboxes = torch.from_numpy(RotBox2Polys(_bboxes.cpu().numpy())).to(multi_scores.device)
        # _bboxes = RotBox2Polys_torch(_bboxes)
        # _bboxes = RotBox2Polys_torch(_bboxes.cpu()).to(multi_scores.device)
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        # TODO: figure out the nms_cfg
        if not DEBUG:
            # start = time.clock()
            cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
            # elapsed = (time.clock() - start)
            # print("Time used:", elapsed)
        # import pdb
        # pdb.set_trace()
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]

    else:
        bboxes = multi_bboxes.new_zeros((0, 9))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels

def Pesudomulticlass_nms_rbbox(multi_bboxes,
                         multi_scores,
                         score_thr,
                         # nms_cfg,
                         max_num=-1,
                         score_factors=None):
    """
    NMS for multi-class bboxes.
    :param multi_bboxes:
    :param multi_scores:
    :param score_thr:
    :param nms_cfg:
    :param max_num:
    :param score_factors:
    :return:
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    # nms_cfg_ = nms_cfg.copy()
    # nms_type = nms_cfg_.pop('type', 'nms')
    # nms_op = py_cpu_nms_poly_fast

    # nms_type = nms_cfg_.pop('type', 'nms')
    # nms_op = getattr(rnms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 5:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 5: (i + 1) * 5]
        _bboxes = torch.from_numpy(RotBox2Polys(_bboxes.cpu().numpy())).to(multi_scores.device)
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        # TODO: figure out the nms_cfg
        # if not DEBUG:
        #     cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        # import pdb
        # pdb.set_trace()
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]

    else:
        bboxes = multi_bboxes.new_zeros((0, 9))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels










