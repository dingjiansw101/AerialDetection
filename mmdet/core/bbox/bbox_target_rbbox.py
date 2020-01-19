import torch

from .transforms_rbbox import dbbox2delta, delta2dbbox, \
    mask2poly, get_best_begin_point, polygonToRotRectangle_batch\
    , best_match_dbbox2delta, delta2dbbox_v3, dbbox2delta_v3, hbb2obb_v2
from ..utils import multi_apply


def bbox_target_rbbox(pos_bboxes_list,
                neg_bboxes_list,
                pos_assigned_gt_inds_list,
                gt_masks_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                concat=True,
                with_module=True,
                hbb_trans='hbb2obb_v2'):
    # import pdb
    # pdb.set_trace()
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_rbbox_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_assigned_gt_inds_list,
        gt_masks_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds,
        with_module=with_module,
        hbb_trans=hbb_trans)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_rbbox_single(pos_bboxes,
                       neg_bboxes,
                       pos_assigned_gt_inds,
                       gt_masks,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                       with_module=True,
                       hbb_trans='hbb2obb_v2'):
    """

    :param pos_bboxes: Tensor, shape (n, 4)
    :param neg_bboxes: Tensor, shape (m, 4)
    :param pos_assigned_gt_inds: Tensor, shape (n)
    :param gt_masks: numpy.ndarray, shape (n, 1024, 1024)
    :param pos_gt_labels:   Tensor, shape (n)
    :param cfg: dict, cfg.pos_weight = -1
    :param reg_classes: 16
    :param target_means:
    :param target_stds:
    :return:
    """
    # import pdb
    # pdb.set_trace()
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 5)
    pos_gt_masks = gt_masks[pos_assigned_gt_inds.cpu().numpy()]
    # TODO: optimizer it
    pos_gt_polys = mask2poly(pos_gt_masks)
    # if len(pos_gt_polys) == 0:
    #     import pdb
    #     pdb.set_trace()
    # print('pos_gt_polys: ', pos_gt_polys)
    pos_gt_bp_polys = get_best_begin_point(pos_gt_polys)
    # print('pos_gt_bp_polys: ', pos_gt_bp_polys)
    # TODO optimizer it
    # import pdb
    # pdb.set_trace()
    pos_gt_obbs = torch.from_numpy(polygonToRotRectangle_batch(pos_gt_bp_polys, with_module)).to(pos_bboxes.device)
    # print('pos_gt_obbs: ', pos_gt_obbs)
    if pos_bboxes.size(1) == 4:
        # if hbb_trans == 'hbb2obb':
        #     pos_ext_bboxes = hbb2obb(pos_bboxes)
        # elif hbb_trans == 'hbbpolyobb':
        #     pos_ext_bboxes = hbbpolyobb(pos_bboxes)
        # elif hbb_trans == 'hbb2obb_v2':
        pos_ext_bboxes = hbb2obb_v2(pos_bboxes)
        # else:
        #     print('no such hbb trans method')
        #     raise Exception
    else:
        pos_ext_bboxes = pos_bboxes
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        # import pdb
        # print('in bbox_target_rbbox')
        # pdb.set_trace()
        if with_module:
            pos_bbox_targets = dbbox2delta(pos_ext_bboxes, pos_gt_obbs, target_means,
                                          target_stds)
        else:
            pos_bbox_targets = dbbox2delta_v3(pos_ext_bboxes, pos_gt_obbs, target_means,
                                              target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights

def rbbox_target_rbbox(pos_rbboxes_list,
                         neg_rbboxes_list,
                         pos_gt_rbboxes_list,
                         pos_gt_labels_list,
                         cfg,
                         reg_classes=1,
                         target_means=[.0, .0, .0, .0, 0],
                         target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                         concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        rbbox_target_rbbox_single,
        pos_rbboxes_list,
        neg_rbboxes_list,
        pos_gt_rbboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights

def rbbox_target_rbbox_single(pos_rbboxes,
                       neg_rbboxes,
                       pos_gt_rbboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]):
    """

    :param pos_bboxes:
    :param neg_bboxes:
    :param gt_masks:
    :param pos_gt_labels:
    :param cfg:
    :param reg_classes:
    :param target_means:
    :param target_stds:
    :return:
    """
    assert pos_rbboxes.size(1) == 5
    num_pos = pos_rbboxes.size(0)
    num_neg = neg_rbboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_rbboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_rbboxes.new_zeros(num_samples)
    bbox_targets = pos_rbboxes.new_zeros(num_samples, 5)
    bbox_weights = pos_rbboxes.new_zeros(num_samples, 5)

    # TODO optimizer it
    # import pdb
    # pdb.set_trace()

    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        # import pdb
        # pdb.set_trace()
        pos_bbox_targets = best_match_dbbox2delta(pos_rbboxes, pos_gt_rbboxes, target_means,
                                      target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights



def expand_target_rbbox(dbbox_targets, dbbox_weights, labels, num_classes):
    dbbox_targets_expand = dbbox_targets.new_zeros((dbbox_targets.size(0),
                                                    5 * num_classes))
    dbbox_weights_expand = dbbox_weights.new_zeros((dbbox_weights.size(0),
                                                    5 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 5, (labels[i] + 1) * 5
        dbbox_targets_expand[i, start: end] = dbbox_targets[i, :]
        dbbox_weights_expand[i, start: end] = dbbox_weights[i, :]
    return dbbox_targets_expand, dbbox_weights_expand
