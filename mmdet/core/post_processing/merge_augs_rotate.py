import torch

import numpy as np

from mmdet.ops import nms
from ..bbox import bbox_mapping_back, bbox_rotate_mapping, \
    dbbox_rotate_mapping, dbbox_mapping_back
from mmdet.core import choose_best_Rroi_batch


def merge_rotate_aug_proposals(aug_proposals, img_metas, rpn_test_cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 5). Note that they are not rescaled to the
            original image size.
        img_metas (list[dict]): image info including "shape_scale" and "flip".
        rpn_test_cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 4), proposals corresponding to original image scale.
    """
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                              scale_factor, flip)
        # rotation mapping
        angle = img_info['angle']
        if angle != 0:
            # TODO: check the angle
            _proposals[:, :4] = bbox_rotate_mapping(_proposals[:, :4], img_shape, -angle)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals, rpn_test_cfg.nms_thr)
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(rpn_test_cfg.max_num, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals


def merge_rotate_aug_bboxes(aug_dbboxes, aug_scores, img_metas, rcnn_test_cfg, index=None):
    """Merge augmented detection dbboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 5*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_dbboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        bboxes = dbbox_mapping_back(bboxes, img_shape, scale_factor, flip)
        # TODO: check the angle
        angle = img_info[0]['angle']
        if angle != 0:
            bboxes = dbbox_rotate_mapping(bboxes, img_shape, -angle)
        recovered_bboxes.append(bboxes)
    # import pdb; pdb.set_trace()
    # bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if index != None:
        bboxes = recovered_bboxes[index]
    else:
        bboxes = torch.cat(recovered_bboxes, 0)
    if aug_scores is None:
        return bboxes
    else:
        # scores = torch.stack(aug_scores).mean(dim=0)
        if index != None:
            scores = aug_scores[index]
        else:
            scores = torch.cat(aug_scores, 0)
        return bboxes, scores


def merge_aug_scores(aug_scores):
    """Merge augmented bbox scores."""
    if isinstance(aug_scores[0], torch.Tensor):
        return torch.mean(torch.stack(aug_scores), dim=0)
    else:
        return np.mean(aug_scores, axis=0)