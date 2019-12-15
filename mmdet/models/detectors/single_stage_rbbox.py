import torch.nn as nn

from .base_new import BaseDetectorNew
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result, dbbox2result

# TODO: make it more flexible to add hbb
@DETECTORS.register_module
class SingleStageDetectorRbbox(BaseDetectorNew):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetectorRbbox, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if bbox_head is not None:
            self.bbox_head = builder.build_head(bbox_head)
        if rbbox_head is not None:
            self.rbbox_head = builder.build_head(rbbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetectorRbbox, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_bbox:
            self.bbox_head.init_weights()
        if self.with_rbbox:
            self.rbbox_head.init_weights()

    def extract_feat(self, img):
        x =self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                img,
                img_metas,
                gt_bboxes,
                gt_masks,
                gt_labels,
                gt_bboxes_ignore=None):
        # print('in single stage rbbox')
        # import pdb
        # pdb.set_trace()
        x = self.extract_feat(img)

        losses = dict()

        if self.with_bbox:
            bbox_outs = self.bbox_head(x)
            bbox_loss_inputs = bbox_outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
            # TODO: make if flexible to add the bbox_head
            bbox_losses = self.bbox_head.loss(
                *bbox_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(bbox_losses)
        if self.with_rbbox:
            rbbox_outs = self.rbbox_head(x)
            rbbox_loss_inputs = rbbox_outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)
            rbbox_losses = self.rbbox_head.loss(
                *rbbox_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rbbox_losses)
        return losses

    # def simple_test(self, img, img_meta, rescale=False):
    #     # TODO: finish it
    #     x = self.extract_feat(img)
    #     if self.with_bbox:
    #         outs = self.bbox_head(x)
    #         bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    #         bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
    #         bbox_results = [
    #             bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #             for det_bboxes, det_labels in bbox_list
    #         ]
    #     if not self.with_rbbox:
    #         return bbox_results[0]
    #     else:
    #         rbbox_outs = self.rbbox_head(x)
    #         rbbox_inputs = rbbox_outs + (img_meta, self.test_cfg, rescale)
    #         rbbox_list =self.rbbox_head.get_bboxes(*rbbox_inputs)
    #         rbbox_results = [
    #             dbbox2result(det_rbboxes, det_labels, self.rbbox_head.num_classes)
    #             for det_rbboxes, det_labels in rbbox_list
    #         ]
    #         return bbox_results[0], rbbox_results[0]

    def simple_test(self, img, img_meta, rescale=False):
        # TODO: make if more flexible to add hbb and obb
        x = self.extract_feat(img)
        rbbox_outs = self.rbbox_head(x)
        rbbox_inputs = rbbox_outs + (img_meta, self.test_cfg, rescale)
        rbbox_list =self.rbbox_head.get_bboxes(*rbbox_inputs)
        rbbox_results = [
            dbbox2result(det_rbboxes, det_labels, self.rbbox_head.num_classes)
            for det_rbboxes, det_labels in rbbox_list
        ]
        return rbbox_results[0]

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError