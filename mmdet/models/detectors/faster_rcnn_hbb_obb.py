import torch
import torch.nn as nn

# from .base import BaseDetector
from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin, RBBoxTestMixin

from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler, dbbox2result


@DETECTORS.register_module
class FasterRCNNHBBOBB(BaseDetectorNew, RPNTestMixin, BBoxTestMixin,
                       RBBoxTestMixin, MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FasterRCNNHBBOBB, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if rbbox_head is not None:
            self.rbbox_roi_extractor = builder.build_roi_extractor(
                rbbox_roi_extractor)
            self.rbbox_head = builder.build_head(rbbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(FasterRCNNHBBOBB, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_rbbox:
            self.rbbox_roi_extractor.init_weights()
            self.rbbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)


        # bbox head forward and loss
        if self.with_rbbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            rbbox_feats = self.rbbox_roi_extractor(
                x[:self.rbbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                rbbox_feats = self.shared_head(rbbox_feats)
            rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)

            rbbox_targets = self.rbbox_head.get_target(
                sampling_results, gt_masks, gt_labels, self.train_cfg.rcnn)

            loss_rbbox = self.rbbox_head.loss(rcls_score, rbbox_pred,
                                            *rbbox_targets)
            losses.update(loss_rbbox)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        assert self.with_rbbox, "RBox head must be implemented."
        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        # TODO: implement the dbbox2result
        # bbox_results = dbbox2result(det_bboxes, det_labels,
        #                            self.bbox_head.num_classes)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_rbbox:
            return bbox_results
        else:
            det_rbboxes, det_rlabels = self.simple_test_rbboxes_v2(
                x, img_meta, det_bboxes, self.test_cfg.rrcnn, rescale=rescale)
            # import pdb
            # pdb.set_trace()
            rbbox_results = dbbox2result(det_rbboxes, det_rlabels,
                                         self.rbbox_head.num_classes)
            return bbox_results, rbbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        assert NotImplementedError
        # proposal_list = self.aug_test_rpn(
        #     self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        # det_bboxes, det_labels = self.aug_test_bboxes(
        #     self.extract_feats(imgs), img_metas, proposal_list,
        #     self.test_cfg.rcnn)
        #
        # if rescale:
        #     _det_bboxes = det_bboxes
        # else:
        #     _det_bboxes = det_bboxes.clone()
        #     _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        # bbox_results = bbox2result(_det_bboxes, det_labels,
        #                            self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        # if self.with_mask:
        #     segm_results = self.aug_test_mask(
        #         self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
        #     return bbox_results, segm_results
        # else:
        #     return bbox_results
