import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import delta2dbbox, multiclass_nms_rbbox, \
    bbox_target_rbbox, accuracy, rbbox_target_rbbox,\
    choose_best_Rroi_batch, delta2dbbox_v2, \
    Pesudomulticlass_nms_rbbox, delta2dbbox_v3, hbb2obb_v2
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class BBoxHeadRbbox(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=19,
                 target_means=[0., 0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
                 reg_class_agnostic=False,
                 with_module=True,
                 hbb_trans='hbb2obb_v2',
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHeadRbbox, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            # TODO: finish the tuple condition
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            if isinstance(self.roi_feat_size, int):
                in_channels *= (self.roi_feat_size * self.roi_feat_size)
            elif isinstance(self.roi_feat_size, tuple):
                assert len(self.roi_feat_size) == 2
                assert isinstance(self.roi_feat_size[0], int)
                assert isinstance(self.roi_feat_size[1], int)
                in_channels *= (self.roi_feat_size[0] * self.roi_feat_size[1])
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 5 if reg_class_agnostic else 5 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None
        self.with_module = with_module
        self.hbb_trans = hbb_trans

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_masks, gt_labels,
                   rcnn_train_cfg):
        """
        obb target hbb
        :param sampling_results:
        :param gt_masks:
        :param gt_labels:
        :param rcnn_train_cfg:
        :param mod: 'normal' or 'best_match', 'best_match' is used for RoI Transformer
        :return:
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        # pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        # TODO: first get indexs of pos_gt_bboxes, then index from gt_bboxes
        # TODO: refactor it, direct use the gt_rbboxes instead of gt_masks
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds  for res in sampling_results
        ]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_assigned_gt_inds,
            gt_masks,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            with_module=self.with_module,
            hbb_trans=self.hbb_trans)
        return cls_reg_targets

    def get_target_rbbox(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        """
        obb target obb
        :param sampling_results:
        :param gt_bboxes:
        :param gt_labels:
        :param rcnn_train_cfg:
        :return:
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        # pos_proposals = choose_best_Rroi_batch(pos_proposals)
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = rbbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduce=True):
        losses = dict()
        if cls_score is not None:
            losses['rbbox_loss_cls'] = self.loss_cls(
                cls_score, labels, label_weights, reduce=reduce)
            losses['rbbox_acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               5)[pos_inds, labels[pos_inds]]
            losses['rbbox_loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0))
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        # TODO: check and simplify it
        if rois.size(1) == 5:
            obbs = hbb2obb_v2(rois[:, 1:])
        elif rois.size(1) == 6:
            obbs = rois[:, 1:]
        else:
            print('strange size')
            import pdb
            pdb.set_trace()
        if bbox_pred is not None:
            # bboxes = delta2dbbox(rois[:, 1:], bbox_pred, self.target_means,
            #                     self.target_stds, img_shape)
            if self.with_module:
                dbboxes = delta2dbbox(obbs, bbox_pred, self.target_means,
                                      self.target_stds, img_shape)
            else:
                dbboxes = delta2dbbox_v3(obbs, bbox_pred, self.target_means,
                                         self.target_stds, img_shape)
        else:
            # bboxes = rois[:, 1:]
            dbboxes = obbs
            # TODO: add clip here

        if rescale:
            # bboxes /= scale_factor
            # dbboxes[:, :4] /= scale_factor
            dbboxes[:, 0::5] /= scale_factor
            dbboxes[:, 1::5] /= scale_factor
            dbboxes[:, 2::5] /= scale_factor
            dbboxes[:, 3::5] /= scale_factor
        # if cfg is None:
        #     c_device = dbboxes.device
        #
        #     det_bboxes, det_labels = Pesudomulticlass_nms_rbbox(dbboxes, scores,
        #                                             0.05,
        #                                             1000)
        #
        #     return det_bboxes, det_labels
        # else:
        c_device = dbboxes.device

        det_bboxes, det_labels = multiclass_nms_rbbox(dbboxes, scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        # det_bboxes = torch.from_numpy(det_bboxes).to(c_device)
        # det_labels = torch.from_numpy(det_labels).to(c_device)
        return det_bboxes, det_labels

    def get_det_rbboxes(self,
                       rrois,
                       cls_score,
                       rbbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if rbbox_pred is not None:
            # bboxes = delta2dbbox(rois[:, 1:], bbox_pred, self.target_means,
            #                     self.target_stds, img_shape)
            dbboxes = delta2dbbox_v2(rrois[:, 1:], rbbox_pred, self.target_means,
                                     self.target_stds, img_shape)
        else:
            # bboxes = rois[:, 1:]
            dbboxes = rrois[:, 1:]
            # TODO: add clip here

        if rescale:
            # bboxes /= scale_factor
            # dbboxes[:, :4] /= scale_factor
            dbboxes[:, 0::5] /= scale_factor
            dbboxes[:, 1::5] /= scale_factor
            dbboxes[:, 2::5] /= scale_factor
            dbboxes[:, 3::5] /= scale_factor
        if cfg is None:
            return dbboxes, scores
        else:
            c_device = dbboxes.device

            det_bboxes, det_labels = multiclass_nms_rbbox(dbboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            # det_bboxes = torch.from_numpy(det_bboxes).to(c_device)
            # det_labels = torch.from_numpy(det_labels).to(c_device)
            return det_bboxes, det_labels

    def refine_rbboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 5) or (n*bs, 5*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class_rbbox(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    def regress_by_class_rbbox(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 5) or (n, 6)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 5*(#class+1)) or (n, 5)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        # import pdb
        # pdb.set_trace()
        assert rois.size(1) == 5 or rois.size(1) == 6

        if not self.reg_class_agnostic:
            label = label * 5
            inds = torch.stack((label, label + 1, label + 2, label + 3, label + 4), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 5

        if rois.size(1) == 5:
            if self.with_module:
                new_rois = delta2dbbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
            else:
                new_rois = delta2dbbox_v3(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
            # choose best Rroi
            new_rois = choose_best_Rroi_batch(new_rois)
        else:
            if self.with_module:
                bboxes = delta2dbbox(rois[:, 1:], bbox_pred, self.target_means,
                                    self.target_stds, img_meta['img_shape'])
            else:
                bboxes = delta2dbbox_v3(rois[:, 1:], bbox_pred, self.target_means,
                                    self.target_stds, img_meta['img_shape'])
            bboxes = choose_best_Rroi_batch(bboxes)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

    # def regress_by_class(self, rois, label, bbox_pred, img_meta):
    #     """Regress the bbox for the predicted class. Used in Cascade R-CNN.
    #
    #     Args:
    #         rois (Tensor): shape (n, 4) or (n, 5)
    #         label (Tensor): shape (n, )
    #         bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
    #         img_meta (dict): Image meta info.
    #
    #     Returns:
    #         Tensor: Regressed bboxes, the same shape as input rois.
    #     """
    #     assert rois.size(1) == 4 or rois.size(1) == 5
    #
    #     if not self.reg_class_agnostic:
    #         label = label * 4
    #         inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
    #         bbox_pred = torch.gather(bbox_pred, 1, inds)
    #     assert bbox_pred.size(1) == 4
    #
    #     if rois.size(1) == 4:
    #         new_rois = delta2bbox(rois, bbox_pred, self.target_means,
    #                               self.target_stds, img_meta['img_shape'])
    #     else:
    #         bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
    #                             self.target_stds, img_meta['img_shape'])
    #         new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)
    #
    #     return new_rois
