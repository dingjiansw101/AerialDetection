from .single_stage_rbbox import SingleStageDetectorRbbox
from ..registry import DETECTORS


@DETECTORS.register_module
class RetinaNetRbbox(SingleStageDetectorRbbox):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetRbbox, self).__init__(backbone, neck, bbox_head, rbbox_head,
                                             train_cfg, test_cfg, pretrained)
