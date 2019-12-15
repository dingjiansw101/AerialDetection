from torch.nn.modules.module import Module
from ..functions.roi_align_rotated import RoIAlignRotatedFunction


class RoIAlignRotated(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(RoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlignRotatedFunction.apply(features, rois, self.out_size,
                                      self.spatial_scale, self.sample_num)
