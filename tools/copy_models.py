import os
import shutil

configs_dota = [
    'retinanet_r50_fpn_2x_dota' ,
    'retinanet_v5_obb_r50_fpn_2x_dota',
    'mask_rcnn_r50_fpn_1x_dota',
    'htc_without_semantic_r50_fpn_1x_dota',
    'faster_rcnn_r50_fpn_1x_dota',
    'faster_rcnn_r50_fpn_1x_dota_obb_v3',
    'faster_rcnn_dpool_v3_r50_fpn_1x_dota_obb',
    'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota',
    'faster_rcnn_r50_fpn_1x_dota_RoITrans_v5',
    'cascade_mask_rcnn_r50_fpn_1x_dota'
]

configs_dota1_5 = ['retinanet_r50_fpn_2x_dota1_5_v2',
                   'retinanet_v5_obb_r50_fpn_2x_dota1_5_v2',
                   'mask_rcnn_r50_fpn_1x_dota1_5_v2',
                   'cascade_mask_rcnn_r50_fpn_1x_dota1_5_v2',
                   'htc_without_semantic_r50_fpn_1x_dota1_5_v2',
                   'faster_rcnn_r50_fpn_1x_dota1_5_v2',
                   'faster_rcnn_r50_fpn_1x_dota1_5_v2_obb_v3',
                   'faster_rcnn_dpool_v3_r50_fpn_1x_dota1_5_v2_obb',
                   'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota1_5_v2',
                   'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5']

configs_dota2 = [
    'retinanet_r50_fpn_2x_dota2_v3',
    'retinanet_v5_obb_r50_fpn_2x_dota2_v3',
    'mask_rcnn_r50_fpn_1x_dota2_v3',
    'cascade_mask_rcnn_r50_fpn_1x_dota2_v3',
    'htc_without_semantic_r50_fpn_1x_dota2_v3',
    'faster_rcnn_r50_fpn_1x_dota2_v3',
    'faster_rcnn_r50_fpn_1x_dota2_v3_obb_v3',
    'faster_rcnn_dpool_v3_r50_fpn_1x_dota2_v3_obb',
    'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota2_v3',
    'faster_rcnn_r50_fpn_1x_dota2_v3_RoITrans_v5'
]


def param_copy(srcworkdir, dstworkdir):
    # filenames = configs_dota + configs_dota1_5
    filenames = configs_dota2
    for file in filenames:
        dst_config_path = os.path.join(dstworkdir, file)
        if not os.path.exists(dst_config_path):
            os.makedirs(dst_config_path)
        src_config_path = os.path.join(srcworkdir, file)
        if '1x' in file:
            shutil.copy(os.path.join(src_config_path, 'epoch_12.pth'),
                        os.path.join(dst_config_path, 'epoch_12.pth'))
        elif '2x' in file:
            shutil.copy(os.path.join(src_config_path, 'epoch_24.pth'),
                        os.path.join(dst_config_path, 'epoch_24.pth'))
        else:
            print('warnining', file)

if __name__ == '__main__':
    param_copy(r'/home/dingjian/project/code/mmdetection_DOTA/work_dirs',
               r'/home/dingjian/project/code/Aerialdetection/work_dirs')