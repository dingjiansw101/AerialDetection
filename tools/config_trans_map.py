import os
import sys

# nametrans_map = {'cascade_mask_rcnn_r50_fpn_1x_dota'}
configs_dota = {
    'retinanet_r50_fpn_2x_dota': 'retinanet_r50_fpn_2x_dota' ,
    'retinanet_v5_obb_r50_fpn_2x_dota': 'retinanet_obb_r50_fpn_2x_dota',
    'mask_rcnn_r50_fpn_1x_dota': 'mask_rcnn_r50_fpn_1x_dota',
    'htc_without_semantic_r50_fpn_1x_dota': 'htc_without_semantic_r50_fpn_1x_dota',
    'faster_rcnn_r50_fpn_1x_dota': 'faster_rcnn_r50_fpn_1x_dota',
    'faster_rcnn_r50_fpn_1x_dota_obb_v3': 'faster_rcnn_obb_r50_fpn_1x_dota',
    'faster_rcnn_dpool_v3_r50_fpn_1x_dota_obb':  'faster_rcnn_obb_dpool_r50_fpn_1x_dota',
    'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota': 'faster_rcnn_h-obb_r50_fpn_1x_dota',
    'faster_rcnn_r50_fpn_1x_dota_RoITrans_v5': 'faster_rcnn_RoITrans_r50_fpn_1x_dota',
    'cascade_mask_rcnn_r50_fpn_1x_dota': 'cascade_mask_rcnn_r50_fpn_1x_dota'
}

configs_dota1_5 = {'retinanet_r50_fpn_2x_dota1_5_v2': 'retinanet_r50_fpn_2x_dota1_5',
                   'retinanet_v5_obb_r50_fpn_2x_dota1_5_v2': 'retinanet_obb_r50_fpn_2x_dota1_5',
                   'mask_rcnn_r50_fpn_1x_dota1_5_v2': 'mask_rcnn_r50_fpn_1x_dota1_5',
                   'cascade_mask_rcnn_r50_fpn_1x_dota1_5_v2': 'cascade_mask_rcnn_r50_fpn_1x_dota1_5',
                   'htc_without_semantic_r50_fpn_1x_dota1_5_v2': 'htc_without_semantic_r50_fpn_1x_dota1_5',
                   'faster_rcnn_r50_fpn_1x_dota1_5_v2': 'faster_rcnn_r50_fpn_1x_dota1_5',
                   'faster_rcnn_r50_fpn_1x_dota1_5_v2_obb_v3': 'faster_rcnn_obb_r50_fpn_1x_dota1_5',
                   'faster_rcnn_dpool_v3_r50_fpn_1x_dota1_5_v2_obb': 'faster_rcnn_obb_dpool_r50_fpn_1x_dota1_5',
                   'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota1_5_v2': 'faster_rcnn_h-obb_r50_fpn_1x_dota1_5',
                   'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5': 'faster_rcnn_RoITrans_r50_fpn_1x_dota1_5'}

configs_dota2 = {
    # 'retinanet_r50_fpn_1x_dota2_v3',
    'retinanet_r50_fpn_2x_dota2_v3': 'retinanet_r50_fpn_2x_dota2',
    'retinanet_v5_obb_r50_fpn_2x_dota2_v3': 'retinanet_obb_r50_fpn_2x_dota2',
    'mask_rcnn_r50_fpn_1x_dota2_v3': 'mask_rcnn_r50_fpn_1x_dota2',
    'cascade_mask_rcnn_r50_fpn_1x_dota2_v3': 'cascade_mask_rcnn_r50_fpn_1x_dota2',
    'htc_without_semantic_r50_fpn_1x_dota2_v3': 'htc_without_semantic_r50_fpn_1x_dota2',
    'faster_rcnn_r50_fpn_1x_dota2_v3': 'faster_rcnn_r50_fpn_1x_dota2',
    'faster_rcnn_r50_fpn_1x_dota2_v3_obb_v3': 'faster_rcnn_r50_fpn_1x_dota2_obb',
    'faster_rcnn_dpool_v3_r50_fpn_1x_dota2_v3_obb': 'faster_rcnn_dpool_r50_fpn_1x_dota2_obb',
    'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota2_v3': 'faster_rcnn_h-obb_r50_fpn_1x_dota2',
    'faster_rcnn_r50_fpn_1x_dota2_v3_RoITrans_v5': 'faster_rcnn_r50_fpn_1x_dota2_RoITrans',

    'faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5': 'faster_rcnn_r101_fpn_1x_dota2_RoITrans',
    'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_RoITrans_v5': 'faster_rcnn_x101_64x4d_fpn_1x_dota2_RoITrans',

    'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_obb_v3': 'faster_rcnn_x101_64x4d_fpn_1x_dota2_obb',
    'faster_rcnn_r101_fpn_1x_dota2_v3_obb_v3': 'faster_rcnn_r101_fpn_1x_dota2_obb',

    'mask_rcnn_x101_64x4d_fpn_1x_dota2_v3': 'mask_rcnn_x101_64x4d_fpn_1x_dota2',
    'mask_rcnn_r101_fpn_1x_dota2_v3': 'mask_rcnn_r101_fpn_1x_dota2',

    'faster_rcnn_dpool_v3_r101_fpn_1x_dota2_v3_obb': 'faster_rcnn_dpool_r101_fpn_1x_dota2_obb',
    'faster_rcnn_dpool_v3_x101_64x4d_fpn_1x_dota2_v3_obb': 'faster_rcnn_dpool_x101_64x4d_fpn_1x_dota2_obb',

    'retinanet_v5_obb_r101_fpn_2x_dota2_v3': 'retinanet_obb_r101_fpn_2x_dota2',
    'retinanet_v5_obb_x101_64x4d_fpn_2x_dota2_v3': 'retinanet_obb_x101_64x4d_fpn_2x_dota2'
}

def nametrans(config_path):
    dota_1_path = os.path.join(config_path, 'DOTA')
    dota1_5_path = os.path.join(config_path, 'DOTA1_5')
    dota2_path = os.path.join(config_path, 'DOTA2')
    for srcconfig in configs_dota:
        os.system('mv {} {}'.format(os.path.join(dota_1_path, srcconfig + '.py'),
                                    os.path.join(dota_1_path, configs_dota[srcconfig] + '.py')))

    for srcconfig in configs_dota1_5:
        os.system('mv {} {}'.format(os.path.join(dota1_5_path, srcconfig + '.py'),
                                    os.path.join(dota1_5_path, configs_dota1_5[srcconfig] + '.py')))

    for srcconfig in configs_dota2:
        os.system('mv {} {}'.format(os.path.join(dota2_path, srcconfig + '.py'),
                                    os.path.join(dota2_path, configs_dota2[srcconfig] + '.py')))

def work_dir_trans(work_dir_path):
    configs_all = {}
    configs_all.update(configs_dota)
    configs_all.update(configs_dota1_5)
    configs_all.update(configs_dota2)
    for config in configs_all:

        os.system('mv {} {}'.format(os.path.join(work_dir_path, config),
                                    os.path.join(work_dir_path, configs_all[config])))

if __name__ == '__main__':
    nametrans(r'/home/dingjian/project/code/Aerialdetection/configs')
    # work_dir_trans(r'/home/dingjian/project/code/Aerialdetection/work_dirs')