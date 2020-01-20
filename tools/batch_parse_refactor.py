import os

def batch_parse(config_list):


    for config in config_list:
        if ('_dota' in config) and ('_dota2' not in config) and ('_dota1_5' not in config):
            dataset = 'DOTA'
        elif '_dota2' in config:
            dataset = 'DOTA2'
        elif '_dota1_5' in config:
            dataset = 'DOTA1_5'
        else:
            raise Exception

        if ('obb' in config):
            if ('hbb' in config):
                type = 'HBBOBB'
            else:
                type = 'POLY'
        elif ('RoITrans' in config):
            type = 'POLY'
        elif ('mask' in config) or ('htc' in config):
            type = 'Mask'
        else:
            print('config name: ', config)
            type = 'HBB'

        # write test_single file
        launch_file = 'parse_{}.sh'.format(config)
        with open(launch_file, 'w') as f_in:
            f_in.write(r'#!/usr/bin/env bash' + '\n')
            f_in.write('python tools/parse_results.py --config configs/{}/{}.py '
                       '--type {}'.format(dataset, config, type))
        # launch file
        status = os.system('sbatch -c 4 -A gsxia {}'.format(launch_file))

if __name__ == '__main__':
    configs_dota = [
        'retinanet_r50_fpn_1x_dota',
        'retinanet_v5_obb_r50_fpn_2x_dota',
        'mask_rcnn_r50_fpn_1x_dota_cy',
        'htc_without_semantic_r50_fpn_1x_dota',
        'faster_rcnn_r50_fpn_1x_dota',
        'faster_rcnn_r50_fpn_1x_dota_obb_v3',
        'faster_rcnn_dpool_v3_r50_fpn_1x_dota_obb',
        'faster_rcnn_mdpool_r50_fpn_1x_dota_obb_v3',
        'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota',
        'faster_rcnn_r50_fpn_1x_dota_RoITrans_v5',
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

    configs_dota2 = ['retinanet_r50_fpn_1x_dota2_v3',
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

    configs_speed_accuracy = [
               'faster_rcnn_r18_fpn_1x_dota2_v3_RoITrans_v5',
               'faster_rcnn_r34_fpn_1x_dota2_v3_RoITrans_v5',
               'faster_rcnn_r50_fpn_1x_dota2_v3_RoITrans_v5',
               'faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5',
               'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_RoITrans_v5',

               'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_obb_v3',
               'faster_rcnn_r101_fpn_1x_dota2_v3_obb_v3',
               'faster_rcnn_r50_fpn_1x_dota2_v3_obb_v3',
               'faster_rcnn_r34_fpn_1x_dota2_v3_obb_v3',
               'faster_rcnn_r18_fpn_1x_dota2_v3_obb_v3',

                'mask_rcnn_r18_fpn_1x_dota2_v3',
               'mask_rcnn_r34_fpn_1x_dota2_v3',
               'mask_rcnn_r50_fpn_1x_dota2_v3',
               'mask_rcnn_x101_64x4d_fpn_1x_dota2_v3',
               'mask_rcnn_r101_fpn_1x_dota2_v3',

                'faster_rcnn_dpool_v3_r18_fpn_1x_dota2_v3_obb',
               'faster_rcnn_dpool_v3_r34_fpn_1x_dota2_v3_obb',
               'faster_rcnn_dpool_v3_r50_fpn_1x_dota2_v3_obb',
               'faster_rcnn_dpool_v3_x101_64x4d_fpn_1x_dota2_v3_obb',

               'retinanet_v5_obb_r18_fpn_2x_dota2_v3',
               'retinanet_v5_obb_r34_fpn_2x_dota2_v3',
               'retinanet_v5_obb_r50_fpn_2x_dota2_v3',
                'retinanet_v5_obb_x101_64x4d_fpn_2x_dota2_v3'
               ]

    # batch_parse(configs_speed_accuracy)

    configs_small_ablations = ['faster_rcnn_r50_fpn_1x_dota2',
                               'faster_rcnn_r50_fpn_1x_dota2_v2',
                               'faster_rcnn_r50_fpn_1x_dota2_v3']

    # batch_parse(configs_small_ablations)

    # batch_parse(configs_dota)
    # batch_parse(configs_dota1_5)

    dota1_5_data_aug = ['faster_rcnn_obb_hbb_v3_r101_fpn_1x_dota1_5_v3_split512_rotation_ms',
                        'faster_rcnn_r101_fpn_1x_dota1_5_v3_RoITrans_v5_split512_rotation_ms']
    # batch_parse(dota1_5_data_aug)

    configs_speed_accuracy_new = [
        # 'faster_rcnn_r18_fpn_1x_dota2_v3_RoITrans_v5',
        # 'faster_rcnn_r34_fpn_1x_dota2_v3_RoITrans_v5',
        'faster_rcnn_r50_fpn_1x_dota2_v3_RoITrans_v5',
        'faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5',
        'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_RoITrans_v5',

        'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_obb_v3',
        'faster_rcnn_r101_fpn_1x_dota2_v3_obb_v3',
        'faster_rcnn_r50_fpn_1x_dota2_v3_obb_v3',
        # 'faster_rcnn_r34_fpn_1x_dota2_v3_obb_v3',
        # 'faster_rcnn_r18_fpn_1x_dota2_v3_obb_v3',

        #  'mask_rcnn_r18_fpn_1x_dota2_v3',
        # 'mask_rcnn_r34_fpn_1x_dota2_v3',
        # 'mask_rcnn_r50_fpn_1x_dota2_v3',
        # 'mask_rcnn_x101_64x4d_fpn_1x_dota2_v3',
        # 'mask_rcnn_r101_fpn_1x_dota2_v3',

        #  'faster_rcnn_dpool_v3_r18_fpn_1x_dota2_v3_obb',
        # 'faster_rcnn_dpool_v3_r34_fpn_1x_dota2_v3_obb',
        'faster_rcnn_dpool_v3_r50_fpn_1x_dota2_v3_obb',
        'faster_rcnn_dpool_v3_r101_fpn_1x_dota2_v3_obb',
        'faster_rcnn_dpool_v3_x101_64x4d_fpn_1x_dota2_v3_obb',

        # 'retinanet_v5_obb_r18_fpn_2x_dota2_v3',
        # 'retinanet_v5_obb_r34_fpn_2x_dota2_v3',
        'retinanet_v5_obb_r50_fpn_2x_dota2_v3',
        'retinanet_v5_obb_r101_fpn_2x_dota2_v3',
        'retinanet_v5_obb_x101_64x4d_fpn_2x_dota2_v3'
    ]

    configs_dota2_sub = ['retinanet_r50_fpn_1x_dota2_v3',
                         # 'retinanet_v5_obb_r50_fpn_2x_dota2_v3',
                         # 'mask_rcnn_r50_fpn_1x_dota2_v3',
                         'cascade_mask_rcnn_r50_fpn_1x_dota2_v3',
                         'htc_without_semantic_r50_fpn_1x_dota2_v3',
                         'faster_rcnn_r50_fpn_1x_dota2_v3',
                         # 'faster_rcnn_r50_fpn_1x_dota2_v3_obb_v3',
                         # 'faster_rcnn_dpool_v3_r50_fpn_1x_dota2_v3_obb',
                         'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota2_v3',
                         # 'faster_rcnn_r50_fpn_1x_dota2_v3_RoITrans_v5'
                         ]

    # dota2_left = configs_speed_accuracy_new + configs_dota2_sub

    # batch_parse(dota2_left)

    configs_mask = ['mask_rcnn_r50_fpn_1x_dota2_v3',
               'mask_rcnn_x101_64x4d_fpn_1x_dota2_v3',
               'mask_rcnn_r101_fpn_1x_dota2_v3']

    # batch_parse(configs_mask)

    configs_left2 = [
                    'retinanet_r50_fpn_2x_dota'
                   # 'cascade_mask_rcnn_r50_fpn_1x_dota'
    ]

    # batch_parse(configs_left2)
    configs_retina_epoch_ablation = ['retinanet_r50_fpn_2x_dota2_v3',
                                     'retinanet_v5_obb_r50_fpn_1x_dota2_v3']
    # batch_parse(configs_retina_epoch_ablation)

    configs_augmentation_ablation = ['faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512',
                                     'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512_msTrain',
                                     'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512_msTrainTest',
                                     'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512_msTrainTest_rotation',
                                     'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512_rotation']
    # batch_parse(configs_augmentation_ablation)

    batch_parse(configs_dota + configs_dota1_5 + configs_dota2)