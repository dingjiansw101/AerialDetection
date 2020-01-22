import os
import time

def batch_test(confg_set):

    for config in confg_set:
        # select dataset
        if ('_dota' in config) and ('_dota2' not in config) and ('_dota1_5' not in config):
            dataset = 'DOTA'
        elif '_dota2' in config:
            dataset = 'DOTA2'
        elif '_dota1_5' in config:
            dataset = 'DOTA1_5'
        else:
            raise Exception
        # time.sleep(60)
        # select epoch
        if '1x' in config:
            epoch = '12'
        elif '2x' in config:
            epoch = '24'
        else:
            raise Exception
        # write test_single file
        launch_file = 'inference_single_{}.sh'.format(config)
        with open(launch_file, 'w') as f_in:
            f_in.write(r'#!/usr/bin/env bash' + '\n')
            f_in.write('python tools/test.py configs/{}/{}.py work_dirs/{}/epoch_{}.pth '
                       '--out work_dirs/{}/results.pkl '
                       '--log_dir work_dirs/{}'.format(dataset, config, config, epoch, config, config))
        # launch file
        status = os.system('sbatch -A gsxia -p gpu --gres=gpu:1 -c 4 {}'.format(launch_file))

if __name__ == '__main__':

    configs_dota = [
        'retinanet_r50_fpn_1x_dota', #
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

    configs_dota2 = [
                    # 'retinanet_r50_fpn_1x_dota2_v3',
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

    configs = configs_dota + configs_dota1_5 + configs_dota2 + configs_speed_accuracy
    configs = set(configs)

    print('len configs: ', len(configs))
    # batch_test(configs)

    configs_faster = [               'faster_rcnn_r50_fpn_1x_dota2_v3_RoITrans_v5',
               'faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5',
               'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_RoITrans_v5']
    configs_retinanet = [               'retinanet_v5_obb_r50_fpn_2x_dota2_v3',
                                        'retinanet_v5_obb_r101_fpn_2x_dota2_v3'
                'retinanet_v5_obb_x101_64x4d_fpn_2x_dota2_v3']
    configs_RoITrans = [               'faster_rcnn_r50_fpn_1x_dota2_v3_RoITrans_v5',
               'faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5',
               'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_RoITrans_v5']

    configs_Dpool = [               'faster_rcnn_dpool_v3_r50_fpn_1x_dota2_v3_obb',
                                    'faster_rcnn_dpool_v3_r101_fpn_1x_dota2_v3_obb',
               'faster_rcnn_dpool_v3_x101_64x4d_fpn_1x_dota2_v3_obb']

    # print('len configs_speed_accuracy: ', len(configs_speed_accuracy))
    #
    # batch_test(configs_speed_accuracy)

    # configs_small_ablations = ['faster_rcnn_r50_fpn_1x_dota2',
    #                            'faster_rcnn_r50_fpn_1x_dota2_v2',
    #                            'faster_rcnn_r50_fpn_1x_dota2_v3']
    #
    # batch_test(configs_small_ablations)

    #--------------------------------------------------------
    configs_mask = ['mask_rcnn_r50_fpn_1x_dota2_v3',
               'mask_rcnn_x101_64x4d_fpn_1x_dota2_v3',
               'mask_rcnn_r101_fpn_1x_dota2_v3']
    # batch_test(configs_mask)


    #--------------------------------------------------------

    # batch_test(configs_dota)

    # batch_test(configs_dota1_5)



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

    dota2_left = configs_speed_accuracy_new + configs_dota2_sub

    # batch_test(dota2_left)

    configs_data_aug = ['faster_rcnn_obb_hbb_v3_r101_fpn_1x_dota1_5_v3_split512_rotation_ms',
                        'faster_rcnn_r101_fpn_1x_dota1_5_v3_RoITrans_v5_split512_rotation_ms']
    # batch_test(configs_data_aug)


    configs_left2 = [
                    'retinanet_r50_fpn_2x_dota'
                   # 'cascade_mask_rcnn_r50_fpn_1x_dota'
                     ]

    configs_retina_epoch_ablation = ['retinanet_r50_fpn_2x_dota2_v3',
                                     'retinanet_v5_obb_r50_fpn_1x_dota2_v3']

    # # batch_test(configs_left2)
    # batch_test(configs_retina_epoch_ablation)

    # dota1.5 speed test
    # batch_test(configs_dota1_5)

    # dota2 speed test
    # batch_test(configs_dota2)

    # configs_augmentation_ablation = ['faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512',
    #                                  'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512_msTrain',
    #                                  'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512_msTrainTest',
    #                                  'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512_msTrainTest_rotation',
    #                                  'faster_rcnn_r50_fpn_1x_dota1_5_v2_RoITrans_v5_gap512_rotation']
    # batch_test(configs_augmentation_ablation)

    dota2_speed_accuracy = [
            r'retinanet_r101_fpn_2x_dota2_v3',
            # r'retinanet_x101_64x4d_fpn_2x_dota2_v3',
            # r'cascade_mask_rcnn_r101_fpn_1x_dota2_v3',
            # r'cascade_mask_rcnn_x101_64x4d_fpn_1x_dota2_v3',
            # r'htc_without_semantic_r101_fpn_1x_dota2_v3',
            # r'htc_without_semantic_x101_64x4d_fpn_1x_dota2_v3',
            r'faster_rcnn_r101_fpn_1x_dota2_v3',
            # r'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3',
            r'faster_rcnn_obb_hbb_v3_r101_fpn_1x_dota2_v3',
            # r'faster_rcnn_obb_hbb_v3_x101_64x4d_fpn_1x_dota2_v3'
        ]

    # batch_test(dota2_speed_accuracy)

    dota2_speed_accuracy_all = [
        # 'faster_rcnn_r50_fpn_1x_dota2_v3_RoITrans_v5',
        # 'faster_rcnn_r101_fpn_1x_dota2_v3_RoITrans_v5',
        # 'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_RoITrans_v5',
        #
        # 'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3_obb_v3',
        # 'faster_rcnn_r101_fpn_1x_dota2_v3_obb_v3',
        # 'faster_rcnn_r50_fpn_1x_dota2_v3_obb_v3',

        'faster_rcnn_dpool_v3_r50_fpn_1x_dota2_v3_obb',
        'faster_rcnn_dpool_v3_r101_fpn_1x_dota2_v3_obb',
        'faster_rcnn_dpool_v3_x101_64x4d_fpn_1x_dota2_v3_obb',

        'retinanet_v5_obb_r50_fpn_2x_dota2_v3',
        'retinanet_v5_obb_r101_fpn_2x_dota2_v3',
        'retinanet_v5_obb_x101_64x4d_fpn_2x_dota2_v3'

        'mask_rcnn_r50_fpn_1x_dota2_v3',
        'mask_rcnn_x101_64x4d_fpn_1x_dota2_v3',
        'mask_rcnn_r101_fpn_1x_dota2_v3',

        r'retinanet_r50_fpn_2x_dota2_v3',
        r'retinanet_r101_fpn_2x_dota2_v3',
        r'retinanet_x101_64x4d_fpn_2x_dota2_v3',

        r'cascade_mask_rcnn_r50_fpn_1x_dota2_v3',
        r'cascade_mask_rcnn_r101_fpn_1x_dota2_v3',
        r'cascade_mask_rcnn_x101_64x4d_fpn_1x_dota2_v3',

        r'htc_without_semantic_r50_fpn_1x_dota2_v3',
        r'htc_without_semantic_r101_fpn_1x_dota2_v3',
        r'htc_without_semantic_x101_64x4d_fpn_1x_dota2_v3',

        r'faster_rcnn_r50_fpn_1x_dota2_v3',
        r'faster_rcnn_r101_fpn_1x_dota2_v3',
        r'faster_rcnn_x101_64x4d_fpn_1x_dota2_v3',

        r'faster_rcnn_obb_hbb_v3_r50_fpn_1x_dota2_v3',
        r'faster_rcnn_obb_hbb_v3_r101_fpn_1x_dota2_v3',
        r'faster_rcnn_obb_hbb_v3_x101_64x4d_fpn_1x_dota2_v3'

    ]

    # batch_test(dota2_speed_accuracy_all)
    configs_dota_sub = [
        'retinanet_r50_fpn_2x_dota',
        # 'retinanet_obb_r50_fpn_2x_dota',
        'mask_rcnn_r50_fpn_1x_dota',
        'faster_rcnn_obb_r50_fpn_1x_dota',
        'faster_rcnn_h-obb_r50_fpn_1x_dota',
        'faster_rcnn_RoITrans_r50_fpn_1x_dota'
    ]

    configs_dota1_5_sub = [
        # 'faster_rcnn_RoITrans_r50_fpn_1x_dota1_5_gap512_msTrainTest_rotationTrainTest',
        'mask_rcnn_r50_fpn_1x_dota1_5',
        'faster_rcnn_obb_r50_fpn_1x_dota1_5',
        'faster_rcnn_RoITrans_r50_fpn_1x_dota1_5',
    ]
    batch_test(configs_dota_sub + configs_dota1_5_sub)