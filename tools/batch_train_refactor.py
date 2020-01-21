import os

def batch_train(confg_list):

    for config in confg_list:
        # select dataset
        if ('_dota' in config) and ('_dota2' not in config) and ('_dota1_5' not in config):
            dataset = 'DOTA'
        elif '_dota2' in config:
            dataset = 'DOTA2'
        elif '_dota1_5' in config:
            dataset = 'DOTA1_5'
        else:
            raise Exception

        # write test_single file
        launch_file = 'train_{}.sh'.format(config)
        with open(launch_file, 'w') as f_in:
            f_in.write(r'#!/usr/bin/env bash' + '\n')
            f_in.write('./tools/dist_train.sh configs/{}/{}.py 4'.format(dataset, config))
        # launch file
        status = os.system('sbatch -A gsxia -p gpu --gres=gpu:4 -c 12 {}'.format(launch_file))

if __name__ == '__main__':

    configs_dota_sub = [
        # 'retinanet_r50_fpn_2x_dota',
        # 'retinanet_obb_r50_fpn_2x_dota',
        # 'mask_rcnn_r50_fpn_1x_dota',
        # 'faster_rcnn_obb_r50_fpn_1x_dota',
        # 'faster_rcnn_h-obb_r50_fpn_1x_dota',
        # 'faster_rcnn_RoITrans_r50_fpn_1x_dota'
    ]

    configs_dota1_5_sub = [
        'faster_rcnn_RoITrans_r50_fpn_1x_dota1_5_gap512_msTrainTest_rotationTrainTest',
        'mask_rcnn_r50_fpn_1x_dota1_5',
        'faster_rcnn_obb_r50_fpn_1x_dota1_5',
        'faster_rcnn_RoITrans_r50_fpn_1x_dota1_5',
    ]

    batch_train(configs_dota_sub + configs_dota1_5_sub)