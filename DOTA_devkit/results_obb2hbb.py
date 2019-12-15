import DOTA_devkit.utils as util
import os

from DOTA_devkit.dota_evaluation_task2 import evaluation_task2_transfer_warp
from DOTA_devkit.dotav2_0_evaluation_task2 import evaluation_task2_warp_refactor
import argparse

dota_annopath = r'data/dota1_test/labelTxt/{:s}.txt'
imagesetfile = r'data/dota1_test/testset.txt'
classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court',
              'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
              'helicopter']

dota1_5_classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court',
              'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
              'helicopter', 'container-crane']
dota1_5_annopath = r'data/dota1_5_test/labelTxt/{:s}.txt'
dota1_5_imagesetfile = r'data/dota1_5_test/testset.txt'

dota2_classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court',
              'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
              'helicopter', 'container-crane',
              'airport', 'helipad']

dota2_annopath = r'data/dota2_test-dev/labelTxt/{:s}.txt'
dota2_imagesetfile = r'data/dota2_test-dev/test.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(r'--path', default=r'/home/dj/code/mmdetection_DOTA/work_dirs/faster_rcnn_r50_fpn_1x_dota_RoITrans_v2/save1_nms2000')
    parser.add_argument('--version', default='dota_v1',
                        help='dota version')
    args = parser.parse_args()

    return args

def OBB2HBB(srcpath, dstpath):
    filenames = util.GetFileFromThisRootDir(srcpath)
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    for file in filenames:
        with open(file, 'r') as f_in:
            with open(os.path.join(dstpath, util.mybasename(file) + '.txt'), 'w') as f_out:
                lines = f_in.readlines()
                splitlines = [x.strip().split() for x in lines]
                for index, splitline in enumerate(splitlines):
                    imgname = splitline[0]
                    score = splitline[1]
                    poly = splitline[2:]
                    poly = list(map(float, poly))
                    xmin, xmax, ymin, ymax = min(poly[0::2]), max(poly[0::2]), min(poly[1::2]), max(poly[1::2])
                    rec_poly = [xmin, ymin, xmax, ymax]
                    outline = imgname + ' ' + score + ' ' + ' '.join(map(str, rec_poly))
                    if index != (len(splitlines) - 1):
                        outline = outline + '\n'
                    f_out.write(outline)

def TransAndEval(srcpath, dstpath, dota_version):
    OBB2HBB(srcpath, dstpath)
    # TODO: refactor these codes
    if dota_version == 'dota_v1':
        evaluation_task2_transfer_warp(os.path.join(dstpath, '{:s}.txt'),
                          dota_annopath,
                          imagesetfile,
                          classnames)
    elif dota_version == 'dota_v2':
        evaluation_task2_warp_refactor(os.path.join(dstpath, '{:s}.txt'),
                            dota2_annopath,
                            dota2_imagesetfile,
                            dota2_classnames,
                           os.path.join(dstpath, '../', 'Transfer_Task2_mAP.txt') )
    elif dota_version == 'dota_v1_5':
        evaluation_task2_warp_refactor(os.path.join(dstpath, '{:s}.txt'),
                            dota1_5_annopath,
                            dota1_5_imagesetfile,
                            dota1_5_classnames,
                            os.path.join(dstpath, '../', 'Transfer_Task2_mAP.txt'))

def TransAndEval_refactor(srcpath, dstpath, dota_annopath, imagesetfile, classnames):
    OBB2HBB(srcpath, dstpath)
    evaluation_task2_warp_refactor(os.path.join(dstpath, '{:s}.txt'),
                                   dota_annopath,
                                   imagesetfile,
                                   classnames,
                                   os.path.join(dstpath, '../', 'Transfer_Task2_mAP.txt'))

if __name__ == '__main__':
    args = parse_args()
    obb_results_path = os.path.join(args.path, r'Task1_results_nms')
    hbb_results_path = os.path.join(args.path, r'Transed_Task2_results_nms')
    dota_version = args.version
    TransAndEval(obb_results_path,
           hbb_results_path,
                 dota_version)
