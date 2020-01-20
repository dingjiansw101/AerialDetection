import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO2 import DOTA2COCOTest, DOTA2COCOTrain
wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)
def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)

def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)

def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')

def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test

          train --> train1024, val --> val1024, test --> test1024
          generate train1024.txt, val1024.txt, test1024.txt
          cp train1024, val1024, test1024 --> 1024split
    :return:
    """
    if not os.path.exists(os.path.join(dstpath, 'train1024')):
        os.mkdir(os.path.join(dstpath, 'train1024'))
    if not os.path.exists(os.path.join(dstpath, 'val1024')):
        os.mkdir(os.path.join(dstpath, 'val1024'))
    if not os.path.exists(os.path.join(dstpath, 'test1024')):
        os.mkdir(os.path.join(dstpath, 'test1024'))
    if not os.path.exists(os.path.join(dstpath, 'trainval1024')):
        os.mkdir(os.path.join(dstpath, 'trainval1024'))
    if not os.path.exists(os.path.join(dstpath, 'images')):
        os.mkdir(os.path.join(dstpath, 'images'))
    if not os.path.exists(os.path.join(dstpath, 'labelTxt')):
        os.mkdir(os.path.join(dstpath, 'labelTxt'))

    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(dstpath, 'trainval1024'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_train.splitdata(1)

    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
                       os.path.join(dstpath, 'trainval1024'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_val.splitdata(1)

    split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test', 'images'),
                       os.path.join(dstpath, 'test1024', 'images'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_test.splitdata(1)
    # split.splitdata(0.5)

    getnamelist(os.path.join(dstpath, 'trainval1024', 'images'), os.path.join(dstpath, 'train.txt'))
    getnamelist(os.path.join(dstpath, 'test1024', 'images'), os.path.join(dstpath, 'test.txt'))

    DOTA2COCOTrain(os.path.join(dstpath, 'trainval1024'), os.path.join(dstpath, 'trainval1024', 'DOTA_trainval1024.json'), wordname_16)
    DOTA2COCOTest(os.path.join(dstpath, 'test1024'), os.path.join(dstpath, 'test1024', 'DOTA_test1024.json'), wordname_16)

if __name__ == '__main__':
    prepare(r'/home/dingjian/project/dota',
            r'/home/dingjian/workfs/dota1-split-1024')