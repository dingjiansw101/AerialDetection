import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO2 import DOTA2COCOTest, DOTA2COCOTrain
wordname_18 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane',
                  'airport', 'helipad']

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

def prepare(srcpath):
    """
    :param srcpath: train, val, test

          train --> train1024, val --> val1024, test --> test1024
          generate train1024.txt, val1024.txt, test1024.txt
          cp train1024, val1024, test1024 --> 1024split
    :return:
    """
    if not os.path.exists(os.path.join(srcpath, 'train1024')):
        os.mkdir(os.path.join(srcpath, 'train1024'))
    if not os.path.exists(os.path.join(srcpath, 'val1024')):
        os.mkdir(os.path.join(srcpath, 'val1024'))
    if not os.path.exists(os.path.join(srcpath, 'test-dev1024')):
        os.mkdir(os.path.join(srcpath, 'test-dev1024'))
    if not os.path.exists(os.path.join(srcpath, 'trainval1024')):
        os.mkdir(os.path.join(srcpath, 'trainval1024'))
    if not os.path.exists(os.path.join(srcpath, 'images')):
        os.mkdir(os.path.join(srcpath, 'images'))
    if not os.path.exists(os.path.join(srcpath, 'labelTxt')):
        os.mkdir(os.path.join(srcpath, 'labelTxt'))

    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(srcpath, 'trainval1024'),
                      gap=500,
                      subsize=1024,
                      num_process=32
                      )
    split_train.splitdata(1)

    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
                       os.path.join(srcpath, 'trainval1024'),
                      gap=500,
                      subsize=1024,
                      num_process=32
                      )
    split_val.splitdata(1)

    # split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test-dev', 'images'),
    #                    os.path.join(srcpath, 'test-dev1024', 'images'),
    #                   gap=500,
    #                   subsize=1024,
    #                   num_process=32
    #                   )
    # split_test.splitdata(1)
    # split.splitdata(0.5)

    getnamelist(os.path.join(srcpath, 'trainval1024', 'images'), os.path.join(srcpath, 'train.txt'))
    getnamelist(os.path.join(srcpath, 'test-dev1024', 'images'), os.path.join(srcpath, 'test.txt'))

    filecopy(os.path.join(srcpath, 'trainval1024', 'images'), os.path.join(srcpath, 'images'))
    filecopy(os.path.join(srcpath, 'trainval1024', 'labelTxt'), os.path.join(srcpath, 'labelTxt'))

    filecopy(os.path.join(srcpath, 'test-dev1024', 'images'), os.path.join(srcpath, 'images'))

    DOTA2COCOTrain(r'/home/dingjian/project/dota2/trainval1024', r'/home/dingjian/project/dota2/trainval1024/DOTA_trainval1024.json', wordname_18)
    DOTA2COCOTest(r'/home/dingjian/project/dota2/test-dev1024', r'/home/dingjian/project/dota2/test-dev1024/DOTA_test-dev1024.json', wordname_18)
if __name__ == '__main__':
    # filecopy(r'/data/dota2/test-dev1024/images',
    #          r'/data/dota2/1024_split/images')
    prepare(r'/home/dingjian/project/dota2')