import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO2 import DOTA2COCOTrain, DOTA2COCOTest
wordname_18 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane',
                  'airport', 'helipad']
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

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

          train --> train800, val --> val800, test --> test800
          generate train800.txt, val800.txt, test800.txt
          cp train800, val800, test800 --> 800split
    :return:
    """
    if not os.path.exists(os.path.join(dstpath, 'test-dev1024')):
        os.mkdir(os.path.join(dstpath, 'test-dev1024'))
    if not os.path.exists(os.path.join(dstpath, 'trainval1024')):
        os.mkdir(os.path.join(dstpath, 'trainval1024'))
    if not os.path.exists(os.path.join(dstpath, 'images')):
        os.mkdir(os.path.join(dstpath, 'images'))
    if not os.path.exists(os.path.join(dstpath, 'labelTxt')):
        os.mkdir(os.path.join(dstpath, 'labelTxt'))

    # split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
    #                    os.path.join(dstpath, 'trainval1024'),
    #                   gap=200,
    #                   subsize=1024,
    #                   num_process=32
    #                   )
    # split_train.splitdata(1)
    #
    # split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
    #                    os.path.join(dstpath, 'trainval1024'),
    #                   gap=200,
    #                   subsize=1024,
    #                   num_process=32
    #                   )
    # split_val.splitdata(1)

    split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test', 'images'),
                       os.path.join(dstpath, 'test1024', 'images'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_test.splitdata(1)

    DOTA2COCOTrain(os.path.join(dstpath, 'trainval1024'), os.path.join(dstpath, 'trainval1024', 'DOTA_trainval1024.json'), wordname_15)
    # DOTA2COCOTest(os.path.join(dstpath, 'test1024'), os.path.join(dstpath, 'test1024', 'DOTA_test1024.json'), wordname_15)

    # getnamelist(os.path.join(dstpath, 'trainval1024', 'images'), os.path.join(dstpath, 'train.txt'))
    # getnamelist(os.path.join(dstpath, 'test-dev1024', 'images'), os.path.join(dstpath, 'test.txt'))
    #
    # filecopy(os.path.join(dstpath, 'trainval1024', 'images'), os.path.join(dstpath, 'images'))
    # filecopy(os.path.join(dstpath, 'trainval1024', 'labelTxt'), os.path.join(dstpath, 'labelTxt'))
    #
    # filecopy(os.path.join(dstpath, 'test-dev1024', 'images'), os.path.join(dstpath, 'images'))


if __name__ == '__main__':
    # filecopy(r'/data/dota2/test-dev800/images',
    #          r'/data/dota2/800_split/images')
    # prepare(r'/home/dingjian/project/dota2', r'/home/dingjian/project/dota2/split-1024')
    prepare(r'/home/dingjian/project/dota', r'/home/dingjian/project/dota/split-1024')
