import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool


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

          train --> trainmultiscale, val --> valmultiscale, test --> testmultiscale
          generate trainmultiscale.txt, valmultiscale.txt, testmultiscale.txt
          cp trainmultiscale, valmultiscale, testmultiscale --> multiscalesplit
    :return:
    """
    if not os.path.exists(os.path.join(srcpath, 'trainmultiscale')):
        os.mkdir(os.path.join(srcpath, 'trainmultiscale'))
    if not os.path.exists(os.path.join(srcpath, 'valmultiscale')):
        os.mkdir(os.path.join(srcpath, 'valmultiscale'))
    if not os.path.exists(os.path.join(srcpath, 'test-devmultiscale')):
        os.mkdir(os.path.join(srcpath, 'test-devmultiscale'))
    if not os.path.exists(os.path.join(srcpath, 'trainvalmultiscale')):
        os.mkdir(os.path.join(srcpath, 'trainvalmultiscale'))
    if not os.path.exists(os.path.join(srcpath, 'images')):
        os.mkdir(os.path.join(srcpath, 'images'))
    if not os.path.exists(os.path.join(srcpath, 'labelTxt')):
        os.mkdir(os.path.join(srcpath, 'labelTxt'))

    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(srcpath, 'trainvalmultiscale'),
                      gap=200,
                      subsize=800,
                      num_process=32
                      )
    split_train.splitdata(1)

    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
                       os.path.join(srcpath, 'trainvalmultiscale'),
                      gap=200,
                      subsize=800,
                      num_process=32
                      )
    split_val.splitdata(1)

    split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(srcpath, 'test-dev', 'images'),
                       os.path.join(srcpath, 'test-devmultiscale', 'images'),
                      gap=200,
                      subsize=800,
                      num_process=32
                      )
    split_test.splitdata(1)

    getnamelist(os.path.join(srcpath, 'trainvalmultiscale', 'images'), os.path.join(srcpath, 'train.txt'))
    getnamelist(os.path.join(srcpath, 'test-devmultiscale', 'images'), os.path.join(srcpath, 'test.txt'))

    filecopy(os.path.join(srcpath, 'trainvalmultiscale', 'images'), os.path.join(srcpath, 'images'))
    filecopy(os.path.join(srcpath, 'trainvalmultiscale', 'labelTxt'), os.path.join(srcpath, 'labelTxt'))

    filecopy(os.path.join(srcpath, 'test-devmultiscale', 'images'), os.path.join(srcpath, 'images'))


if __name__ == '__main__':
    # filecopy(r'/data/dota2/test-devmultiscale/images',
    #          r'/data/dota2/multiscale_split/images')
    prepare(r'/home/dingjian/project/dota2')