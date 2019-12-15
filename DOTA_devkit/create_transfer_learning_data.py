import DOTA_devkit.utils as util
import os
import json
from PIL import Image

# 1. create txt file for val set
# 2. create json file for train 1024 set

# extract source1, 2, 3 from train, cp them to soruce1, 2, 3 file
# extract source1, 2, 3 from val, cp them to sourc1, 2, 3 file
# source1: GF-2, JL-1, source2: GoogleEarth, source3: Aerial

#GoogleEarth, JL, GF, Aerial
wordname_18 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane',
                  'airport', 'helipad']
def getsource(filepath, origpath):
    ## change to find source in the orig images
    orig_filepath = os.path.join(origpath, util.mybasename(filepath).split('__')[0] + '.txt')
    with open(orig_filepath, 'r') as f_in:
        lines = f_in.readlines()
        souceline = lines[0].strip()
        # print('filename: ', filepath)
        imgsouce = souceline.split(':')[1]
        print('imgsource: ', imgsouce)
    return imgsouce

def DOTA2COCOTrain(srcpath, destfile, cls_names, filenames):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    info = {'contributor': 'captain group',
           'data_created': '2019',
           'description': 'This is 2.0 version of DOTA dataset.',
           'url': 'http://captain.whu.edu.cn/DOTAweb/',
           'version': '2.0',
           'year': 2019}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        # filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.mybasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            # img = cv2.imread(imagepath)
            # height, width, c = img.shape
            img = Image.open(imagepath)
            height, width = img.height, img.width

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)

def getsource_dict(filenames, origpath):
    source_dict = {}
    # source_list = ['GoogleEarth', 'Aerial', 'Satellite']
    for filename in filenames:
        imgsource = getsource(filename, origpath)
        # merge GF and JL
        if (imgsource == 'GF') or (imgsource == 'JL'):
            imgsource = 'Satellite'
        if imgsource not in source_dict:
            source_dict[imgsource] = []
        source_dict[imgsource].append(filename)
    return source_dict

def orig_filter(filenames, orig_path):
    basenames = [util.mybasename(x) for x in filenames]
    orignames = [x.split('__')[0] for x in basenames]
    train_names = util.GetFileFromThisRootDir(orig_path)
    train_basenames = [util.mybasename(x) for x in train_names]

    out_names = []
    for idex, name in enumerate(orignames):
        if name in train_basenames:
            out_names.append(filenames[idex])
    return out_names

def extractfilenames(srcpath):
    """
    :param srcpath: train or val, including images, labelTxt
    :param dstpath: 
    :return:
    """
    # source_splits = ['satellite', 'GoogleEarth', 'Aerial']
    srclabelpath = os.path.join(srcpath, 'labelTxt')
    filenames = util.GetFileFromThisRootDir(srclabelpath)

    train_filenames = orig_filter(filenames, r'/home/dingjian/project/dota2/train/labelTxt')
    val_filenames = orig_filter(filenames, r'/home/dingjian/project/dota2/val/labelTxt')

    train_source_dict = getsource_dict(train_filenames, r'/home/dingjian/project/dota2/train/labelTxt')
    val_source_dict = getsource_dict(val_filenames, r'/home/dingjian/project/dota2/val/labelTxt')

    return train_source_dict, val_source_dict

def DOTA2COCOTrain_diffsource(source_dict, srcpath):
    # source_dict = extractfilenames(srcpath)
    for source in source_dict:
        sourfiles = source_dict[source]
        DOTA2COCOTrain(srcpath, os.path.join(srcpath, 'train_' + source + '.json'), wordname_18, sourfiles)

def DOTA2COCOVal_diffsource(source_dict, srcpath):
    # source_dict = extractfilenames(srcpath)
    for source in source_dict:
        sourfiles = source_dict[source]
        DOTA2COCOTrain(srcpath, os.path.join(srcpath, 'val_' + source + '.json'), wordname_18, sourfiles)

def DOTA2COCOTrainVal_diffsource(srcpath):
    train_source_dict, val_source_dict = extractfilenames(srcpath)
    DOTA2COCOTrain_diffsource(train_source_dict, srcpath)
    DOTA2COCOVal_diffsource(val_source_dict, srcpath)

if __name__ == '__main__':
    DOTA2COCOTrainVal_diffsource(r'/home/dingjian/project/dota2/split-1024/trainval1024')