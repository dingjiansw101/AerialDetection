import dota_utils as util
import os
import cv2
import json
from PIL import Image

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter 2, 1, or do not filter

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    info = {'contributor': 'captain group',
           'data_created': '2018',
           'description': 'This is 1.0 version of DOTA dataset.',
           'url': 'http://captain.whu.edu.cn/DOTAweb/',
           'version': '1.0',
           'year': 2018}
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
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            print('filename: ', file)
            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    continue
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

def DOTA2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    # labelparent = os.path.join(srcpath, 'labelTxt')
    data_dict = {}
    info = {'contributor': 'captain group',
           'data_created': '2018',
           'description': 'This is 1.0 version of DOTA dataset.',
           'url': 'http://captain.whu.edu.cn/DOTAweb/',
           'version': '1.0',
           'year': 2018}
    data_dict['info'] = info
    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            # img = cv2.imread(imagepath)
            img = Image.open(imagepath)
            # height, width, c = img.shape
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)

if __name__ == '__main__':
    # DOTA2COCO(r'/data0/data_dj/1024_new', r'/data0/data_dj/1024_new/DOTA_trainval1024.json')
    # DOTA2COCOTrain(r'/home/dingjian/project/dota2/trainval1024', r'/home/dingjian/project/dota2/trainval1024/DOTA_trainval1024.json', wordname_18)
    # DOTA2COCOTest(r'/home/dingjian/project/dota2/test-dev1024', r'/home/dingjian/project/dota2/test-dev1024/DOTA_test-dev1024.json', wordname_18)

    # DOTA2COCOTrain(r'/home/dingjian/project/dota2/split-800/trainval800', r'/home/dingjian/project/dota2/split-800/trainval800/DOTA_trainval800.json', wordname_18)
    # DOTA2COCOTestWF(r'/home/dingjian/project/dota/split-1024/test1024_v2', r'/home/dingjian/project/dota/split-1024/test1024_v2/DOTA_test1024_filter.json', wordname_15)
    # DOTA2COCOTrain(r'/home/dingjian/project/dota/split-1024/trainval1024_ms',
    #                r'/home/dingjian/project/dota/split-1024/trainval1024_ms/DOTA_trainval1024_ms.json', wordname_15)
    # DOTA2COCOTest(r'/home/dingjian/project/dota/split-1024/test1024_ms',
    #               r'/home/dingjian/project/dota/split-1024/test1024_ms/DOTA_test1024_ms.json', wordname_15)
    # DOTA2COCOTest(r'/home/dingjian/project/dota/split-1024/debug_imgs',
    #               r'/home/dingjian/project/dota/split-1024/debug_imgs/DOTA_debug.json', wordname_15)

    # DOTA2COCOTrain(r'/data/mmlab-dota1.5/split-1024/trainval1024',
    #                r'/data/mmlab-dota1.5/split-1024/trainval1024/DOTA_trainval1024.json', wordname_16)
    # DOTA2COCOTrain(r'/data/mmlab-dota1.5/split-1024/trainval1024_ms',
    #                r'/data/mmlab-dota1.5/split-1024/trainval1024_ms/DOTA_trainval1024_ms.json', wordname_16)
    # DOTA2COCOTest(r'/home/dingjian/project/code/mmdetection_DOTA/data/dota1_1024/test1024',
    #               r'/home/dingjian/project/code/mmdetection_DOTA/data/dota1_1024/test1024/DOTA1_5_test1024.json',
    #               wordname_16)
    # DOTA2COCOTest(r'/home/dingjian/project/dota/split-1024/test1024_ms',
    #               r'/home/dingjian/project/dota/split-1024/test1024_ms/DOTA1_5_test1024_ms.json',
    #               wordname_16)
    # DOTA2COCOTrain(r'/data/Data_dj/mmdetection_DOTA/data/dota1_5_1024/debug',
    #                r'/data/Data_dj/mmdetection_DOTA/data/dota1_5_1024/debug/DOTA1_5_debug.json',
    #                wordname_16)

    # DOTA2COCOTrain(r'/data/mmlab-dota1.5/split-1024_v2/trainval1024',
    #                r'/data/mmlab-dota1.5/split-1024_v2/trainval1024/DOTA1_5_trainval.json',
    #                wordname_16)
    #
    # DOTA2COCOTrain(r'/data/mmlab-dota1.5/split-1024_v2/trainval1024_ms',
    #                r'/data/mmlab-dota1.5/split-1024_v2/trainval1024_ms/DOTA1_5_trainval_ms.json',
    #                wordname_16)
    # DOTA2COCOTrain(r'/data/mmlab-dota1.5/split-1024_v2/trainval1024',
    #                r'/data/mmlab-dota1.5/split-1024_v2/trainval1024/DOTA1_5_trainval_debug.json',
    #                wordname_16)
    #
    # DOTA2COCOTrain(r'/data/mmlab-dota1.5/split-1024_v2/trainval1024_ms',
    #                r'/data/mmlab-dota1.5/split-1024_v2/trainval1024_ms/DOTA1_5_trainval_ms_debug.json',
    #                wordname_16)

    # DOTA2COCOTest(r'/data/Data_dj/mmdetection_DOTA/data/dota1_5_1024_v2/test1024',
    #               r'/data/Data_dj/mmdetection_DOTA/data/dota1_5_1024_v2/test1024/DOTA1_5_test1024.json',
    #               wordname_16)
    #
    # DOTA2COCOTest(r'/data/Data_dj/mmdetection_DOTA/data/dota1_5_1024_v2/test1024_ms',
    #               r'/data/Data_dj/mmdetection_DOTA/data/dota1_5_1024_v2/test1024_ms/DOTA1_5_test1024_ms.json',
    #               wordname_16)

    # DOTA2COCOTest(r'/home/dingjian/workfs/dota1_5/split-1024_v2/test1024_ms',
    #               r'/home/dingjian/workfs/dota1_5/split-1024_v2/test1024_ms/DOTA1_5_test1024_allms.json',
    #               wordname_16)

    DOTA2COCOTrain(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024',
                   r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024/DOTA_trainval1024.json',
                   wordname_15)
    DOTA2COCOTrain(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms',
                   r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms/DOTA_trainval1024_ms.json',
                   wordname_15)
    DOTA2COCOTest(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024',
                  r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024/DOTA_test1024.json',
                  wordname_15)
    DOTA2COCOTest(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024_ms',
                  r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/test1024_ms/DOTA_test1024_ms.json',
                  wordname_15)
