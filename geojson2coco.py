import os
import json
from typing import List
import math

import cv2
import numpy as np
from PIL import Image

roksi_15classes = ['소형 선박', '대형 선박', '민간 항공기', '군용 항공기', '소형 승용차', '버스', '트럭', '기차', '크레인', '다리', '정유탱크',
               '댐', '운동경기장', '헬리패드', '원형 교차로']


def convert_xywha_to_8coords(xywha, is_clockwise=False):
    x, y, w, h, a = xywha
    angle = a if is_clockwise else -a

    lt_x, lt_y = -w / 2, -h / 2
    rt_x, rt_y = w / 2, - h/ 2
    rb_x, rb_y = w / 2, h / 2
    lb_x, lb_y = - w / 2, h / 2

    lt_x_ = lt_x * math.cos(angle) - lt_y * math.sin(angle) + x
    lt_y_ = lt_x * math.sin(angle) + lt_y * math.cos(angle) + y
    rt_x_ = rt_x * math.cos(angle) - rt_y * math.sin(angle) + x
    rt_y_ = rt_x * math.sin(angle) + rt_y * math.cos(angle) + y
    lb_x_ = lb_x * math.cos(angle) - lb_y * math.sin(angle) + x
    lb_y_ = lb_x * math.sin(angle) + lb_y * math.cos(angle) + y
    rb_x_ = rb_x * math.cos(angle) - rb_y * math.sin(angle) + x
    rb_y_ = rb_x * math.sin(angle) + rb_y * math.cos(angle) + y

    return [lt_x_, lt_y_, rt_x_, rt_y_, rb_x_, rb_y_, lb_x_, lb_y_]


def convert_8coords_to_4coords(coords):
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    
    xmin = min(x_coords)
    ymin = min(y_coords)

    xmax = max(x_coords)
    ymax = max(y_coords)

    w = xmax-xmin
    h = ymax-ymin

    return [xmin, ymin, w, h]


def convert_labels_to_objects(coords, class_ids, classes, class_names, image_ids, difficult=0, is_clockwise=False):
    objs = list()
    inst_count = 1

    for xywha, cls_id, cls_name, img_id in zip(coords, class_ids, classes, image_ids):
        x, y, w, h, a = xywha
        polygons = convert_xywha_to_8coords(xywha)
        single_obj = {}
        single_obj['difficult'] = difficult
        single_obj['area'] = w*h
        if cls_name in class_names:
            single_obj['category_id'] = class_names.index(cls_name) + 1
        else:
            continue
        single_obj['segmentation'] = [[int(p) for p in polygons]]
        single_obj['iscrowd'] = 0
        single_obj['bbox'] = convert_8coords_to_4coords(polygons)
        single_obj['image_id'] = img_id
        single_obj['id'] = inst_count
        inst_count += 1
        objs.append(single_obj)
    return objs


def load_geojson(filename):
    """ Gets label data from a geojson label file

    :param (str) filename: file path to a geojson label file
    :return: (numpy.ndarray, numpy.ndarray ,numpy.ndarray) coords, chips, and classes corresponding to
            the coordinates, image names, and class codes for each ground truth.
    """

    with open(filename) as f:
        data = json.load(f)

    # obj_coords = np.zeros((len(data['features']), 5))
    # image_ids = np.zeros((len(data['features'])), dtype='object')
    # class_indices = np.zeros((len(data['features'])), dtype=int)
    # class_names = np.zeros((len(data['features'])), dtype='object')
    obj_coords = list()
    image_ids = list()
    class_indices = list()
    class_names = list()

    for idx in range(len(data['features'])):
        properties = data['features'][idx]['properties']
        #image_ids.append(properties['image_id'])
        image_ids.append(properties['image_id'].replace('PS4', 'PS3')[:-4]+'.png')
        obj_coords.append([float(num) for num in properties['object_imcoords'].split(",")])
        class_indices.append(properties['type_id'])
        class_names.append(properties['type_name'])
        # image_ids[idx] = properties['image_id']
        # obj_coords[idx] = np.array([float(num) for num in properties['object_imcoords'].split(",")])
        # class_indices[idx] = properties['type_id']
        # class_names[idx] = properties['type_name']

    return image_ids, obj_coords, class_indices, class_names


def geojson2coco(imageroot: str, geojsonpath: str, destfile, class_names: List, difficult='0'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(class_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        img_files, obj_coords, cls_ids, classes = load_geojson(geojsonpath)
        img_id_map= {img_file:i+1 for i, img_file in enumerate(list(set(img_files)))}
        image_ids = [img_id_map[img_file] for img_file in img_files]
        objs = convert_labels_to_objects(obj_coords, cls_ids, classes, class_names, image_ids, difficult=difficult, is_clockwise=False)
        data_dict['annotations'].extend(objs)

        for imgfile in img_id_map:
            imagepath = os.path.join(imageroot, imgfile)
            img_id = img_id_map[imgfile]
            img = cv2.imread(imagepath)
            height, width, c = img.shape
            single_image = {}
            single_image['file_name'] = imgfile
            single_image['id'] = img_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

        json.dump(data_dict, f_out)


if __name__ == '__main__':

    geojson2coco(imageroot=r'/mnt/workspace/hakjinlee/datasets/NIA20A/object/train/images',
                 geojsonpath=r'/mnt/workspace/hakjinlee/datasets/NIA20A/object/train/train.geojson',
                 destfile=r'/mnt/workspace/hakjinlee/datasets/NIA20A/object/train/traincoco.json',
                 class_names=roksi_15classes)
    geojson2coco(imageroot=r'/mnt/workspace/hakjinlee/datasets/NIA20A/object/val/images',
                 geojsonpath=r'/mnt/workspace/hakjinlee/datasets/NIA20A/object/val/val.geojson',
                 destfile=r'/mnt/workspace/hakjinlee/datasets/NIA20A/object/val/valcoco.json',
                 class_names=roksi_15classes)