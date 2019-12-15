#-----------------------------------------
# some frequently used functions in data, and file process
# way of naming:
# for example, a full path name E:\code\Test\labelTxt2\1.txt, then
# basename indicates 1.txt, name indicates 1, suffix indicates .txt, path indicates E:\code\Test\labelTxt2, dir indicates E:\code\Test\labelTxt2\1.txt
# written by Jian Ding
#-----------------------------------------
## warp to calculate the set union, difference and intersection of files in two paths, not include the suffix, need to add by yourself
import os
import xml.etree.ElementTree as ET
import codecs
import cv2
import sys
import numpy as np
import random
import shutil
import shapely.geometry as shgeo
import re
import pickle
import math
import copy


## initail annotation
datamap = {'0A': 'passenger plane', '0B': 'fighter aeroplane', '0C': 'radar warning aircraft',
           '1': 'baseball diamond', '2': 'bridge', '3': 'ground track', '4A': 'car', '4B': 'truck',
           '4C': 'bus', '5A': 'ship', '5': 'ship', '5B': 'warship', '6': 'tennis court', '7': 'Basketball court',
           '7B': 'half basketball', '8': 'storage tank', '9': 'soccer ball field', '10': 'Turntable',
           '11': 'harbor', '12': 'electric pole', '13': 'parking lot', '14': 'swimming pool', '15': 'lake',
           '16': 'helicopter', '17': 'airport', '18A': 'viaduct', '18B': '18B', '18C': '18C', '18D': '18D',
           '18E': '18E', '18F': '18F', '18G': '18G', '18H': '18H', '18I': '18I', '18J': '18J', '18K': '18K',
           '18L': '18L', '18M': '18M', '18N': '18N', '4A_area': '4A_area', '4B_area': '4B_area',
           '5A_area': '5A_area', '8_area': '8_area', '13_area': '13_area', 'bridge': 'bridge', 'plane': 'plane',
           'ship': 'ship', 'storage': 'storage', 'harbor': 'harbor'}
classname = ['0A', '0B', '0C', '1', '2', '3', '4A', '4B', '4C', '5A', '5B', '6', '7', '8', '9', '10'
    , '11', '12', '13', '14', '15', '16', '17', '18A', '18B', '18C', '18D', '18E'
    , '18F', '18G', '18H', '18I', '18J', '18K', '18L', '18M', '18N', '5', 'plane', 'ship', 'storage', 'bridge',
             'harbor']
clsdict = {'0A': 0, '0B': 0, '0C': 0, '1': 0, '2': 0, '3': 0, '4A': 0, '4B': 0, '4C': 0, '5A': 0, '5B': 0, '6': 0,
           '7': 0, '8': 0, '9': 0, '10': 0
    , '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18A': 0, '18B': 0, '18C': 0, '18D': 0,
           '18E': 0
    , '18F': 0, '18G': 0, '18H': 0, '18I': 0, '18J': 0, '18K': 0, '18L': 0, '18M': 0, '18N': 0, '5': 0
    , 'plane': 0, 'ship': 0, 'storage': 0, 'bridge': 0, 'harbor': 0}
### tmp experiments
datamap2 = {'0A': 'passenger plane', '0B': 'fighter aeroplane', '0C': 'radar',
           '1': 'baseball diamond', '2': 'bridge', '3': 'ground track', '4A': 'car', '4B': 'trunck',
           '4C': 'bus', '5A': 'ship','5B': 'big ship', '6': 'tennis court', '7': 'baseketball court',
           '8': 'storage tank', '9': 'soccer ball field', '10': 'turntable',
           '11': 'harbor', '12': 'electric pole', '13': 'parking lot', '14': 'swimming pool', '15': 'lake',
           '16': 'helicopter', '17': 'airport', '18A': 'viaduct'}
classname_part = ['0A', '0B', '0C', '1', '2', '3', '4A', '4B', '4C', '5A', '5B', '6', '7', '8', '9', '10'
    , '11', '12', '13', '14', '15', '16', '17', '18A']


## prepare for release v1
datamap_15_new = {'0A': 'plane', '0B':'plane', '0C': 'plane',  '1': 'baseball-diamond', '2': 'bridge', '3': 'ground-track-field', '4A': 'small-vehicle', '4B': 'large-vehicle',
           '4C': 'large-vehicle', '5A': 'ship', '5B':'ship', '6': 'tennis-court', '7': 'basketball-court',
           '8': 'storage-tank', '9': 'soccer-ball-field', '10': 'roundabout',
           '11': 'harbor', '14': 'swimming-pool',
           '16': 'helicopter'}

wordname_15_new = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

## before release 1.0 version
datamap_15 = {'0A': 'plane', '0B':'plane', '0C': 'plane',  '1': 'baseball-diamond', '2': 'bridge', '3': 'ground-track-field', '4A': 'small-vehicle', '4B': 'large-vehicle',
           '4C': 'large-vehicle', '5A': 'ship', '5B':'ship', '6': 'tennis-court', '7': 'basketball-court',
           '8': 'storage-tank', '9': 'soccer-ball-field', '10': 'turntable',
           '11': 'harbor', '14': 'swimming-pool',
           '16': 'helicopter'}
identity_15 = {x:x for x in datamap_15}

noorientationnames = ['bridge', 'ground-track-field', 'tennis-court', 'basketball-court',
           'soccer-ball-field',
           'swimming-pool',
           ]

classname_15 = ['0A', '0B', '0C', '1', '2', '3', '4A', '4B', '4C', '5A', '5B', '6', '7', '8', '9', '10', '11', '14', '16']

clsdict_15 = {'0A': 0, '0B': 0, '0C': 0, '1': 0, '2': 0, '3': 0, '4A': 0, '4B': 0, '4C': 0, '5A': 0, '5B': 0, '6': 0,
           '7': 0, '8': 0, '9': 0, '10': 0
    , '11': 0, '14': 0, '16': 0}

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'turntable', 'harbor', 'swimming-pool', 'helicopter']
classnums_15 = { 'ground-track-field':0, 'small-vehicle':0, 'large-vehicle':0, 'harbor':0, 'plane':0, 'ship':0, 'basketball-court':0,
              'swimming-pool':0, 'helicopter':0, 'bridge':0, 'tennis-court':0,
              'baseball-diamond':0, 'storage-tank':0, 'soccer-ball-field':0, 'turntable':0 }

subcategory = ['helicopter', 'bridge', 'baseball-diamond',
                    'ground-track-field', 'basketball-court',
                    'soccer-ball-field', 'harbor']


datamap_getlabelme = { 'baseball-diamond': '1',
     'basketball': '7',
     'bridge': '2',
     'ground': '3',
     'harbor': '11',
     'helicopter': '16',
     'large-vehicle': '4C',
     'plane': '0A',
     'passenger': '0A',
     'ship': '5A',
     'warship': '5B',
     'small-vehicle': '4A',
     'soccer-ball-field': '9',
     'storage': '8',
     'swimming-pool': '14',
     'tennis-court': '6',
     'turntable': '10'
}

###
JL2bod = {'0': '0A', '1': '1', '2': '2', '3': '3', '5': '8', '6': '11', '7': '10', '8': '9', '9': '5B'}

###
GF2bod = {'0': '0A', '0A': '0A', '2': '2', '5': '5B', '5A': '5A', '5B': '5B', '8': '8', '11': '11'}


## ucas darklabel id ==> word
ucas_dark2word = {'0':'small-vehicle', '1': 'plane'}
## bod darklabel id ==> word
bod__dark2word = {'0':'plane', '1': 'small-vehicle'}

def latlon2decimals(degreestr):
    pattern = re.compile(r'-*[0-9]+')
    src = re.findall(pattern, degreestr)
    dst = float(src[0]) + float(src[1])/60 + float(src[2])/3600
    return dst

def keyvalueReverse(inputdic):
    return dict(zip(inputdic.values(), inputdic.keys()))
def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles
def filesetcalc(path1, path2, calc = ''):
    if calc == '':
        print('please assigh a calc')
        return
    file1_list = GetFileFromThisRootDir(path1)
    file_set1 = {os.path.splitext(os.path.basename(x))[0] for x in GetFileFromThisRootDir(path1)}
    file_set2 = {os.path.splitext(os.path.basename(x))[0] for x in GetFileFromThisRootDir(path2)}
    inter_set = file_set1.intersection(file_set2)
    diff_set = file_set1.difference(file_set2)
    union_set = file_set1.union(file_set2)
    #suffix1 = os.path.splitext(os.path.basename(file1_list[0]))[1]
    if calc == 'u':
        print('union_set:', union_set)
        return union_set
    elif calc == 'd':
        print('diff_dict:', diff_set)
        return diff_set
    elif calc == 'i':
        print('inter_dict:', inter_set)
        return inter_set
def dots2ToRecC(rec):
    xmin, xmax, ymin, ymax = dots2ToRec4(rec)
    x = (xmin + xmax)/2
    y = (ymin + ymax)/2
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h
def dots2ToRec4(rec):
    xmin, xmax, ymin, ymax = rec[0], rec[0], rec[1], rec[1]
    for i in range(3):
        xmin = min(xmin, rec[i * 2 + 1])
        xmax = max(xmax, rec[i * 2 + 1])
        ymin = min(ymin, rec[i * 2 + 2])
        ymax = max(ymax, rec[i * 2 + 2])
    return xmin, ymin, xmax, ymax
def dots4ToRecC(poly):
    xmin, ymin, xmax, ymax = dots4ToRec4(poly)
    x = (xmin + xmax)/2
    y = (ymin + ymax)/2
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h
def dots4ToRec4(poly):
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
                            max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    return xmin, ymin, xmax, ymax
def dots4ToRec8(poly):
    xmin, ymin, xmax, ymax = dots4ToRec4(poly)
    return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax
    #return dots2ToRec8(dots4ToRec4(poly))
def dots2ToRec8(rec):
    xmin, ymin, xmax, ymax = rec[0], rec[1], rec[2], rec[3]
    return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax
def orderdict_byvalue():
    pass
def testparse_labelme_poly():
    objects = parse_labelme_poly(r'E:\GAOFEN2\gaofen2Labelme\annotations\singapore-2016-4-27-1.xml')
    print(objects)
def filecopy(srcpath, dstpath, filenames, extent):
    for name in filenames:
        srcdir = os.path.join(srcpath, name + extent)
        dstdir = os.path.join(dstpath, name + extent)
        print('srcdir:', srcdir)
        print('dstdir:', dstdir)
        if os.path.exists(srcdir):
            shutil.copyfile(srcdir, dstdir)

def filemove(srcpath, dstpath, filenames, extent):
    for name in filenames:
        srcdir = os.path.join(srcpath, name + extent)
        dstdir = os.path.join(dstpath, name + extent)
        print('srcdir:', srcdir)
        print('dstdir:', dstdir)
        # import pdb
        # pdb.set_trace()
        if os.path.exists(srcdir):
            shutil.move(srcdir, dstdir)

def TrainTestSplit():
    basepath = r'E:\bod-dataset'
    filelist = GetFileFromThisRootDir(os.path.join(basepath, 'images'))
    name = [os.path.basename(os.path.splitext(x)[0]) for x in filelist if (x != 'Thumbs')]
    train_len = int(len(name) * 0.5)
    val_len = int(len(name) * 1 /6)
    test_len = len(name) - train_len - val_len
    print('train_len:', train_len)
    print('val_len:', val_len)
    print('test_len:', test_len)
    random.shuffle(name)
    print('shuffle name:', name)
    train_set= set(name[0:train_len])
    val_set = set(name[train_len:(train_len + val_len)])
    test_set = set(name[(train_len + val_len):])
    print('intersection:', train_set.intersection(test_set))
    imgsrcpath = os.path.join(basepath, 'images')
    txtsrcpath = os.path.join(basepath, 'wordlabel')
    imgtestpath = os.path.join(basepath, 'testset', 'images')
    txttestpath = os.path.join(basepath, 'testset', 'wordlabel')
    imgtrainpath = os.path.join(basepath, 'trainset', 'images')
    txttrainpath = os.path.join(basepath, 'trainset', 'wordlabel')
    imgvalpath = os.path.join(basepath, 'valset', 'images')
    txtvalpath = os.path.join(basepath, 'valset', 'wordlabel')

    #filemove(imgsrcpath, imgtestpath, test_set, '.png')
    # filemove(txtsrcpath, txttestpath, test_set, '.txt')

    #filemove(imgsrcpath, imgtrainpath, train_set, '.png')
    # filemove(txtsrcpath, txttrainpath, train_set, '.txt')

    #filemove(imgsrcpath, imgvalpath, val_set, '.png')
    # filemove(txtsrcpath, txtvalpath, val_set, '.txt')

    # for imgname in train_set:
    #     if (imgname == 'Thumbs'):
    #         continue
    #     srcname = os.path.join(basepath, 'images', imgname + '.tif')
    #     dstname = os.path.join(basepath, 'train', 'images', imgname + '.tif')
    #     shutil.move(srcname, dstname)
    #     srctxt = os.path.join(basepath, 'labelTxt', imgname + '.txt')
    #     dsttxt = os.path.join(basepath, 'train', 'labelTxt', imgname + '.txt')
    #     print(srctxt)
    #     print(dsttxt)
    #     shutil.move(srctxt, dsttxt)
    # for imgname in test_set:
    #     if (imgname == 'Thumbs'):
    #         continue
    #     srcname = os.path.join(basepath, 'images', imgname + '.tif')
    #     dstname = os.path.join(basepath, 'test', 'images', imgname + '.tif')
    #     shutil.move(srcname, dstname)
    #     srctxt = os.path.join(basepath, 'labelTxt', imgname + '.txt')
    #     dsttxt = os.path.join(basepath, 'test', 'labelTxt', imgname + '.txt')
    #     shutil.move(srctxt, dsttxt)
def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = shgeo.Polygon([(dets[i][0], dets[i][1]),
                                    (dets[i][2], dets[i][3]),
                                    (dets[i][4], dets[i][5]),
                                    (dets[i][6], dets[i][7])
                                    ])
        polys.append(tm_polygon)
        areas.append(tm_polygon.area)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(len(order.size - 1)):
            inter_poly = polys[order[0]].intersection(polys[order[order[j + 1]]])
            inter_area = inter_poly.area
            ovr.append(inter_area / (areas[i] + areas[order[j + 1]] - inter_area))
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]
        order = order([inds + 1])
    return keep

## when get dets on several scale images, use the folowing function to do nms, then get the final predict
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
def test_py_cpu_nms():
    dets = np.array([ [0, 0, 4, 4, 0.7],
                        [2, 2, 7, 6, 0.8],
                        [3, 2, 8, 5, 0.6],
                        [0, 0, 7, 7, 0.75]
                    ])
    keep = py_cpu_nms(dets, 0.5)
    print(keep)
def getorderLabel(filename):
    f = open(filename, 'r', encoding='utf_16')
    lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    labellist = [x[8] for x in splitlines]
    orderlabel = {}
    for cls in clsdict:
        orderlabel[cls] = labellist.count(cls) / len(labellist)
    return orderlabel
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects

def parse_pascal(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [float(bbox.find('xmin').text),
                              float(bbox.find('ymin').text),
                              float(bbox.find('xmax').text),
                              float(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects

def pascal2poly():
    filenames = GetFileFromThisRootDir(r'E:\bod-dataset\cuttestpath2\pascalLabel')
    for filename in filenames:
        objects = parse_pascal(filename)
        basename = mybasename(filename)
        with codecs.open(os.path.join(r'E:\bod-dataset\cuttestpath2\voc2dota',basename + '.txt'), 'w', 'utf_16') as f_out:
            for obj in objects:
                rect = obj['bbox']
                poly = dots2ToRec8(rect)
                outline = ' '.join(map(str, poly)) + ' ' + obj['name']
                f_out.write(outline + '\n')

def parse_labelme_poly(filename):
    """ Parse a labelme xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['deleted'] = obj.find('deleted').text
        obj_struct['verified'] = int(obj.find('verified').text)
        obj_struct['occluded'] = obj.find('occluded').text
        obj_struct['attributes'] = obj.find('attributes').text
        poly = obj.find('polygon').findall('pt')
        obj_struct['polygon'] = []
        for point in poly:
            pt = [point.find('x').text, point.find('y').text]
            obj_struct['polygon'] = obj_struct['polygon'] + pt
        objects.append(obj_struct)
    return objects

def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

small_count = 0

def parse_dota_poly(filename):
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    count = 0
    while True:
        line = f.readline()
        count = count + 1
        # if count < 2:
        #     continue
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                # if splitlines[9] == '1':
                if (splitlines[9] == 'tr'):
                    object_struct['difficult'] = '1'
                else:
                    object_struct['difficult'] = splitlines[9]
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            poly = list(map(lambda x:np.array(x), object_struct['poly']))
            object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            if (object_struct['long-axis'] < 15):
                object_struct['difficult'] = '1'
                global small_count
                small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects

def parse_dota_poly2(filename):
    objects = parse_dota_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    return objects

def parse_bod_poly(filename):
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r', encoding='utf_16')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r', 'utf-16')
        f = fd
    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                # if splitlines[9] == '1':
                if (splitlines[9] == 'tr'):
                    object_struct['difficult'] = '1'
                else:
                    object_struct['difficult'] = splitlines[9]
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            poly = list(map(lambda x:np.array(x), object_struct['poly']))
            object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            if (object_struct['long-axis'] < 15):
                object_struct['difficult'] = '1'
                global small_count
                small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly
def Poly2TuplePoly(poly):
    outpoly = [(poly[0], poly[1]),
               (poly[2], poly[3]),
               (poly[4], poly[5]),
               (poly[6], poly[7]),
               ]
    return outpoly
def parse_bod_poly2(filename):
    objects = parse_bod_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    return objects

def parse_bod_rec(filename):
    objects = parse_bod_poly(filename)
    for obj in objects:
        poly = obj['poly']
        bbox = dots4ToRec4(poly)
        obj['bndbox'] = bbox
        obj['long-axis'] = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        obj['size'] = obj['long-axis']
        obj['short-axis'] = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    return objects
def ImgFormT(srcpath, dstpath, srcform, dstform):
    namelist = GetFileFromThisRootDir(srcpath, srcform)
    for imgname in namelist:
        src = cv2.imread(imgname)
        basename = os.path.splitext(os.path.basename(imgname))[0]
        cv2.imwrite(os.path.join(dstpath, basename + dstform), src)

def saveimageWithMask(img, outname, mask_poly):

    dstimg = copy.deepcopy(img)
    for mask in mask_poly:
        bound = mask.bounds
        if (len(bound) < 4):
            continue
        xmin, ymin, xmax, ymax = bound[0], bound[1], bound[2], bound[3]
        for x in range(int(xmin), int(xmax)):
            for y in range(int(ymin), int(ymax)):
                point = shgeo.Point(x, y)
                if point.within(mask):
                    #print('withing')

                    dstimg[int(y)][int(x)] = 0

    cv2.imwrite(outname, dstimg)

def reWriteImgWithMask(srcpath, dstpath, gtpath, srcform, dstform):
    namelist = GetFileFromThisRootDir(gtpath)
    for fullname in namelist:
        objects = parse_bod_poly(fullname)
        mask_polys = []
        for obj in objects:
            clsname = obj['name']
            matches = re.findall('area|mask', clsname)
            if 'mask' in matches:
                #print('mask:')
                mask_polys.append(shgeo.Polygon(obj['poly']))
            elif 'area' in matches:
                #print('area:')
                mask_polys.append(shgeo.Polygon(obj['poly']))
        basename = mybasename(fullname)
        imgname = os.path.join(srcpath, basename + srcform)
        img = cv2.imread(imgname)
        dstname = os.path.join(dstpath, basename + dstform)
        if len(mask_polys) > 0:
            saveimageWithMask(img, dstname, mask_polys)
def testReWriteimgWithMask():
    gtpath = r'E:\bod-dataset\labelTxt'
    srcpath = r'E:\bod-dataset\images'
    dstpath = r'E:\bod-dataset\jpgswithMask'
    reWriteImgWithMask(srcpath,
                       dstpath,
                       gtpath,
                       '.png',
                       '.jpg')
def testImgTrans(basepath):
    dstpath = os.path.join(basepath, 'Secondjpg')
    srcpath = os.path.join(basepath, 'secondQuality')
    ImgFormT(srcpath, dstpath, '.jpg')
def getcategory(
        basepath,
        label,
        ):
    classedict = {}
    def initdic():
        for clsname in classname_15:
            wordname = datamap_15[clsname]
            classedict[wordname] = []
    initdic()
    picklepath = os.path.join(basepath, 'pickle')
    pickledir = os.path.join(picklepath, 'category-file.pickle')
    if not os.path.isfile(pickledir):
        labelpath = os.path.join(basepath, label)
        filelist = GetFileFromThisRootDir(labelpath)
        for fullname in filelist:
            name = mybasename(fullname)
            objects = parse_bod_poly(fullname)
            for obj in objects:
                #wordname = datamap[obj['name']]
                wordname = obj['name']
                if name not in classedict[wordname]:
                    classedict[wordname].append(name)

        with open(pickledir, 'wb') as f:
            pickle.dump(classedict, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickledir, 'rb') as f:
            classedict = pickle.load(f)
    return classedict

def bod2darknet(srcpath, dstpath, extractclassname):
    filelist = GetFileFromThisRootDir(srcpath)
    for fullname in filelist:
        objects = parse_bod_poly(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']
                bbox = np.array(dots4ToRecC(poly)) / 1024
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])
                else:
                    continue
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))
                f_out.write(outline + '\n')

def nwpubodcoord2darknet(srcpath, dstpath):
    filelist = GetFileFromThisRootDir(srcpath)
    for fullname in filelist:
        objects = parse_bod_poly(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']
                bbox = np.array(dots4ToRecC(poly)) / 1024
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                    continue
                index = int(obj['name']) - 1

                outline = str(index) + ' ' + ' '.join(list(map(str, bbox)))
                f_out.write(outline + '\n')
def testbod2darknet(basepath):
    bod2darknet(os.path.join(basepath, r'testsplit'))
    bod2darknet(os.path.join(basepath, r'trainsplit-2'))
def generatefilelist(basepath, filepath, outname):
    filelist = GetFileFromThisRootDir(os.path.join(filepath, 'images'))
    with open(os.path.join(basepath, outname), 'w') as f_out:
        for fullname in filelist:
            name = os.path.basename(os.path.splitext(fullname)[0])
            outline = os.path.join(basepath, 'JPEGImages', name + '.jpg')
            f_out.write(outline + '\n')
def testgeneratefilelist(basepath):
    testpath = os.path.join(basepath, 'testsplit')
    trainpath = os.path.join(basepath, 'trainsplit-2')
    generatefilelist(basepath, trainpath, 'train.txt')
    generatefilelist(basepath, testpath, 'test.txt')

def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """

    if len(points) != 4:
        raise Exception("Points list not valid." + str(len(points)))

    point = [
        [int(points[0][0]), int(points[0][1])],
        [int(points[1][0]), int(points[1][1])],
        [int(points[2][0]), int(points[2][1])],
        [int(points[3][0]), int(points[3][1])]
    ]
    edge = [
        (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
        (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
        (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
        (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
    ]

    summatory = edge[0] + edge[1] + edge[2] + edge[3];
    if summatory > 0:
        return False
    else:
        return True
def mybasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

## need tuple poly
def Get_clockOrderInPictureCoordinate(poly):
    tmpoly = shgeo.Polygon(poly)
    outpoly = shgeo.polygon.orient(tmpoly, sign=1)
    outpoly = list(outpoly.exterior.coords)[0: -1]
    return outpoly

def get_clockwiseorderwithfirstpoint(poly):
    check = validate_clockwise_points(poly)
    if not check:
        outpoly = [[poly[0][0], poly[0][1]],
                   [poly[3][0], poly[3][1]],
                   [poly[2][0], poly[2][1]],
                   [poly[1][0], poly[1][1]]
                   ]
    else:
        outpoly = poly
    return outpoly
def get_best_begin_point(coordinate):
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        print("choose one direction!")
    return  combinate[force_flag]

def choose_best_begin_point(pre_result):
    final_result = []
    for coordinate in pre_result:
        x1 = coordinate[0][0]
        y1 = coordinate[0][1]
        x2 = coordinate[1][0]
        y2 = coordinate[1][1]
        x3 = coordinate[2][0]
        y3 = coordinate[2][1]
        x4 = coordinate[3][0]
        y4 = coordinate[3][1]
        xmin = min(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        xmax = max(x1, x2, x3, x4)
        ymax = max(y1, y2, y3, y4)
        combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]], [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
        dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        force = 100000000.0
        force_flag = 0
        for i in range(4):
            temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1], dst_coordinate[1]) + cal_line_length(combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
            if temp_force < force:
                force = temp_force
                force_flag = i
        if force_flag != 0:
            print("choose one direction!")
        final_result.append(combinate[force_flag])
    return final_result

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))




class FormatTransBase():
    def __init__(self,
                 basepath):
        self.basepath = basepath
        self.labelpath = os.path.join(basepath, 'labelTxt')
        self.imagepath = os.path.join(basepath, 'images')
        self.Polypath = os.path.join(basepath, r'polylabelTxt')
        self.wordlabelpath = os.path.join(basepath, 'wordlabel')
        self.darkpath = os.path.join(basepath, 'labels')
        self.namelist = [os.path.basename(os.path.splitext(x)[0] ) for x in GetFileFromThisRootDir(self.labelpath)]
        self.wordnamelist = [os.path.basename(os.path.split(x)[0]) for x in GetFileFromThisRootDir(self.wordlabelpath)]
    def testGenerateClassLabel(self):
        classlabel_path = os.path.join(self.basepath, 'classlabel')
        for basename in self.namelist:
            orderlabel = getorderLabel(os.path.join(self.labelpath, basename + '.txt'))
            print('orderlabel:', orderlabel)
            outline = ''
            with open(os.path.join(classlabel_path, basename + '.txt'), 'w') as f:
                for cls in classname:
                    outline = outline + str(cls) + ':' + str(orderlabel[cls]) + ', '
            f.write(outline + '\n')
    def bodpolyToRec(self, label):
        Recpath = os.path.join(self.basepath, r'ReclabelTxt')
        for basename in self.namelist:
#            objects = parse_bod_poly(os.path.join(self.labelpath, basename + '.txt'))
            objects = parse_bod_poly(os.path.join(self.basepath, label, basename + '.txt'))
            f_out = codecs.open(os.path.join(Recpath, basename + '.txt'), 'w', 'utf_16')
            for obj in objects:
                bbox = dots4ToRec8(obj['poly'])
                name = obj['name']
                difficult = obj['difficult']
                bbox = list(map(str, bbox))
                outline = ' '.join(bbox)
                outline = outline + ' ' + name
                if difficult:
                    outline = outline + ' ' + str(difficult)
                f_out.write(outline + '\n')
    def labelme2txt(self):
        annotations_path = os.path.join(self.basepath, 'annotations')
        xmllist = GetFileFromThisRootDir(annotations_path, 'xml')
        for xmlfile in xmllist:
            objects = parse_labelme_poly(xmlfile)
            print('xmlfile:', xmlfile)
            basename = mybasename(xmlfile)
            with codecs.open(os.path.join(self.labelpath, basename + '.txt'), 'w', 'utf_16') as f_out:
                for obj in objects:
                    if (not int(obj['deleted']) ) and (obj['name'] in datamap_getlabelme):
                        outline = ' '.join(obj['polygon']) + ' ' + datamap_getlabelme[obj['name']]
                        f_out.write(outline + '\n')
    def bod2pascal(self):
        pascalLabel_path = os.path.join(self.basepath, r'pascalLabel')
        #pascalLabel_path = os.pardir.join(self.basepath, r'')
        print('go in name list')
        for basename in self.namelist:
            print('basename:', basename)
            #objects = parse_bod_poly(os.path.join(self.labelpath, basename + '.txt'))
            objects = parse_bod_poly(os.path.join(self.wordlabelpath, basename + '.txt'))
            tree_root = ET.Element('annotation')
            folder = ET.SubElement(tree_root, 'secondjpg')
            filename = ET.SubElement(tree_root, basename)
            size = ET.SubElement(tree_root, 'size')
            width = ET.SubElement(size, 'width')
            height = ET.SubElement(size, 'height')
            ## TODO: read imagesize from img or info
            imgname = os.path.join(self.basepath, 'images', basename + '.jpg')
            # img = cv2.imread(imgname)

            ## need change with different width, height
            width.text = str(608)
            height.text = str(608)
            for obj in objects:
                object = ET.SubElement(tree_root, 'object')
                ET.dump(tree_root)
                name = ET.SubElement(object, 'name')
                #name.text = datamap[obj['name']]
                name.text = obj['name']
                difficult = ET.SubElement(object, 'difficult')
                print('difficult:', obj['difficult'])
                difficult.text = str(obj['difficult'])
                print('type difficult.text:', type(difficult.text))
                bndbox = ET.SubElement(object, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmax = ET.SubElement(bndbox, 'xmax')
                ymin = ET.SubElement(bndbox, 'ymin')
                ymax = ET.SubElement(bndbox, 'ymax')
                poly = obj['poly']
                bbox = dots4ToRec4(poly)
                xmin.text = str(bbox[0])
                ymin.text = str(bbox[1])
                xmax.text = str(bbox[2])
                ymax.text = str(bbox[3])
            tree = ET.ElementTree(tree_root)
            tree.write(os.path.join(pascalLabel_path, basename + '.xml'))
    def testtxt2pascal(self):
        self.bod2pascal(self.basepath)
    def imageformatTrans(self):
        srcpath = self.imagepath
        dstpath = os.path.join(self.basepath, 'jpgs')
        filelist = GetFileFromThisRootDir(srcpath)
        for fullname in filelist:
            img = cv2.imread(fullname)
            basename = mybasename(fullname)
            dstname = os.path.join(dstpath, basename + '.jpg')
            cv2.imwrite(dstname, img)

    def ParseTxtAndWrite(self, srcpath, dstpath, transmap=None):
        filelist = GetFileFromThisRootDir(srcpath)
        for fullname in filelist:
            print('fullname:', fullname)
            objects = parse_bod_poly(fullname)
            name = mybasename(fullname)
            outname = os.path.join(self.basepath, dstpath, name + '.txt')
            f_out = codecs.open(outname, 'w', 'utf_16')
            for obj in objects:
                outpoly = obj['poly']
                outpoly = get_clockwiseorderwithfirstpoint(outpoly)
                if obj['difficult'] == '0':
                    difficult = '0'
                elif obj['difficult'] == '2':
                    #outpoly = Get_clockOrderInPictureCoordinate(outpoly)
                    outpoly = get_best_begin_point(outpoly)
                    difficult = '0'
                else:
                    difficult = '1'
                print('obj:', obj)
                if transmap != None:
                    if obj['name'] in transmap:
                        if transmap[obj['name']] in noorientationnames:
                            #outpoly = Get_clockOrderInPictureCoordinate(outpoly)
                            outpoly = get_best_begin_point(outpoly)
                        outpoly = TuplePoly2Poly(outpoly)
                        outline = ' '.join(map(str, outpoly)) + ' ' + transmap[obj['name']] + ' ' + str(difficult)

#                        outline = ' '.join(map(str, obj['poly'])) + ' ' + transmap[obj['name']] + ' ' + str(obj['difficult'])
                        print('outline:', outline)
                        f_out.write(outline + '\n')
                else:
                    outpoly = TuplePoly2Poly(outpoly)
#                    outline = ' '.join(map(str, obj['poly'])) + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    outline = ' '.join(map(str, outpoly)) + ' ' + obj['name'] + ' ' + str(difficult)
                    print('outline:', outline)
                    f_out.write(outline + '\n')

    def ParseAndWriteAllBestFirstPoint(self, srcpath, dstpath, transmap=None):
        filelist = GetFileFromThisRootDir(srcpath)
        for fullname in filelist:
            objects = parse_bod_poly(fullname)
            name = mybasename(fullname)
            outname = os.path.join(self.basepath, dstpath, name + '.txt')
            f_out = codecs.open(outname, 'w', 'utf_16')
            for obj in objects:
                outpoly = obj['poly']
                outpoly = get_clockwiseorderwithfirstpoint(outpoly)
                outpoly = get_best_begin_point(outpoly)
                if obj['difficult'] == '0':
                    difficult = '0'
                elif obj['difficult'] == '2':
                    difficult = '0'
                else:
                    difficult = '1'
                outpoly = TuplePoly2Poly(outpoly)
                if transmap != None:
                    if obj['name'] in transmap:
                        outline = ' '.join(map(str, outpoly)) + ' ' + transmap[obj['name']] + ' ' + str(difficult)
                        print('outline:', outline)
                        f_out.write(outline + '\n')
                else:

                    #                    outline = ' '.join(map(str, obj['poly'])) + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    outline = ' '.join(map(str, outpoly)) + ' ' + obj['name'] + ' ' + str(difficult)
                    print('outline:', outline)
                    f_out.write(outline + '\n')
    def TransTo15ID_gt(self):
        dstpath = r'label5Txt'
        self.ParseTxtAndWrite(self.labelpath, dstpath, identity_15)

    def TransToDota15Word_gt(self):
        dstpath = r'wordlabel'
        self.ParseTxtAndWrite(self.labelpath, dstpath, datamap_15_new)
    def TransTo15Word_gt(self):
        dstpath = r'wordlabel'
        self.ParseTxtAndWrite(self.labelpath, dstpath, datamap_15)
    # def TransTo15class(self, path):
    #     filelist = GetFileFromThisRootDir(self.labelpath)
    #     for fullname in filelist:
    #         objects = parse_bod_poly2(fullname)
    #         name = mybasename(fullname)
    #         outname = os.path.join(self.basepath, path, name + '.txt')
    #         f_out = codecs.open(outname, 'w', 'utf_16')
    #
    #         for obj in objects:
    #             if obj['name'] in classname_15:
    #                 if path == 'wordlabel':
    #                     outline = ' '.join(map(str, obj['poly'])) + ' ' + datamap_15[obj['name']] + ' ' + str(obj['difficult'])
    #                     print('outline:', outline)
    #                     #f_out.write(outline + '\n')
    #                 elif path == 'label15Txt':
    #                     outline = ' '.join(map(str, obj['poly'])) + ' ' + obj['name'] + ' ' + str(obj['difficult'])
    #                     print('outline:', outline)
    #                     f_out.write(outline + '\n')
    def JLLabel2bod(self):
        dstpath = r'bodlabelTxt'
        srcpath = r'E:\GFJL\JL\original-labelTxt'
        #testpath = r'E:\GFJL\JL\testlabelTxt'
        self.ParseTxtAndWrite(srcpath, dstpath, JL2bod)

    def GFLabel2bod(self):
        dstpath = r'bodlabelTxt'
        srcpath = r'E:\GFJL\gaofen2\labelTxt'
        self.ParseTxtAndWrite(srcpath, dstpath, GF2bod)

    def TransTo15Word_gtAllBestPoint(self):
        dstpath = r'wordlabelBestStart'
        self.ParseAndWriteAllBestFirstPoint(self.labelpath, dstpath, datamap_15)
    def wordlabel2dark(self):
        filelist = GetFileFromThisRootDir(self.wordlabelpath)
        #print(filelist)
        for fullname in filelist:
            objects = parse_bod_poly(fullname)
            name = mybasename(fullname)
            with open(os.path.join(self.darkpath, name + '.txt'), 'w') as f_out:
                for obj in objects:
                    poly = obj['poly']
                    bbox = np.array(dots4ToRecC(poly)) / 1024
                    ## note: the box is x_center, y_center, w, h, that means the whole box can be out of border
                    if (str(obj['difficult']) == '1'):
                        continue
                    if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                        continue
                    if (obj['name'] in wordname_15):
                        id = wordname_15.index(obj['name'])
                    else:
                        continue
                    outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))
                    f_out.write(outline + '\n')
# def testmergepatchlabel():
#     mergepatchlabel('pridictpath', 'mergepredictpath')
def npu2bod():
    basepath = r'E:\downloaddataset\NWPU VHR-10 dataset\NWPU'
    filelist = GetFileFromThisRootDir(os.path.join(basepath, 'ground truth'))
    outpath = os.path.join(basepath, 'bod_gt')
    for fullname in filelist:
        f = open(fullname)
        lines = f.readlines()
        basename = mybasename(fullname)
        outdir = os.path.join(outpath, basename + '.txt')
        f_out = codecs.open(outdir, 'w', 'utf_16')
        for line in lines:
            obj = re.findall(r'\d+', line)
            if (len(obj) < 5):
                continue
            bbox = list(map(int, obj[0:4]))
            #print('bbox:', bbox)
            bbox = dots2ToRec8(bbox)
            outline = ' '.join(map(str, bbox)) + ' ' + obj[-1]
            f_out.write(outline + '\n')


## the function is not secure, if use the function for some initialname with '-'
# def extractInitailName(name):
#     splitname_last = name.split('-')[-1]
#     initialname = name[0:-(len(splitname_last) + 1)]
#     return initialname
def extractInitailName(name):
    initialname = name.split('__')[0]
    return initialname
def GetListFromfile(fullname):
    with open(fullname, 'r') as f:
        lines = f.readlines()
        names = {x.strip() for x in lines}
    return names

def testGetListFromfile():
    names = GetListFromfile(r'E:\bod-dataset\trainset\trainset.txt')
    print(names)
    print(len(names))

def bodpolyToRec(srcpath, dstpath):
    #dstpath = os.path.join(r'E:\bod-dataset\patches\subcategorylabel\results\ReclabelTxt')
    filelist = GetFileFromThisRootDir(srcpath)
    namelist = [mybasename(x.strip()) for x in filelist]
    for basename in namelist:
#            objects = parse_bod_poly(os.path.join(self.labelpath, basename + '.txt'))
        objects = parse_bod_poly(os.path.join(srcpath,basename + '.txt'))
        f_out = codecs.open(os.path.join(dstpath, basename + '.txt'), 'w', 'utf_16')
        for obj in objects:
            bbox = dots4ToRec8(obj['poly'])
            name = obj['name']
            difficult = obj['difficult']
            bbox = list(map(str, bbox))
            outline = ' '.join(bbox)
            outline = outline + ' ' + name
            if difficult:
                outline = outline + ' ' + str(difficult)
            f_out.write(outline + '\n')

def comp4trans4to8(srcpath, dstpath):
    filenames = GetFileFromThisRootDir(srcpath)
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()
            splitlines = [x.strip().split() for x in lines]
            basename = mybasename(filename)
            with open(os.path.join(dstpath, basename + '.txt'), 'w') as f_out:
                for splitline in splitlines:
                    if (len(splitline) < 6):
                        continue
                    imgname = splitline[0]
                    confidence = splitline[1]
                    rect = splitline[2:]
                    poly = dots2ToRec8(rect)
                    outline = imgname + ' ' + confidence + ' ' +  ' '.join(map(str, poly))
                    f_out.write(outline + '\n')

if __name__ == '__main__':
    #testgeneratefilelist(r'/home/dj/data/bod')
    #nwpubodcoord2darknet(r'E:\downloaddataset\NWPU\NWPU\labelTxt',
     #                    r'E:\downloaddataset\NWPU\NWPU\darknet_gt')
    #npu2bod()
    #testparsecomp4()
    #mergemark(r'E:\GoogleEarth\up-9-25-data\secondjpg\test\mark', r'E:\GoogleEarth\up-9-25-data\secondjpg\test\reannotation\mergemark.txt')
    #names = ['small-vehicle', 'large-vehicle']
    # bod2darknet(r'/home/dj/data/vehicleDetection/wordlabel',
    #             r'/home/dj/data/vehicleDetection/labels',
    #             names)
    trans = FormatTransBase(r'I:\dota')
    trans.TransToDota15Word_gt()

    # pascal2poly()
    # comp4trans4to8(r'E:\bod-dataset\results\bod_ssd1024_2000000-nms',
    #                r'E:\bod-dataset\results\bod_ssd1024_2000000-nms_dots8')
    #trans.bodpolyToRec('labelTxt')
    # bodpolyToRec(r'E:\bod-dataset\testset\wordlabel',
    #              r'E:\bod-dataset\testset\ReclabelTxt')
    #trans.TransTo15Word_gt()
    #TrainTestSplit()
    # trans = FormatTransBase(r'E:\bod-dataset')
    # trans.TransTo15Word_gt()
    #trans.imageformatTrans()
    #trans.TransTo15Word_gt()
    #testGetListFromfile()
    #bodpolyToRec(srcpath, dstpath)
