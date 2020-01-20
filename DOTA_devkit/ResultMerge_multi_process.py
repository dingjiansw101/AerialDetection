"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import re
import time
import sys
# sys.path.insert(0,'..')
# print('fffffffffffffffffff')

try:
    import dota_utils as util
except:
    import DOTA_devkit.dota_utils as util
import DOTA_devkit.polyiou as polyiou
import pdb
import math
from multiprocessing import Pool
from functools import partial
from DOTA_devkit.nms import obb_hybrid_NMS, obb_HNMS

#TODO: there is a bug at 5 decimal places of mAP when using the program
def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def py_cpu_nms_poly_fast(dets, thresh):
    # TODO: test it
    try:
        obbs = dets[:, 0:-1]
    except:
        print('fail index')
        pdb.set_trace()
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
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

def nmsbynamedict(nameboxdict, nms, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        outdets = []
        for index in keep:
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict
def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def mergesingle(dstpath, nms, nms_thresh, fullname):
    name = util.custombasename(fullname)
    #print('name:', name)
    dstname = os.path.join(dstpath, name + '.txt')
    with open(fullname, 'r') as f_in:
        # print('fullname: ', fullname)
        nameboxdict = {}
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            subname = splitline[0]
            splitname = subname.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            #print('subname:', subname)
            x_y = re.findall(pattern1, subname)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])

            pattern2 = re.compile(r'__([\d+\.]+)__\d+___')

            rate = re.findall(pattern2, subname)[0]

            confidence = splitline[1]
            poly = list(map(float, splitline[2:]))
            origpoly = poly2origpoly(poly, x, y, rate)
            det = origpoly
            det.append(confidence)
            det = list(map(float, det))
            if (oriname not in nameboxdict):
                nameboxdict[oriname] = []
            nameboxdict[oriname].append(det)
        nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
        with open(dstname, 'w') as f_out:
            for imgname in nameboxnmsdict:
                for det in nameboxnmsdict[imgname]:
                    #print('det:', det)
                    confidence = det[-1]
                    bbox = det[0:-1]
                    outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                    #print('outline:', outline)
                    f_out.write(outline + '\n')

def mergebase_parallel(srcpath, dstpath, nms, nms_thresh):
    pool = Pool(16)
    filelist = util.GetFileFromThisRootDir(srcpath)

    mergesingle_fn = partial(mergesingle, dstpath, nms, nms_thresh)
    # pdb.set_trace()
    pool.map(mergesingle_fn, filelist)

def mergebase(srcpath, dstpath, nms, nms_thresh):
    filelist = util.GetFileFromThisRootDir(srcpath)
    for filename in filelist:
        mergesingle(dstpath, nms, nms_thresh, filename)

def mergebyrec(srcpath, dstpath, nms_thresh=0.3):
    mergebase_parallel(srcpath,
              dstpath,
              py_cpu_nms, nms_thresh)

def mergebypoly_multiprocess(srcpath, dstpath, nms_type='py_cpu_nms_poly_fast', o_thresh=0.1, h_thresh=0.5):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'
    if nms_type == 'py_cpu_nms_poly_fast':
        mergebase_parallel(srcpath,
                           dstpath,
                           py_cpu_nms_poly_fast, o_thresh)
    elif nms_type == 'obb_HNMS':
        mergebase_parallel(srcpath,
                           dstpath,
                           obb_HNMS, o_thresh)
    elif nms_type == 'obb_hybrid_NMS':
        obb_hybrid_NMS_partial = partial(obb_hybrid_NMS, o_thresh)
        mergebase_parallel(srcpath,
                           dstpath,
                           obb_hybrid_NMS_partial, h_thresh)
if __name__ == '__main__':
    # mergebypoly(r'/home/dingjian/code/DOTA_devkit/Test_nms2/Task1_results', r'/home/dingjian/code/DOTA_devkit/Test_nms2/Task1_results_0.1_nms_fast')
    mergebyrec(r'/home/dingjian/Documents/Research/experiments/mmdetection_DOTA/scratch_faster_rcnn_r50_fpn_gn_2x_dota2/Task2_results',
               r'/home/dingjian/Documents/Research/experiments/mmdetection_DOTA/scratch_faster_rcnn_r50_fpn_gn_2x_dota2/Task2_results_nms')