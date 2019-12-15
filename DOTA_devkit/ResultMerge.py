"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import dota_utils as util
import re
import time
import polyiou
from poly_nms_gpu.nms_wrapper import poly_nms_gpu
import pdb

## the thresh for nms when merge image
nms_thresh = 0.1

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
            if (iou != iou):
                print('poly i:', polys[i], 'polys j + 1:', polys[j + 1])
                print('det i:', dets[i], 'dets j + 1:', dets[order[j + 1]])
                pdb.set_trace()
        if (np.sum(ovr != ovr) > 0):
            print('before')
            pdb.set_trace()

        ovr2 = np.array(ovr)
        if (np.sum(ovr2 != ovr2) > 0):
            print('after')
            pdb.set_trace()
        inds = np.where(ovr2 <= thresh)[0]
        order = order[inds + 1]
    return keep
def test_nms():
    dets = [ [  6.86000000e+02,   2.97600000e+03,   7.09000000e+02,   2.97600000e+03,
              7.24000000e+02,   2.97600000e+03,   7.01000000e+02,   2.97600000e+03,
   2.71368679e-03], [  6.86000000e+02,   2.97600000e+03,   7.09000000e+02,   2.97600000e+03,
   7.24000000e+02,   2.97600000e+03,   7.01000000e+02,   2.97600000e+03,
   2.70966860e-03] ]
    dets = np.array(dets)
    keep = py_cpu_nms_poly(dets, nms_thresh)
    print(keep)
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
        #print('imgname:', imgname)
        #keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        #print('type nameboxdict:', type(nameboxnmsdict))
        #print('type imgname:', type(imgname))
        #print('type nms:', type(nms))
        try:
            keep = nms(np.array(nameboxdict[imgname], dtype=np.float32), thresh)  ## for gpu
        except:
            keep = nms(np.array(nameboxdict[imgname]), thresh) ## for cpu
        #print('keep:', keep)
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
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

def mergebase(srcpath, dstpath, nms):
    assert os.path.exists(srcpath), "The srcpath is not exists!"
    filelist = util.GetFileFromThisRootDir(srcpath)
    assert os.path.exists(dstpath), "The dstpath is not exists!"
    for fullname in filelist:
        name = util.custombasename(fullname)
        #print('name:', name)
        dstname = os.path.join(dstpath, name + '.txt')
        with open(fullname, 'r') as f_in:
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
def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)
def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'
    mergebase(srcpath,
              dstpath,
              py_cpu_nms_poly)
    # mergebase(srcpath,
    #           dstpath,
    #           poly_nms_gpu)
if __name__ == '__main__':

    start = time.clock()
    # mergebypoly(r'/home/dingjian/data/ODAI/ODAI_submmit/baseline/task1_results',
    #             r'/home/dingjian/data/ODAI/ODAI_submmit/baseline/task1_merge2')
    # mergebypoly(r'/home/dingjian/Documents/Research/experiments/rotateanchorrotateregion/40_angle_agnostic',
    #             r'/home/dingjian/Documents/Research/experiments/rotateanchorrotateregion/40_angle_agnostic_0.1_nms')
    mergebypoly(r'/home/dingjian/Documents/Research/experiments/Deform_FPN_Naive_poly/Task1_results_epoch12',
                r'/home/dingjian/Documents/Research/experiments/Deform_FPN_Naive_poly/Task_results_epoch12_0.1_nms')
    # mergebyrec(r'/home/dingjian/Documents/Research/experiments/Deform_FPN_HBB/Task2_results',
    #            r'/home/dingjian/Documents/Research/experiments/Deform_FPN_HBB/Task2_results_0.3_nms')
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
    # test_nms()