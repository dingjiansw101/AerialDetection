import DOTA_devkit.utils as util
import codecs
import os

datamap_16 = {'0A': 'plane', '0B':'plane', '0C': 'plane',  '1': 'baseball-diamond', '2': 'bridge', '3': 'ground-track-field', '4A': 'small-vehicle', '4B': 'large-vehicle',
           '4C': 'large-vehicle', '5A': 'ship', '5B':'ship', '6': 'tennis-court', '7': 'basketball-court',
           '8': 'storage-tank', '9': 'soccer-ball-field', '10': 'roundabout',
           '11': 'harbor', '14': 'swimming-pool',
           '16': 'helicopter', '19': 'container-crane'}



datamap_inverse = {datamap_16[x]:x for x in datamap_16}

def parsecomp4_poly(srcpath, dstpath, thresh=0.1):

    if not os.path.exists(dstpath):
        os.makedirs(dstpath)

    filedict = {}
    complist = util.GetFileFromThisRootDir(srcpath, '.txt')

    for compfile in complist:
        idname = util.mybasename(compfile).split('_')[-1]
        idname = datamap_inverse[idname]
        f = open(compfile, 'r')
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            # print('line:', line)
            splitline = line.strip().split(' ')
            filename = splitline[0]
            confidence = splitline[1]
            bbox = splitline[2:]
            if float(confidence) > thresh:
                if filename not in filedict:
                    filedict[filename] = codecs.open(os.path.join(dstpath, filename + '.txt'), 'w', 'utf_16')
                #poly = util.dots2ToRec8(bbox)
                poly = bbox
#               filedict[filename].write(' '.join(poly) + ' ' + idname + '_' + str(round(float(confidence), 2)) + '\n')
#             print('idname:', idname)

            #filedict[filename].write(' '.join(poly) + ' ' + idname + '_' + str(round(float(confidence), 2)) + '\n')
            # print('filename:', filename)
                filedict[filename].write(' '.join(poly) + ' ' + idname + '\n')

# def dots2ToRec8(rec):
#     xmin, ymin, xmax, ymax = rec[0], rec[1], rec[2], rec[3]
#     return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax

def parsecomp4_rec(srcpath, dstpath, thresh=0.1):

    if not os.path.exists(dstpath):
        os.makedirs(dstpath)

    filedict = {}
    complist = util.GetFileFromThisRootDir(srcpath, '.txt')

    for compfile in complist:
        idname = util.mybasename(compfile).split('_')[-1]
        idname = datamap_inverse[idname]
        f = open(compfile, 'r')
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            # print('line:', line)
            splitline = line.strip().split(' ')
            filename = splitline[0]
            confidence = splitline[1]
            bbox = splitline[2:]
            if float(confidence) > thresh:
                if filename not in filedict:
                    filedict[filename] = codecs.open(os.path.join(dstpath, filename + '.txt'), 'w', 'utf_16')
                poly = util.dots2ToRec8(list(map(float,bbox)))
                # poly = bbox
#               filedict[filename].write(' '.join(poly) + ' ' + idname + '_' + str(round(float(confidence), 2)) + '\n')
#             print('idname:', idname)

            #filedict[filename].write(' '.join(poly) + ' ' + idname + '_' + str(round(float(confidence), 2)) + '\n')
            # print('filename:', filename)
                filedict[filename].write(' '.join(map(str, poly)) + ' ' + idname + '\n')

if __name__ == '__main__':
    parsecomp4_poly(r'/data/Data_dj/mmdetection_DOTA/work_dirs/mask_rcnn_r50_fpn_1x_dota1_5_v2/Task1_results_nms',
                    r'/data/Data_dj/mmdetection_DOTA/work_dirs/mask_rcnn_r50_fpn_1x_dota1_5_v2/Task1_results_nms_single_results')
    parsecomp4_rec(r'/data/Data_dj/mmdetection_DOTA/work_dirs/mask_rcnn_r50_fpn_1x_dota1_5_v2/Task2_results_nms',
                   r'/data/Data_dj/mmdetection_DOTA/work_dirs/mask_rcnn_r50_fpn_1x_dota1_5_v2/Task2_results_nms_single_results')
    parsecomp4_rec(r'/data/Data_dj/mmdetection_DOTA/work_dirs/mask_rcnn_r50_fpn_1x_dota1_5_v2/Transed_Task2_results_nms',
                   r'/data/Data_dj/mmdetection_DOTA/work_dirs/mask_rcnn_r50_fpn_1x_dota1_5_v2/Transed_Task2_results_nms_single_results')