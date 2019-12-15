import DOTA_devkit.utils as util
import codecs
import os

datamap_15 = {'0A': 'plane', '0B':'plane', '0C': 'plane',  '1': 'baseball-diamond', '2': 'bridge', '3': 'ground-track-field', '4A': 'small-vehicle', '4B': 'large-vehicle',
           '4C': 'large-vehicle', '5A': 'ship', '5B':'ship', '6': 'tennis-court', '7': 'basketball-court',
           '8': 'storage-tank', '9': 'soccer-ball-field', '10': 'roundabout',
           '11': 'harbor', '14': 'swimming-pool',
           '16': 'helicopter'}



datamap_inverse = {datamap_15[x]:x for x in datamap_15}

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