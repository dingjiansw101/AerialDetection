import os
import dota_utils as util
from multiprocessing import Pool
import cv2
import numpy as np
from functools import partial
import codecs

def rotate_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])

def rotate_single_run(name, srcpath, dstpath):
    """
    only support 0, 90, 180, 270 now
    :param img:
    :param boxes:
    :param angle:
    :return:
    """

    src_imgpath = os.path.join(srcpath, 'images')
    dst_imgpath = os.path.join(dstpath, 'images')

    src_labelTxt = os.path.join(srcpath, 'labelTxt')
    dst_labelTxt = os.path.join(dstpath, 'labelTxt')

    objs = util.parse_dota_poly2(os.path.join(src_labelTxt, name + '.txt'))
    img = cv2.imread(os.path.join(src_imgpath, name + '.png'))
    angle = [np.pi / 2, np.pi, np.pi/2 * 3]

    img_90 = np.rot90(img, 1)
    img_180 = np.rot90(img, 2)
    img_270 = np.rot90(img, 3)

    # cv2.imwrite(os.path.join(dst_imgpath, name + '_90.png'), img_90)
    # cv2.imwrite(os.path.join(dst_imgpath, name + '_180.png'), img_180)
    # cv2.imwrite(os.path.join(dst_imgpath, name + '_270.png'), img_270)

    h, w, c = img.shape
    print('h:', h, 'w:', w, 'c:', c)

    angles = [np.pi/2, np.pi, np.pi/2 * 3]

    rotate_90 = rotate_matrix(np.pi/2)
    rotate_180 = rotate_matrix(np.pi)
    rotate_270 = rotate_matrix(np.pi/2 * 3)


    rotate_90_polys = []
    rotate_180_polys = []
    rotate_270_polys = []

    for obj in objs:
        poly = np.array(obj['poly'])
        poly = np.reshape(poly, newshape=(2, 4), order='F')
        centered_poly = poly - np.array([[w/2.], [h/2.]])
        rotated_poly_90 = np.matmul(rotate_90, centered_poly) + np.array([[h/2.], [w/2.]])
        rotated_poly_180 = np.matmul(rotate_180, centered_poly)+ np.array([[w/2.], [h/2.]])
        rotated_poly_270 = np.matmul(rotate_270, centered_poly) + np.array([[h/2.], [w/2.]])

        rotate_90_polys.append(np.reshape(rotated_poly_90, newshape=(8), order='F'))
        rotate_180_polys.append(np.reshape(rotated_poly_180, newshape=(8), order='F'))
        rotate_270_polys.append(np.reshape(rotated_poly_270, newshape=(8), order='F'))

    with open(os.path.join(dst_labelTxt, name + '_90.txt'), 'w') as f_out:
        for index, poly in enumerate(rotate_90_polys):
            cls = objs[index]['name']
            diff =objs[index]['difficult']
            outline = ' '.join(map(str, list(poly))) + ' ' + cls + ' ' + diff
            f_out.write(outline + '\n')

    with open(os.path.join(dst_labelTxt, name + '_180.txt'), 'w') as f_out:
        for index, poly in enumerate(rotate_180_polys):
            cls = objs[index]['name']
            diff =objs[index]['difficult']
            outline = ' '.join(map(str, list(poly))) + ' ' + cls + ' ' + diff
            f_out.write(outline + '\n')

    with open(os.path.join(dst_labelTxt, name + '_270.txt'), 'w') as f_out:
        for index, poly in enumerate(rotate_270_polys):
            cls = objs[index]['name']
            diff =objs[index]['difficult']
            outline = ' '.join(map(str, list(poly))) + ' ' + cls + ' ' + diff
            f_out.write(outline + '\n')

def rotate(srcpath, dstpath):

    pool = Pool(16)
    imgnames = util.GetFileFromThisRootDir(os.path.join(srcpath, 'images'))
    names = [util.custombasename(x) for x in imgnames]

    dst_imgpath = os.path.join(dstpath, 'images')
    dst_labelTxt = os.path.join(dstpath, 'labelTxt')

    if not os.path.exists(dst_imgpath):
        os.mkdir(dst_imgpath)

    if not os.path.exists(dst_labelTxt):
        os.mkdir(dst_labelTxt)

    rotate_fun = partial(rotate_single_run, srcpath=srcpath, dstpath=dstpath)

    pool.map(rotate_fun, names)

if __name__ == '__main__':
    rotate(r'/data/dj/dota/trainval_large-split-1024',
           r'/data/dj/dota/trainval_large-split_rotate')
