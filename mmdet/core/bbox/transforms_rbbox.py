import mmcv
import numpy as np
import torch
import math
import cv2
import copy


# TODO: check the angle and module operation
def dbbox2delta(proposals, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    """
    :param proposals: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :param gt: (x_ctr, y_ctr, w, h, angle)
    :param means:
    :param stds:
    :return: encoded targets: shape (n, 5)
    """
    proposals = proposals.float()
    gt = gt.float()
    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angle = proposals[..., 4]

    coord = gt[..., 0:2] - proposals[..., 0:2]
    dx = (torch.cos(proposals[..., 4]) * coord[..., 0] +
          torch.sin(proposals[..., 4]) * coord[..., 1]) / proposals_widths
    dy = (-torch.sin(proposals[..., 4]) * coord[..., 0] +
          torch.cos(proposals[..., 4]) * coord[..., 1]) / proposals_heights
    dw = torch.log(gt_widths / proposals_widths)
    dh = torch.log(gt_heights / proposals_heights)
    dangle = (gt_angle - proposals_angle) % (2 * math.pi) / (2 * math.pi)
    deltas = torch.stack((dx, dy, dw, dh, dangle), -1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    # TODO: expand bbox regression
    return deltas

def delta2dbbox(Rrois,
                deltas,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1],
                max_shape=None,
                wh_ratio_clip=16 / 1000):
    """

    :param Rrois: (cx, cy, w, h, theta)
    :param deltas: (dx, dy, dw, dh, dtheta)
    :param means:
    :param stds:
    :param max_shape:
    :param wh_ratio_clip:
    :return:
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
    Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
    Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
    Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
    Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
    # import pdb
    # pdb.set_trace()
    gx = dx * Rroi_w * torch.cos(Rroi_angle) \
         - dy * Rroi_h * torch.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * torch.sin(Rroi_angle) \
         + dy * Rroi_h * torch.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()

    # TODO: check the hard code
    gangle = (2 * np.pi) * dangle + Rroi_angle
    gangle = gangle % ( 2 * np.pi)

    if max_shape is not None:
        pass

    bboxes = torch.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
    return bboxes

def dbbox2delta_v3(proposals, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    """
    This version removes the module operation
    :param proposals: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :param gt: (x_ctr, y_ctr, w, h, angle)
    :param means:
    :param stds:
    :return: encoded targets: shape (n, 5)
    """
    proposals = proposals.float()
    gt = gt.float()
    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angle = proposals[..., 4]

    coord = gt[..., 0:2] - proposals[..., 0:2]
    dx = (torch.cos(proposals[..., 4]) * coord[..., 0] +
          torch.sin(proposals[..., 4]) * coord[..., 1]) / proposals_widths
    dy = (-torch.sin(proposals[..., 4]) * coord[..., 0] +
          torch.cos(proposals[..., 4]) * coord[..., 1]) / proposals_heights
    dw = torch.log(gt_widths / proposals_widths)
    dh = torch.log(gt_heights / proposals_heights)
    # import pdb
    # print('in dbbox2delta v3')
    # pdb.set_trace()
    # dangle = (gt_angle - proposals_angle) % (2 * math.pi) / (2 * math.pi)
    # TODO: debug for it, proposals_angle are -1.5708, gt_angle should close to -1.57, actully they close to 5.0153
    dangle = gt_angle - proposals_angle
    deltas = torch.stack((dx, dy, dw, dh, dangle), -1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta2dbbox_v3(Rrois,
                deltas,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1],
                max_shape=None,
                wh_ratio_clip=16 / 1000):
    """
    This version removes the module operation
    :param Rrois: (cx, cy, w, h, theta)
    :param deltas: (dx, dy, dw, dh, dtheta)
    :param means:
    :param stds:
    :param max_shape:
    :param wh_ratio_clip:
    :return:
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
    Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
    Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
    Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
    Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
    # import pdb
    # pdb.set_trace()
    gx = dx * Rroi_w * torch.cos(Rroi_angle) \
         - dy * Rroi_h * torch.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * torch.sin(Rroi_angle) \
         + dy * Rroi_h * torch.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()

    # TODO: check the hard code
    # gangle = (2 * np.pi) * dangle + Rroi_angle
    gangle = dangle + Rroi_angle
    # gangle = gangle % ( 2 * np.pi)

    if max_shape is not None:
        pass

    bboxes = torch.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
    return bboxes

def dbbox2delta_v2(proposals, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    """
    :param proposals: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :param gt: (x_ctr, y_ctr, w, h, angle)
    :param means:
    :param stds:
    :return: encoded targets: shape (n, 5)
    """
    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    roi_widths = proposals[..., 2]
    roi_heights = proposals[..., 3]
    roi_angle = proposals[..., 4]

    coord = gt[..., 0:2] - proposals[..., 0:2]
    targets_dx = (torch.cos(roi_angle) * coord[..., 0] + torch.sin(roi_angle) * coord[:, 1]) / roi_widths
    targets_dy = (-torch.sin(roi_angle) * coord[..., 0] + torch.cos(roi_angle) * coord[:, 1]) / roi_heights
    targets_dw = torch.log(gt_widths / roi_widths)
    targets_dh = torch.log(gt_heights / roi_heights)
    targets_dangle = (gt_angle - roi_angle)
    dist = targets_dangle % (2 * np.pi)
    dist = torch.min(dist, np.pi * 2 - dist)
    try:
        assert np.all(dist.cpu().numpy() <= (np.pi/2. + 0.001) )
    except:
        import pdb
        pdb.set_trace()

    inds = torch.sin(targets_dangle) < 0
    dist[inds] = -dist[inds]
    # TODO: change the norm value
    dist = dist / (np.pi / 2.)
    deltas = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, dist), -1)


    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta2dbbox_v2(Rrois,
                deltas,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1],
                max_shape=None,
                wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
    Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
    Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
    Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
    Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)

    gx = dx * Rroi_w * torch.cos(Rroi_angle) \
         - dy * Rroi_h * torch.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * torch.sin(Rroi_angle) \
         + dy * Rroi_h * torch.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()

    gangle = (np.pi / 2.) * dangle + Rroi_angle

    if max_shape is not None:
        # TODO: finish it
        pass

    bboxes = torch.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
    return bboxes

def choose_best_match_batch(Rrois, gt_rois):
    """
    choose best match representation of gt_rois for a Rrois
    :param Rrois: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :param gt_rois: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :return: gt_roi_news: gt_roi with new representation
            shape: (n, 5)
    """
    # TODO: check the dimensions
    Rroi_angles = Rrois[:, 4].unsqueeze(1)

    gt_xs, gt_ys, gt_ws, gt_hs, gt_angles = copy.deepcopy(gt_rois[:, 0]), copy.deepcopy(gt_rois[:, 1]), \
                                            copy.deepcopy(gt_rois[:, 2]), copy.deepcopy(gt_rois[:, 3]), \
                                            copy.deepcopy(gt_rois[:, 4])

    gt_angle_extent = torch.cat((gt_angles[:, np.newaxis], (gt_angles + np.pi/2.)[:, np.newaxis],
                                      (gt_angles + np.pi)[:, np.newaxis], (gt_angles + np.pi * 3/2.)[:, np.newaxis]), 1)
    dist = (Rroi_angles - gt_angle_extent) % (2 * np.pi)
    dist = torch.min(dist, np.pi * 2 - dist)
    min_index = torch.argmin(dist, 1)

    gt_rois_extent0 = copy.deepcopy(gt_rois)
    gt_rois_extent1 = torch.cat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_hs.unsqueeze(1), gt_ws.unsqueeze(1), gt_angles.unsqueeze(1) + np.pi/2.), 1)
    gt_rois_extent2 = torch.cat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_ws.unsqueeze(1), gt_hs.unsqueeze(1), gt_angles.unsqueeze(1) + np.pi), 1)
    gt_rois_extent3 = torch.cat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_hs.unsqueeze(1), gt_ws.unsqueeze(1), gt_angles.unsqueeze(1) + np.pi * 3/2.), 1)
    gt_rois_extent = torch.cat((gt_rois_extent0.unsqueeze(1),
                                     gt_rois_extent1.unsqueeze(1),
                                     gt_rois_extent2.unsqueeze(1),
                                     gt_rois_extent3.unsqueeze(1)), 1)

    gt_rois_new = torch.zeros_like(gt_rois)
    # TODO: add pool.map here
    for curiter, index in enumerate(min_index):
        gt_rois_new[curiter, :] = gt_rois_extent[curiter, index, :]

    gt_rois_new[:, 4] = gt_rois_new[:, 4] % (2 * np.pi)

    return gt_rois_new

def choose_best_Rroi_batch(Rroi):
    """
    There are many instances with large aspect ratio, so we choose the point, previous is long side,
    after is short side, so it makes sure h < w
    then angle % 180,
    :param Rroi: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :return: Rroi_new: Rroi with new representation
    """
    x_ctr, y_ctr, w, h, angle = copy.deepcopy(Rroi[:, 0]), copy.deepcopy(Rroi[:, 1]), \
                                copy.deepcopy(Rroi[:, 2]), copy.deepcopy(Rroi[:, 3]), copy.deepcopy(Rroi[:, 4])
    indexes = w < h

    Rroi[indexes, 2] = h[indexes]
    Rroi[indexes, 3] = w[indexes]
    Rroi[indexes, 4] = Rroi[indexes, 4] + np.pi / 2.
    # TODO: check the module
    Rroi[:, 4] = Rroi[:, 4] % np.pi

    return Rroi

def best_match_dbbox2delta(Rrois, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    """
    :param Rrois: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :param gt: (x_ctr, y_ctr, w, h, angle)
    :param means:
    :param stds:
    :return: encoded targets: shape (n, 5)
    """
    # TODO: for comparision, dot not change the regression range for angle in 2 stage currently
    #  This is a simplified version
    # TODO: preprocess angle of gt_boxes according to the catogeries
    # Here, use a choose beste match angle, similar to choose best point instead
    gt_boxes_new = choose_best_match_batch(Rrois, gt)
    try:
        assert np.all(Rrois.cpu().numpy()[:, 4] <= (np.pi + 0.001))
    except:
        import pdb
        pdb.set_trace()
    bbox_targets = dbbox2delta_v2(Rrois, gt_boxes_new, means, stds)

    return bbox_targets

# TODO: check the negative situation of flip
def dbbox_flip(dbboxes, img_shape):
    """
    Flip dbboxes horizontally
    :param dbboxes: (Tensor): Shape (..., 5*k), (x_ctr, y_ctr, w, h, angle)
    :param img_shape: (tuple): Image shape.
    :return: Same type as 'dbboxes': Flipped dbboxes
    """
    assert dbboxes.shape[-1] % 5 == 0
    flipped = dbboxes.clone()
    # flip x
    flipped[:, 0::5] = img_shape[1] - dbboxes[:, 0::5] - 1
    # flip angle
    flipped[:, 4::5] = np.pi - dbboxes[:, 4::5]

    return flipped

def dbbox_mapping(dbboxes, img_shape, scale_factor, flip):
    """
    Map dbboxes from testing scale to original image scale
    :param dbboxes:
    :param img_shape:
    :param scale_factor:
    :param flip:
    :return:
    """
    new_dbboxes = dbboxes.clone()
    new_dbboxes[..., 0::5] = dbboxes[..., 0::5] * scale_factor
    new_dbboxes[..., 1::5] = dbboxes[..., 1::5] * scale_factor
    new_dbboxes[..., 2::5] = dbboxes[..., 2::5] * scale_factor
    new_dbboxes[..., 3::5] = dbboxes[..., 3::5] * scale_factor
    if flip:
        new_dbboxes = dbbox_flip(new_dbboxes, img_shape)

    return new_dbboxes

def dbbox_mapping_back(dbboxes, img_shape, scale_factor, flip):
    """
    Map dbboxes from testing scael to original image scale
    :param dbboxes:
    :param img_shape:
    :param scale_factor:
    :param flip:
    :return:
    """
    new_dbboxes = dbbox_flip(dbboxes, img_shape) if flip else dbboxes
    new_dbboxes[..., 0::5] = new_dbboxes[..., 0::5] / scale_factor
    new_dbboxes[..., 1::5] = new_dbboxes[..., 1::5] / scale_factor
    new_dbboxes[..., 2::5] = new_dbboxes[..., 2::5] / scale_factor
    new_dbboxes[..., 3::5] = new_dbboxes[..., 3::5] / scale_factor
    return new_dbboxes

def dbbox_rotate_mapping(bboxes, img_shape, angle):
    """
        map bboxes from the original image angle to testing angle
        only support descrete angle currently,
        do not consider the single class currently, do not consider batch images currently
    :param bboxes: [n, 5 * #C] (x, y, w, h, theta) repeat #C
    :param img_shape:
    :param angle: angle in degeree
    :return:
    """
    # print('angle: ', angle)
    assert angle in [0, 90, 180, 270, -90, -180, -270]
    assert len(bboxes.size()) == 2
    num = bboxes.size(0)
    h, w = img_shape[:2]
    if angle in [90, 270] :
        new_h, new_w = w, h
    else:
        new_h, new_w = h, w
    center = torch.FloatTensor([(w) * 0.5, (h) * 0.5]).to(bboxes.device)

    # import pdb; pdb.set_trace()
    xys = torch.cat((bboxes[..., 0::5].view(-1, 1), bboxes[..., 1::5].view(-1, 1)), -1)
    norm_xys = xys - center

    rotate_matrix = torch.FloatTensor([[np.cos(angle/180 * np.pi), np.sin(angle/180 * np.pi)],
                              [-np.sin(angle/180 * np.pi), np.cos(angle/180 * np.pi)]]).to(bboxes.device)

    norm_rotated_xys = torch.matmul(norm_xys, rotate_matrix)

    new_center = torch.FloatTensor([(new_w) * 0.5, (new_h) * 0.5]).to(bboxes.device)
    rotated_xys = norm_rotated_xys + new_center

    rotated_xys = rotated_xys.view(num, -1)
    rotated_dbboxes = torch.zeros(bboxes.size()).to(bboxes.device)
    rotated_dbboxes[..., 0::5] = rotated_xys[..., 0::2]
    rotated_dbboxes[..., 1::5] = rotated_xys[..., 1::2]
    rotated_dbboxes[..., 2::5] = bboxes[..., 2::5]
    rotated_dbboxes[..., 3::5] = bboxes[..., 3::5]
    rotated_dbboxes[..., 4::5] = bboxes[..., 4::5] + angle/180 * np.pi

    return rotated_dbboxes


def bbox_rotate_mapping(bboxes, img_shape, angle):
    """TODO: test this code
        map bboxes from the original image angle to testing angle
        only support descrete angle currently,
        do not consider the single class currently, do not consider batch images currently
    :param bboxes: [n, 4 * #C] (xmin, ymin, xmax, ymax) repeat #C
    :param img_shape:
    :param angle: angle in degeree
    :return:
    """
    assert angle in [0, 90, 180, 270, -90, -180, -270]
    assert len(bboxes.size()) == 2
    num = bboxes.size(0)
    h, w = img_shape[:2]
    if angle in [90, 270]:
        new_h, new_w = w, h
    else:
        new_h, new_w = h, w
    # TODO: check (w - 1) or (w)
    center = torch.FloatTensor([(w) * 0.5, (h) * 0.5]).to(bboxes.device)

    c_bboxes = xy2wh_c(bboxes)

    if angle in [90, 270]:
        new_box_hs, new_box_ws = c_bboxes[..., 2::4], c_bboxes[..., 3::4]
    else:
        new_box_hs, new_box_ws = c_bboxes[..., 3::4], c_bboxes[..., 2::4]

    xys = torch.cat((c_bboxes[..., 0::4].view(-1, 1), c_bboxes[..., 1::4].view(-1, 1)), -1)
    norm_xys = xys - center

    rotate_matrix = torch.FloatTensor([[np.cos(angle / 180 * np.pi), np.sin(angle / 180 * np.pi)],
                                       [-np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)]]).to(bboxes.device)

    norm_rotated_xys = torch.matmul(norm_xys, rotate_matrix)

    new_center = torch.FloatTensor([(new_w) * 0.5, (new_h) * 0.5]).to(bboxes.device)
    rotated_xys = norm_rotated_xys + new_center

    rotated_xys = rotated_xys.view(num, -1)
    rotated_cbboxes = torch.zeros(bboxes.size()).to(bboxes.device)
    rotated_cbboxes[..., 0::4] = rotated_xys[..., 0::2]
    rotated_cbboxes[..., 1::4] = rotated_xys[..., 1::2]
    rotated_cbboxes[..., 2::4] = new_box_ws
    rotated_cbboxes[..., 3::4] = new_box_hs

    rotated_bboxes = wh2xy_c(rotated_cbboxes)

    return rotated_bboxes


def dbbox2delta_warp(proposals, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    """
    :param proposals: (xmin, ymin, xmax, ymax)
    :param gt: (x1, y_ctr, w, h, angle)
    :param means:
    :param stds:
    :return:
    """

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def Tuplelist2Polylist(tuple_poly_list):
    polys = map(TuplePoly2Poly, tuple_poly_list)

    return list(polys)
#
# def mask2poly_single(binary_mask):
#     """
#
#     :param binary_mask:
#     :return:
#     """
#     # try:
#     contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     contour_lens = np.array(list(map(len, contours)))
#     max_id = contour_lens.argmax()
#     max_contour = contours[max_id]
#     rect = cv2.minAreaRect(max_contour)
#     poly = cv2.boxPoints(rect)
#     # poly = TuplePoly2Poly(poly)
#
#     return poly
    # except:
    #     # TODO: assure there is no empty mask_poly
    #     return []

# TODO: test the function
def mask2poly_single(binary_mask):
    """

    :param binary_mask:
    :return:
    """
    try:
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contour_lens = np.array(list(map(len, contours)))
        # max_id = contour_lens.argmax()
        # max_contour = contours[max_id]
        max_contour = max(contours, key=len)
        rect = cv2.minAreaRect(max_contour)
        poly = cv2.boxPoints(rect)
        # poly = TuplePoly2Poly(poly)
    except:
        import pdb
        pdb.set_trace()
    return poly

def mask2poly(binary_mask_list):
    polys = map(mask2poly_single, binary_mask_list)
    # polys = np.stack(polys
    return list(polys)

def gt_mask_bp_obbs(gt_masks, with_module=True):

    # trans gt_masks to gt_obbs
    gt_polys = mask2poly(gt_masks)
    gt_bp_polys = get_best_begin_point(gt_polys)
    gt_obbs = polygonToRotRectangle_batch(gt_bp_polys, with_module)

    return gt_obbs

def gt_mask_bp_obbs_list(gt_masks_list):

    gt_obbs_list = map(gt_mask_bp_obbs, gt_masks_list)

    return list(gt_obbs_list)

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
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
        pass
        # print("choose one direction!")
    return  combinate[force_flag]

def get_best_begin_point_warp_single(coordinate):

    return TuplePoly2Poly(get_best_begin_point_single(coordinate))

def get_best_begin_point(coordinate_list):
    best_coordinate_list = map(get_best_begin_point_warp_single, coordinate_list)
    # import pdb
    # pdb.set_trace()
    best_coordinate_list = np.stack(list(best_coordinate_list))

    return best_coordinate_list

# def polygonToRotRectangle(polys):
#     """
#     pytorch version, batch operation
#     :param polys: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
#             shape [num_boxes, 8]
#     :return: Rotated Rectangle in format [cx, cy, w, h, theta]
#             shape [num_rot_recs, 5]
#     """
#     polys = polys.view(-1, 4, 2)

def xy2wh(boxes):
    """

    :param boxes: (xmin, ymin, xmax, ymax) (n, 4)
    :return: out_boxes: (x_ctr, y_ctr, w, h) (n, 4)
    """
    num_boxes = boxes.size(0)

    ex_widths = boxes[..., 2] - boxes[..., 0] + 1.0
    ex_heights = boxes[..., 3] - boxes[..., 1] + 1.0
    ex_ctr_x = boxes[..., 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = boxes[..., 1] + 0.5 * (ex_heights - 1.0)

    return torch.cat((ex_ctr_x.unsqueeze(1), ex_ctr_y.unsqueeze(1), ex_widths.unsqueeze(1), ex_heights.unsqueeze(1)), 1)

def xy2wh_c(boxes):
    """

    :param boxes: (xmin, ymin, xmax, ymax) (n, 4 * #C)
    :return: out_boxes: (x_ctr, y_ctr, w, h) (n, 4 * #C)
    """
    num_boxes = boxes.size(0)
    out_boxes = boxes.clone()
    ex_widths = boxes[..., 2::4] - boxes[..., 0::4] + 1.0
    ex_heights = boxes[..., 3::4] - boxes[..., 1::4] + 1.0
    ex_ctr_x = boxes[..., 0::4] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = boxes[..., 1::4] + 0.5 * (ex_heights - 1.0)
    # import pdb; pdb.set_trace()
    out_boxes[..., 2::4] = ex_widths
    out_boxes[..., 3::4] = ex_heights
    out_boxes[..., 0::4] = ex_ctr_x
    out_boxes[..., 1::4] = ex_ctr_y

    return out_boxes

def wh2xy(bboxes):
    """
    :param bboxes: (x_ctr, y_ctr, w, h) (n, 4)
    :return: out_bboxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    num_boxes = bboxes.size(0)

    xmins = bboxes[..., 0] - (bboxes[..., 2] - 1) / 2.0
    ymins = bboxes[..., 1] - (bboxes[..., 3] - 1) / 2.0
    xmaxs = bboxes[..., 0] + (bboxes[..., 2] - 1) / 2.0
    ymaxs = bboxes[..., 1] + (bboxes[..., 3] - 1) / 2.0

    return torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1), xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), 1)

def wh2xy_c(bboxes):
    """
    :param bboxes: (x_ctr, y_ctr, w, h) (n, 4 * #C)
    :return: out_bboxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    num_boxes = bboxes.size(0)
    out_bboxes = bboxes.clone()
    xmins = bboxes[..., 0::4] - (bboxes[..., 2::4] - 1) / 2.0
    ymins = bboxes[..., 1::4] - (bboxes[..., 3::4] - 1) / 2.0
    xmaxs = bboxes[..., 0::4] + (bboxes[..., 2::4] - 1) / 2.0
    ymaxs = bboxes[..., 1::4] + (bboxes[..., 3::4] - 1) / 2.0

    out_bboxes[..., 0::4] = xmins
    out_bboxes[..., 1::4] = ymins
    out_bboxes[..., 2::4] = xmaxs
    out_bboxes[..., 3::4] = ymaxs
    return out_bboxes

def hbb2obb(bboxes):
    """

    :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    num_boxes = bboxes.size(0)
    c_bboxes = xy2wh(bboxes)
    initial_angles = -c_bboxes.new_ones((num_boxes, 1)) * np.pi / 2
    # initial_angles = -torch.ones((num_boxes, 1)) * np.pi/2
    dbboxes = torch.cat((c_bboxes, initial_angles), 1)

    return dbboxes

def hbb2obb_v2(boxes):
    """
    fix a bug
    :param boxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    num_boxes = boxes.size(0)
    ex_heights = boxes[..., 2] - boxes[..., 0] + 1.0
    ex_widths = boxes[..., 3] - boxes[..., 1] + 1.0
    ex_ctr_x = boxes[..., 0] + 0.5 * (ex_heights - 1.0)
    ex_ctr_y = boxes[..., 1] + 0.5 * (ex_widths - 1.0)
    c_bboxes = torch.cat((ex_ctr_x.unsqueeze(1), ex_ctr_y.unsqueeze(1), ex_widths.unsqueeze(1), ex_heights.unsqueeze(1)), 1)
    initial_angles = -c_bboxes.new_ones((num_boxes, 1)) * np.pi / 2
    # initial_angles = -torch.ones((num_boxes, 1)) * np.pi/2
    dbboxes = torch.cat((c_bboxes, initial_angles), 1)

    return dbboxes

def roi2droi(rois):
    """
    :param rois: Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    :return: drois: Tensor: shape (n, 6), [batch_ind, x, y, w, h, theta]
    """
    hbbs = rois[:, 1:]
    obbs = hbb2obb_v2(hbbs)

    return torch.cat((rois[:, 0].unsqueeze(1), obbs), 1)

def polygonToRotRectangle_batch(bbox, with_module=True):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]
    """
    # print('bbox: ', bbox)
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)


    xmin = np.min(normalized[:, 0, :], axis=1)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[:, 0, :], axis=1)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    # TODO: check it
    if with_module:
        angle = angle[:, np.newaxis] % ( 2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes

def RotBox2Polys(dboxes):
    """
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    cs = np.cos(dboxes[:, 4])
    ss = np.sin(dboxes[:, 4])
    w = dboxes[:, 2] - 1
    h = dboxes[:, 3] - 1

    ## change the order to be the initial definition
    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    x1 = x1[:, np.newaxis]
    y1 = y1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    y2 = y2[:, np.newaxis]
    x3 = x3[:, np.newaxis]
    y3 = y3[:, np.newaxis]
    x4 = x4[:, np.newaxis]
    y4 = y4[:, np.newaxis]

    polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)
    return polys

def RotBox2Polys_torch(dboxes):
    """

    :param dboxes:
    :return:
    """
    cs = torch.cos(dboxes[:, 4])
    ss = torch.sin(dboxes[:, 4])
    w = dboxes[:, 2] - 1
    h = dboxes[:, 3] - 1

    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    polys = torch.cat((x1.unsqueeze(1),
                       y1.unsqueeze(1),
                       x2.unsqueeze(1),
                       y2.unsqueeze(1),
                       x3.unsqueeze(1),
                       y3.unsqueeze(1),
                       x4.unsqueeze(1),
                       y4.unsqueeze(1)), 1)

    return polys


def poly2bbox(polys):
    """
    without label
    :param polys: (x1, y1, ..., x4, y4) (n, 8)
    :return: boxes: (xmin, ymin, xmax, ymax) (n, 4)
    """
    n = polys.shape[0]
    xs = np.reshape(polys, (n, 4, 2))[:, :, 0]
    ys = np.reshape(polys, (n, 4, 2))[:, :, 1]

    xmin = np.min(xs, axis=1)
    ymin = np.min(ys, axis=1)
    xmax = np.max(xs, axis=1)
    ymax = np.max(ys, axis=1)

    xmin = xmin[:, np.newaxis]
    ymin = ymin[:, np.newaxis]
    xmax = xmax[:, np.newaxis]
    ymax = ymax[:, np.newaxis]

    return np.concatenate((xmin, ymin, xmax, ymax), 1)

def dbbox2roi(dbbox_list):
    """
    Convert a list of dbboxes to droi format.
    :param dbbox_list: (list[Tensor]): a list of dbboxes corresponding to a batch of images
    :return: Tensor: shape (n, 6) [batch_ind, x_ctr, y_ctr, w, h, angle]
    """
    drois_list = []
    for img_id, dbboxes in enumerate(dbbox_list):
        if dbboxes.size(0) > 0:
            img_inds = dbboxes.new_full((dbboxes.size(0), 1), img_id)
            drois = torch.cat([img_inds, dbboxes[:, :5]], dim=-1)
        else:
            drois = dbboxes.new_zeros((0, 6))

        drois_list.append(drois)
    drois = torch.cat(drois_list, 0)
    return drois

def droi2dbbox(drois):
    dbbox_list = []
    img_ids = torch.unique(drois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (drois[:, 0] == img_id.item())
        dbbox = drois[inds, 1:]
        dbbox_list.append(dbbox)
    return dbbox_list

def dbbox2result(dbboxes, labels, num_classes):
    """
    Convert detection results to a list of numpy arrays.
    :param dbboxes: (Tensor): shape (n, 9)
    :param labels:  (Tensor): shape (n, )
    :param num_classes: (int), class number, including background class
    :return: list (ndarray): dbbox results of each class
    """
    # TODO: merge it with bbox2result
    if dbboxes.shape[0] == 0:
        return [
            np.zeros((0, 9), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        dbboxes = dbboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [dbboxes[labels == i, :] for i in range(num_classes - 1)]

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)
