import math
import sys
import time
from functools import partial

import numpy as np
import scipy
import torch as th
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize

from utils import info


def im2tens(im, h, w):
    im = resize(im, (h, w))
    tens = th.FloatTensor(im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
    return tens


def predict(model, im1, im2, h, w):
    h, w, c = im1.shape
    tens1 = im2tens(im1, h, w)
    tens2 = im2tens(im2, h, w)
    model_out = model(tens1, tens2).unsqueeze(0)
    output = F.interpolate(input=model_out, size=(h, w), mode="bilinear", align_corners=False)[0, :, :, :]
    return output.cpu().numpy().transpose(1, 2, 0)


def get_flow_model(args):
    models = []

    if "unflow" in args.flow_models:
        del sys.argv[1:]
        sys.path.append("unflow/correlation")
        from sniklaus.unflow.run import estimate

        del sys.path[-1]
        th.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away
        models += lambda im1, im2: predict(estimate, im1, im2, h=384, w=1280)

    if "pwc" in args.flow_models:
        del sys.argv[1:]
        sys.path.append("pwc/correlation")
        from sniklaus.pwc.run import estimate

        del sys.path[-1]
        th.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away
        models += lambda im1, im2: predict(estimate, im1, im2, h=436, w=1024)

    if "spynet" in args.flow_models:
        del sys.argv[1:]
        from sniklaus.spynet.run import estimate

        th.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away
        models += lambda im1, im2: predict(estimate, im1, im2, h=416, w=1024)

    if "liteflownet" in args.flow_models:
        del sys.argv[1:]
        sys.path.append("liteflownet/correlation")
        from sniklaus.liteflownet.run import estimate

        del sys.path[-1]
        th.set_grad_enabled(True)  # estimate run.py disables grads, so re-enable right away
        models += lambda im1, im2: predict(estimate, im1, im2, h=436, w=1024)

    if "deepflow2" in args.flow_models:
        raise Exception("deepflow2 not working quite yet...")
        from thoth.deepflow2 import deepflow2
        from thoth.deepmatching import deepmatching

        models += lambda im1, im2: deepflow2(im1, im2, deepmatching(im1, im2))

    return lambda im1, im2: sum(model(im1, im2) for model in models)


def check_consistency(flow1, flow2):
    flow1 = np.flip(flow1, axis=2)
    flow2 = np.flip(flow2, axis=2)
    h, w, _ = flow1.shape

    # get grid of coordinates for each pixel
    orig_coord = np.flip(np.mgrid[:w, :h], 0).T

    # find where the flow1 maps each pixel
    warp_coord = orig_coord + flow1

    # clip the coordinates in bounds and round down
    warp_coord_inbound = np.zeros_like(warp_coord)
    warp_coord_inbound[:, :, 0] = np.clip(warp_coord[:, :, 0], 0, h - 2)
    warp_coord_inbound[:, :, 1] = np.clip(warp_coord[:, :, 1], 0, w - 2)
    warp_coord_floor = np.floor(warp_coord_inbound).astype(np.int)

    # for each pixel: bilinear interpolation of the corresponding flow2 values around the point mapped to by flow1
    alpha = warp_coord_inbound - warp_coord_floor
    flow2_00 = flow2[warp_coord_floor[:, :, 0], warp_coord_floor[:, :, 1]]
    flow2_01 = flow2[warp_coord_floor[:, :, 0], warp_coord_floor[:, :, 1] + 1]
    flow2_10 = flow2[warp_coord_floor[:, :, 0] + 1, warp_coord_floor[:, :, 1]]
    flow2_11 = flow2[warp_coord_floor[:, :, 0] + 1, warp_coord_floor[:, :, 1] + 1]
    flow2_0_blend = (1 - alpha[:, :, 1, None]) * flow2_00 + alpha[:, :, 1, None] * flow2_01
    flow2_1_blend = (1 - alpha[:, :, 1, None]) * flow2_10 + alpha[:, :, 1, None] * flow2_11
    warp_coord_flow2 = (1 - alpha[:, :, 0, None]) * flow2_0_blend + alpha[:, :, 0, None] * flow2_1_blend

    # coordinates that flow2 remaps each flow1-mapped pixel to
    rewarp_coord = warp_coord + warp_coord_flow2

    # where the difference in position after flow1 and flow2 are applied is larger than a threshold there is likely an
    # occlusion. set values to -1 so the final gaussian blur will spread the value a couple pixels around this area
    squared_diff = np.sum((rewarp_coord - orig_coord) ** 2, axis=2)
    threshold = 0.01 * np.sum(warp_coord_flow2 ** 2 + flow1 ** 2, axis=2) + 0.5
    reliable_flow = np.where(squared_diff >= threshold, -1, 1)

    # areas mapping outside of the frame are also occluded (don't need extra region around these though, so set 0)
    reliable_flow = np.where(
        np.logical_or.reduce(
            (
                warp_coord[:, :, 0] < 0,
                warp_coord[:, :, 1] < 0,
                warp_coord[:, :, 0] >= h - 1,
                warp_coord[:, :, 1] >= w - 1,
            )
        ),
        0,
        reliable_flow,
    )

    # get derivative of flow, large changes in derivative => edge of moving object
    dx = np.diff(flow1, axis=1, append=0)
    dy = np.diff(flow1, axis=0, append=0)
    motion_edge = np.sum(dx ** 2 + dy ** 2, axis=2)
    motion_threshold = 0.01 * np.sum(flow1 ** 2, axis=2) + 0.002
    reliable_flow = np.where(np.logical_and(motion_edge > motion_threshold, reliable_flow != -1), 0, reliable_flow)

    # blur and clip values between 0 and 1
    reliable_flow = scipy.ndimage.gaussian_filter(reliable_flow, [5, 5])
    reliable_flow = reliable_flow.clip(0, 1)
    return reliable_flow


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    UNKNOWN_FLOW_THRESH = 1e7

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col : col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col : col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col : col + MR, 0] = 255

    return colorwheel
