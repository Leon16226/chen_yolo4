import torch
import random
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms.functional as torchtransform
from utils.metrics.metrics import bbox_iou
import numpy as np
import torch.nn.functional as F
import math
import cv2

def to_heatmap(data, scale_factor=4, cls_num=10):
    """
    Transform annotations to heatmap.
    :param data: (img, annos), tensor
    :param scale_factor:
    :param cls_num:
    :return:
    """
    # init
    img = data[0]
    annos = data[1].clone()
    h, w = img.size(1), img.size(2)

    hm = torch.zeros(cls_num, h // scale_factor, w // scale_factor)

    # annos
    annos[:, 2] += annos[:, 0]
    annos[:, 3] += annos[:, 1]
    annos[:, :4] = annos[:, :4] / scale_factor
    cls_idx = annos[:, 5] - 1
    bboxs_h, bboxs_w = annos[:, 3:4] - annos[:, 1:2], annos[:, 2:3] - annos[:, 0:1]

    wh = torch.cat([bboxs_w, bboxs_h], dim=1)

    # center
    ct = torch.cat(((annos[:, 0:1] + annos[:, 2:3]) / 2., (annos[:, 1:2] + annos[:, 3:4]) / 2.), dim=1)
    ct_int = ct.floor()
    offset = ct - ct_int

    reg_mask = ((bboxs_h > 0) * (bboxs_w > 0))

    ind = ct_int[:, 1:2] * (w // 4) + ct_int[:, 0:1]

    # radius------------------------------------------------------------------------------------------------------------
    radius = gaussian_radius((bboxs_h.ceil(), bboxs_w.ceil()))
    radius = radius.floor().clamp(min=0)
    for k, cls in enumerate(cls_idx):
        draw_umich_gaussian(hm[cls.long().item()], ct_int[k], radius[k])

    return data[0], data[1], hm, wh, ind, offset, reg_mask

# radius----------------------------------------------------------------------------------------------------------------
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2.

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    r = torch.cat((r1, r2, r3), dim=1).min(dim=1)[0]
    return r

def gaussian2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    m = m.numpy()
    n = n.numpy()
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma.numpy() * sigma.numpy()))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h = torch.from_numpy(h).float()
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1

    gaussian = gaussian2d((diameter, diameter), sigma=diameter / 6)

    x, y = center[0], center[1]

    height, width = heatmap.size()[0:2]
    left, right = torch.min(x, radius), torch.min(width - x, radius + 1)
    top, bottom = torch.min(y, radius), torch.min(height - y, radius + 1)

    masked_heatmap = heatmap[int(y - top):int(y + bottom), int(x - left):int(x + right)]
    masked_gaussian = gaussian[int(radius - top):int(radius + bottom), int(radius - left):int(radius + right)]
    if min(list(masked_gaussian.size())) > 0 and min(list(masked_heatmap.size())) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap