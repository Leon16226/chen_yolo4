import random
import torch
from torch.nn.functional import pad
import PIL
import numpy as np
from . import functional as F
from torch.nn.functional import interpolate


class ToHeatmap(object):
    def __init__(self, scale_factor=4, cls_num=10):
        self.scale_factor = scale_factor
        self.cls_num = cls_num

    def __call__(self, data):
        img, annos, hm, wh, ind, offset, reg_mask = F.to_heatmap(data, self.scale_factor, self.cls_num)
        return img, annos, hm, wh, ind, offset, reg_mask