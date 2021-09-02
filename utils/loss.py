import glob
import math
import os
import random
import shutil
import subprocess
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from sys import platform

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from utils.torch_utils import init_seeds, is_parallel
import pkg_resources as pkg
from subprocess import check_output
import logging
from general import *


# focal loss
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        # loss -> true : target 0~1
        loss = self.loss_fcn(pred, true)
        # prob from logits
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # ??? easy -> weighted sum
        # factor -> exp -> gamma should large
        modulating_factor = (1.0 - p_t) ** self.gamma
        # alpha
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # ??? -> weights sum
        loss *= alpha_factor * modulating_factor

        # reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


def compute_loss(p, targets, model):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp

    # small proportion -------------------------------------------------------------------------------------------------
    lbox_s = torch.zeros(1, device=device)

    # loss
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)
    cp, cn = smooth_BCE(eps=0.0)
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # small proportion------------------------------------------------------------------------------------------
            pbox_small, tbox_small = [], []
            small_thresh = 0.0500
            for j, box in enumerate(ps[:, 2:4]):
                if(box[0] < small_thresh and box[1] < small_thresh):
                    pbox_small.append(pbox[j].cpu().detach().numpy())
                    tbox_small.append(tbox[i][j].cpu().detach().numpy())
            if(len(pbox_small) > 0):
                pbox_small = torch.tensor(pbox_small).cuda()
                tbox_small = torch.tensor(tbox_small).cuda()
                giou_small = bbox_iou(pbox_small.T, tbox_small, x1y1x2y2=False, CIoU=True)
                lbox_s += (1.0 - giou_small).mean()


            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling

    lbox *= h['giou'] * s
    lbox_s *= h['giou'] * s
    lobj *= h['obj'] * s * (1.4 if np == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach(), 1 - lbox_s / lbox


# label encode ---------------------------------------------------------------------------------------------------------
def build_targets_yolox(p, targets, model):

    # p : [[16, 3, 52, 52, 3], [16, 3, 26, 26, 3]]
    # targets: [[[0, cx, cy, w, h],....],
    #           [[0, cx, cy, w, h],....]]

    # init -------------------------------------------------------------------------------------------------------------
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets
    g = 0.5  # offset

    # do
    for i, jj in enumerate(model.yolo_layers):
        # [[10,20],.....]
        anchors = model.module_list[jj].anchor_vec  # list
        # gain : [1, 1, 52, 52, 52, 52]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # (x, y, x, y)

        # Match targets to anchors--------------------------------------------------------------------------------------
        # t : targets's xywh * 52
        # tensor([[0.0000, 0.0000, 2.4000, 4.8000, 7.2000, 9.6000],
        #       [ 0.0000,  1.0000,  3.6000,  6.0000, 19.2000, 21.6000]])
        a, t, offsets = [], targets * gain, 0  # t : targets in current feature map
        if nt:
            na = anchors.shape[0]  # 3
            # tensor([[0, 0,],
            #         [1, 1,],
            #         [2, 2,]])
            at = torch.arange(na).view(na, 1).repeat(1, nt)  # repeat
            # tensor([[[1.4400, 0.9600],
            #          [3.8400, 2.1600]],
            #
            #         [[3.6000, 0.9600],
            #          [9.6000, 2.1600]],
            #
            #         [[1.8000, 1.6000],
            #          [4.8000, 3.6000]]])
            r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            # 每个位置取最大值 filter-------------------------------------------------------------------------------------
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define--------------------------------------------------------------------------------------------------------
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

        # tcls : [???]
        # tbox : [[gx, gy, gw, gh] .....]

    return tcls, tbox, indices, anch


# nms -> conf & iou -> the shape of input same as outputs'
def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    # prediction -> (batch, 13x13 + 26x26 + 52x52 = 3549, 7) -> (x1, y1, x2, y2, conf, cls) -> cls : no_mask, mask
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates -> 2dim

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image -> handcraft
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]  # the array of output
    # iter -> x:[?, 7]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence select

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T  # find index with the pra != 0
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # nms
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        # limit
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # merge ?
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        # output -> xi
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

































































