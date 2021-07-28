#!/usr/bin/env python
# encoding: utf-8
import sys
import os

import sys
import os
import numpy as np
import cv2
from PIL import Image

path = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import acl
import base64
from PIL import Image, ImageDraw, ImageFont
from atlas_utils.acl_dvpp import Dvpp
import atlas_utils.constants as const
from atlas_utils.acl_model import Model
from atlas_utils.acl_image import AclImage
from atlas_utils.acl_resource import AclResource
import time
from tqdm import tqdm

# para init
SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
MODEL_PATH = os.path.join(SRC_PATH, "./weights/mask.om")

image_dir = './datasets/Mask/images/val'
label_dir = image_dir.replace('images', 'labels')
IMG_EXT = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']

MODEL_WIDTH = 416
MODEL_HEIGHT = 416
NMS_THRESHOLD_CONST = 0.65  # nms
CLASS_SCORE_CONST = 0.6  # clss
MODEL_OUTPUT_BOXNUM = 10647
labels = ["on_mask", "mask"]


# nms
def func_nms(boxes, nms_threshold):
        b_x = boxes[:, 0]
        b_y = boxes[:, 1]
        b_w = boxes[:, 2]
        b_h = boxes[:, 3]
        scores = boxes[:, 5]
        areas = (b_w + 1) * (b_h + 1)

        order = scores.argsort()[::-1]
        keep = []  # keep box
        while order.size > 0:
            i = order[0]
            keep.append(i)  # keep max score
            # inter area  : left_top   right_bottom
            xx1 = np.maximum(b_x[i], b_x[order[1:]])
            yy1 = np.maximum(b_y[i], b_y[order[1:]])
            xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
            yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])
            # inter area
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # union area : area1 + area2 - inter
            union = areas[i] + areas[order[1:]] - inter
            # calc IoU
            IoU = inter / union
            inds = np.where(IoU <= nms_threshold)[0]
            order = order[inds + 1]

        final_boxes = [boxes[i] for i in keep]
        return final_boxes


# letterbox
def letterbox(img, new_shape=(416, 416), color=(114, 114, 114)):

    shape = img.shape[:2]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def main():
    # init
    acl_resource = AclResource()
    acl_resource.init()
    np.set_printoptions(threshold=np.inf)
    t0, t1 = 0., 0.

    # model load
    model = Model(MODEL_PATH)

    # images
    images_list = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if os.path.splitext(img)[1] in IMG_EXT]
    labels_list = [os.path.join(label_dir, lab) for lab in os.listdir(label_dir) if os.path.splitext(lab)[1] is '.txt']

    # map
    stats = []

    for img_index, image_file in enumerate(tqdm(images_list)):

        # 1.img
        bgr_img = cv2.imread(image_file).astype(np.float32)
        orig_shape = bgr_img.shape[:2]
        # 2.letterbox
        # resized_img, ratio, pad = letterbox(bgr_img, (MODEL_WIDTH, MODEL_HEIGHT))
        resized_img = cv2.resize(bgr_img, (MODEL_WIDTH, MODEL_HEIGHT)).astype(np.float32)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img / 255.0
        # 3.convert
        resized_img = np.transpose(rgb_img, axes=(2, 0, 1))
        resized_img = np.ascontiguousarray(resized_img)
        resized_img = resized_img[np.newaxis, :].astype(np.float32)

        # 模型推理
        t = time.time()
        infer_output = model.execute(resized_img)  # (1, 3, 416, 616)
        infer_output = infer_output[0]
        t0 += time.time() - t

        # 1.根据模型的输出以及对检测网络的认知，可以知道：-------------------------------------------------
        result_box = infer_output[:, :, 0:6].reshape((-1, 6)).astype('float32')
        list_class = infer_output[:, :, 5:7].reshape((-1, 2)).astype('float32')
        list_max = list_class.argmax(axis=1)
        list_max = list_max.reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 5] = list_max[:, 0]

        # 2.整合
        boxes = np.zeros(shape=(MODEL_OUTPUT_BOXNUM, 6), dtype=np.float32)  # 创建一个
        boxes[:, :4] = result_box[:, :4]
        boxes[:, 4] = result_box[:, 5]
        boxes[:, 5] = result_box[:, 4]
        all_boxes = boxes[boxes[:, 5] >= CLASS_SCORE_CONST]

        # 3.nms
        t = time.time()
        real_box = func_nms(np.array(all_boxes), NMS_THRESHOLD_CONST)
        t1 += time.time() - t

        # 4.通过网络输出计算出预测框在原图像上位置，使用PIL将检测的结果标注在原始图像上,画框并标记物体类别
        x_scale = orig_shape[1] / MODEL_HEIGHT
        y_scale = orig_shape[0] / MODEL_WIDTH

        for detect_result in real_box:
            top_x = int((detect_result[0] - detect_result[2] / 2) * x_scale)
            top_y = int((detect_result[1] - detect_result[3] / 2) * y_scale)
            bottom_x = int((detect_result[0] + detect_result[2] / 2) * x_scale)
            bottom_y = int((detect_result[1] + detect_result[3] / 2) * y_scale)
            cv2.rectangle(bgr_img, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1)
            cv2.putText(bgr_img, labels[int(detect_result[4])], (top_x, top_y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 5.保存
        if not os.path.isdir('./outputs'):
            os.mkdir('./outputs')
        output_path = os.path.join("./outputs", "out_" + os.path.basename(images_list[img_index]))
        cv2.imwrite(output_path, bgr_img)

    # print speed
    print("speed: %f %f" % (t0 / len(images_list), t1 / len(images_list)))

if __name__ == '__main__':
    main()