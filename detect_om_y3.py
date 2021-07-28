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

# road
SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
MODEL_PATH = os.path.join(SRC_PATH, "./yolov3.om")
# images
image_dir = './data/images'
IMG_EXT = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']
# sizes
MODEL_WIDTH = 416
MODEL_HEIGHT = 416

# post_process
def post_process(infer_output, image_file):
    print("post process")
    #
    data = infer_output[0]
    vals = data.flatten()
    top_k = vals.argsort()[-1:-6:-1]
    #
    object_class = get_image_net_class(top_k[0])
    #
    output_path = os.path.join(os.path.join(SRC_PATH, "../outputs"), os.path.basename(image_file))
    origin_image = Image.open(image_file)
    draw = ImageDraw.Draw(origin_image)
    # write
    font = ImageFont.load_default()
    font.size = 50
    draw.text((10, 50), object_class, font=font, fill=255)
    # save
    origin_image.save(output_path)
    # object_class = get_image_net_class(top_k[0])
    return

# image info
def construct_image_info():
    """construct image info"""
    image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT,
                           MODEL_WIDTH, MODEL_HEIGHT],
                          dtype=np.float32)
    return image_info

# 使用非极大值抑制IoU来消除重叠较大的预测框
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

def main():
    # init
    acl_resource = AclResource()
    acl_resource.init()
    # model load
    model = Model(MODEL_PATH)
    dvpp = Dvpp(acl_resource)

    # images
    images_list = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if os.path.splitext(img)[1] in IMG_EXT]

    # image info
    image_info = construct_image_info()

    for image_file in images_list:
        print(image_file)
        Image.open(image_file)

        # 1.使用OpenCV的imread接口读取图片，读取出来的是BGR格式
        bgr_img = cv2.imread(image_file).astype(np.float32)
        print('bgr_img.shape=' + str(bgr_img.shape))
        orig_shape = bgr_img.shape[:2]
        print('orig_shape=' + str(orig_shape))

        # 2.色域格式转换 （BGR -> RGB)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        print('Image shape after color gamut conversion:' + str(rgb_img.shape))

        # 3.对图片进行归一化处理
        rgb_img = rgb_img / 255.0
        print('rgb_img.shape after normalization : ' + str(bgr_img.shape))

        # 4.将RGB图像缩放到模型输入要求宽高比例
        resized_img = cv2.resize(rgb_img, (MODEL_WIDTH, MODEL_HEIGHT)).astype(np.float32)
        print('Image shape after resize:' + str(resized_img.shape))
        print('Image size(byte) after resize : ' + str(resized_img.nbytes))

        # 模型推理
        print('init infer')
        infer_output = model.execute([resized_img,])
        print(infer_output)
        print('end infer')

        # 解析结果
        NMS_THRESHOLD_CONST = 0.5
        CLASS_SCORE_CONST = 0.4
        #  box num
        MODEL_OUTPUT_BOXNUM = 10647
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane",
              "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
              "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
              "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
              "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table",
              "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]

        # 1.根据模型的输出以及对检测网络的认知，可以知道：
        # 模型的第一路输出infer_output[0]表示的是所有检测框的类别及对应置信度信息；
        # 模型的第二路输出infer_output[1]表示的是所有检测框的中心点坐标x,y和检测框宽高w,h；
        result_class = infer_output[0].reshape(MODEL_OUTPUT_BOXNUM, 80).astype('float32')  # 每一个检测框的类别及对应的相似度得分
        result_box = infer_output[1].reshape(MODEL_OUTPUT_BOXNUM, 4).astype('float32')  # 每一个检测框对应的坐标

        # 2.整合两路输出信息并筛选出所有置信度大于CLASS_SCORE_CONST的检测框all_boxes
        boxes = np.zeros(shape=(MODEL_OUTPUT_BOXNUM, 6), dtype=np.float32)  # 创建一个
        boxes[:, :4] = result_box
        list_score = result_class.max(axis=1)
        list_class = result_class.argmax(axis=1)
        list_score = list_score.reshape(MODEL_OUTPUT_BOXNUM, 1)
        list_class = list_class.reshape(MODEL_OUTPUT_BOXNUM, 1)
        boxes[:, 4] = list_class[:, 0]
        boxes[:, 5] = list_score[:, 0]
        all_boxes = boxes[boxes[:, 5] >= CLASS_SCORE_CONST]


        # 3.nms
        real_box = func_nms(np.array(all_boxes), NMS_THRESHOLD_CONST)

        # 4.通过网络输出计算出预测框在原图像上位置，使用PIL将检测的结果标注在原始图像上,画框并标记物体类别
        x_scale = orig_shape[1] / MODEL_HEIGHT
        y_scale = orig_shape[0] / MODEL_WIDTH

        for detect_result in real_box:
            top_x = int((detect_result[0] - detect_result[2] / 2) * x_scale)
            top_y = int((detect_result[1] - detect_result[3] / 2) * y_scale)
            bottom_x = int((detect_result[0] + detect_result[2] / 2) * x_scale)
            bottom_y = int((detect_result[1] + detect_result[3] / 2) * y_scale)
            ret = cv2.rectangle(bgr_img, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1)
            ret = cv2.putText(bgr_img, labels[int(detect_result[4])], (top_x, top_y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 5.保存
        if not os.path.isdir('./outputs'):
            os.mkdir('./outputs')
        output_path = os.path.join("./outputs", "out_" + os.path.basename(images_list[0]))
        cv2.imwrite(output_path, bgr_img)
        Image.open(output_path)

if __name__ == '__main__':
    main()