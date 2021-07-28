import sys

import cv2

sys.path.append("../../../../common")
sys.path.append("../")
import os
import numpy as np
import time
import atlas_utils.constants as const
from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource
from PIL import Image

labels = ["person",
          "bicycle", "car", "motorbike", "aeroplane",
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

MODEL_PATH = "./sim.om"
image_dir = ""
MODEL_WIDTH = 640
MODEL_HEIGHT = 640


def preprocess(img_path):
    image = Image.open(img_path)
    img_h = image.size[1]
    img_w = image.size[0]
    net_h = MODEL_HEIGHT
    net_w = MODEL_WIDTH

    scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    shift_x = (net_w - new_w) // 2
    shift_y = (net_h - new_h) // 2
    shift_x_ratio = (net_w - new_w) / 2.0 / net_w
    shift_y_ratio = (net_h - new_h) / 2.0 / net_h

    image_ = image.resize((new_w, new_h))
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
    new_image = new_image.astype(np.float32)
    new_image = new_image / 255
    new_image = new_image.transpose(2, 0, 1).copy()
    print('new_image.shape', new_image.shape)
    return new_image, image

# time
def time_synchronized():
    return time.time()


def main():

    acl_resource = AclResource()
    acl_resource.init()
    model = Model(MODEL_PATH)

    images_list = [os.path.join(image_dir, img)
                   for img in os.listdir(image_dir)
                   if os.path.splitext(img)[1] in const.IMG_EXT]

    for pic in images_list:
        t0, t1 = 0., 0.

        # preprocess
        data, orig = preprocess(pic)

        # Send into model inference
        print('init infer')
        t = time_synchronized()
        result_list = model.execute([data, ])
        t0 += time_synchronized() - t
        print('end infer')

        print('print speed')
        print('%f' % t0)


if __name__ == '__main__':
    main()
