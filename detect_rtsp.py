import argparse
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from threading import Thread
from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource
import json
import requests
import base64
import time
import datetime

rtsp = "rtsp://admin:xsy12345@192.168.1.86:554/h264/ch1/main/av_stream"
post_url = "http://192.168.1.19:8080/v1/app/interface/uploadEvent"
ponit_ip = "10.17.1.20"
out = "./inference"
SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
MODEL_PATH = os.path.join(SRC_PATH, "./weights/mask.om")

MODEL_WIDTH = 608
MODEL_HEIGHT = 608
NMS_THRESHOLD_CONST = 0.65  # nms
CLASS_SCORE_CONST = 0.6  # clss
MODEL_OUTPUT_BOXNUM = 10647
labels = ["bag", "cup", "bottle"]


def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


# load rtsp
class LoadStreams:
    def __init__(self, source='', img_size=608):
        # init
        self.mode = 'images'
        self.img_size = img_size
        self.imgs = [None]
        self.source = source

        # Start
        cap = cv2.VideoCapture(source)
        self.cap = cap
        assert cap.isOpened(), 'Failed to open %s' % source
        # width & height & fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) % 100

        # read
        _, self.imgs = cap.read()

        # thread
        thread = Thread(target=self.update, args=([cap]), daemon=True)
        print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
        thread.start()

        print('')  # newline

    def update(self, cap):
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            # fps = 25--------------------------------------------------------------------------------------------------
            if n == 12:
                _, self.imgs = cap.retrieve()
                n = 0
            # time.sleep(0.01)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        print("get a img:", img0.shape)

        # Letterbox
        img = cv2.resize(img0, self.img_size)
        img = img[np.newaxis, :]

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)

        return self.source, img, img0, self.cap

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


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


def detect():

    # init time
    t0, t1 = 0., 0.

    # Initialize
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # Load model--------------------------------------------------------------------------------------------------------
    acl_resource = AclResource()
    acl_resource.init()
    model = Model(MODEL_PATH)

    # Set Dataloader----------------------------------------------------------------------------------------------------
    vid_path, vid_writer = None, None
    dataset = LoadStreams(rtsp, img_size=(MODEL_WIDTH, MODEL_HEIGHT))

    # iter
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):
        orig_shape = im0s.shape[:2]
        resized_img = img.astype(np.float32)
        resized_img /= 255.0

        # 模型推理
        t = time.time()
        infer_output = model.execute(resized_img)
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

        # 4.scale
        x_scale = orig_shape[1] / MODEL_HEIGHT
        y_scale = orig_shape[0] / MODEL_WIDTH

        # im0s -> h, w, n
        coords = []
        for detect_result in real_box:
            top_x = int((detect_result[0] - detect_result[2] / 2) * x_scale)
            top_y = int((detect_result[1] - detect_result[3] / 2) * y_scale)
            bottom_x = int((detect_result[0] + detect_result[2] / 2) * x_scale)
            bottom_y = int((detect_result[1] + detect_result[3] / 2) * y_scale)
            cv2.rectangle(im0s, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1)
            cv2.putText(im0s, labels[int(detect_result[4])], (top_x, top_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            coords.append((top_x, top_y, bottom_x - top_x, bottom_y - top_y, detect_result[5]))

        # push----------------------------------------------------------------------------------------------------------

        push(im0s, coords)



    vid_writer.release()

# Event  Post-----------------------------------------------------------------------------------------------------------
class Event(object):
    def __init__(self, cameraIp, timestamp,
                 roadId, roadName, code, subCode, dateTime, status, no, distance, picture,
                 coords,
                 miniPicture, carNo,
                 remark
                 ):
        self.cameraIp = cameraIp
        self.timestamp = timestamp
        self.events = [
         {
            "roadId": roadId,
            "roadName": roadName,
            "code": code,
            "subCode": subCode,
            "dateTime": dateTime,
            "status": status,
            "no": no,
            "distance": distance,
            "picture": picture,
            "coordinate": coords,
             "carNoAI": {
                 "miniPicture": miniPicture,
                 "carNo": carNo
             },
             "remark": remark
         }
        ]


class Coordinate(object):
    def __init__(self, targetType, xAxis, yAxis, height, width, prob):
        self.targetType = targetType
        self.xAxis = xAxis
        self.yAxis = yAxis
        self.height = height
        self.width = width
        self.prob = prob


def push(frame, coords):
    # event ------------------------------------------------------------------------------------------------------------
    img = base64.b64encode(frame.read())
    img = str(img)
    img = img[2:]

    coordinate = []
    for i, coord in enumerate(coords):
        coordinate.append(Coordinate("material", coord[0], coord[1], coord[2], coord[3], coord[4]))

    event = Event(ponit_ip, int(round(time.time() * 1000)),
                  1, "yzw1-dxcd", "throwThings", "", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 1, [1], 30, img,
                  coordinate,
                  "", "",
                  "")
    event = json.dumps(event, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)

    # post -------------------------------------------------------------------------------------------------------------
    url = post_url
    headers = {"content-type": "application/json"}
    ret = requests.post(url, data=event, headers=headers)
    print(ret.text)


if __name__ == '__main__':

    detect()
