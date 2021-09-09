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
from shapely.geometry import Polygon


out = "./inference"
SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]

MODEL_WIDTH = 608
MODEL_HEIGHT = 608
NMS_THRESHOLD_CONST = 0.65
CLASS_SCORE_CONST = 0.6
MODEL_OUTPUT_BOXNUM = 10647

vfps = 0

# Tools-----------------------------------------------------------------------------------------------------------------
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

        thread_fps = Thread(target=showfps, args=(), daemon=True)
        thread_fps.start()

        print('')  # newline

    def update(self, cap):
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            # fps = 25--------------------------------------------------------------------------------------------------
            if n == 1:
                _, self.imgs = cap.retrieve()
                n = 0
            # time.sleep(0.01)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        print("get a img-----------------:", img0.shape)

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

# shapley
def Cal_area_2poly(point1,point2):

    poly1 = Polygon(point1).convex_hull      # Polygon：多边形对象
    poly2 = Polygon(point2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area

# fps
def showfps():
    print("rtsp success")
    global vfps
    while(True):
        time.sleep(1.0)
        print("fps:", vfps)
        vfps = 0

# Detect----------------------------------------------------------------------------------------------------------------
def detect(opt):

    # opt---------------------------------------------------------------------------------------------------------------
    rtsp = opt.rtsp
    MODEL_PATH = os.path.join(SRC_PATH, opt.om)
    print('rtsp:', rtsp)
    print("om:", MODEL_PATH)

    # Load labels-------------------------------------------------------------------------------------------------------
    names = opt.name
    labels = load_classes(names)
    print('labels:', labels)
    assert len(labels) > 0, "label file load fail"

    # init
    t0, t1 = 0., 0.
    point2 = [[] for i in labels]
    nf = 0
    threshold = 1
    threshold_box = 30
    global vfps

    # dir
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # Load model--------------------------------------------------------------------------------------------------------
    acl_resource = AclResource()
    acl_resource.init()
    model = Model(MODEL_PATH)

    # Load dataset----------------------------------------------------------------------------------------------------
    dataset = LoadStreams(rtsp, img_size=(MODEL_WIDTH, MODEL_HEIGHT))

    # do
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):

        orig_shape = im0s.shape[:2]
        resized_img = img.astype(np.float32)
        resized_img /= 255.0

        # 模型推理-------------------------------------------------------------------------------------------------------
        t = time.time()
        infer_output = model.execute(resized_img)
        assert infer_output[0].shape[1] > 0, "model no output, please check"
        infer_output_size = len(infer_output)
        if(infer_output_size == 1):
            infer_output = infer_output[0]
        elif(infer_output_size == 2):
            infer_output_0 = infer_output[0]
            infer_output_1 = infer_output[1].reshape((1, -1, 4))
            infer_output_2 = np.ones([1, infer_output_1.shape[1], 1])
            infer_output = np.concatenate((infer_output_1, infer_output_2, infer_output_0), axis=2)
        print(infer_output.shape)
        t0 += time.time() - t

        # 1.根据模型的输出以及对检测网络的认知，可以知道：---------------------------------------------------------------------
        nc = len(labels)
        MODEL_OUTPUT_BOXNUM = infer_output.shape[1]
        result_box = infer_output[:, :, 0:6].reshape((-1, 6)).astype('float32')
        list_class = infer_output[:, :, 5:5+nc].reshape((-1, nc)).astype('float32')
        list_max = list_class.argmax(axis=1)
        list_max = list_max.reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 5] = list_max[:, 0]
        if(infer_output_size == 2):
            list_max = list_class.max(axis=1)
            list_max = list_max.reshape((MODEL_OUTPUT_BOXNUM, 1))
            result_box[:, 4] = list_max[:, 0]

        # 2.整合
        boxes = np.zeros(shape=(MODEL_OUTPUT_BOXNUM, 6), dtype=np.float32)  # 创建一个
        boxes[:, :4] = result_box[:, :4]
        boxes[:, 4] = result_box[:, 5]
        boxes[:, 5] = result_box[:, 4]
        all_boxes = boxes[boxes[:, 5] >= CLASS_SCORE_CONST]
        # only people---------------------------------------------------------------------------------------------------
        if(infer_output_size == 2):
            all_boxes = all_boxes[all_boxes[:, 4] == (8 or 2 or 3 or 4 or 5 or 6 or 7)]

        # 3.nms
        t = time.time()
        real_box = func_nms(np.array(all_boxes), NMS_THRESHOLD_CONST)
        t1 += time.time() - t

        # 4.scale
        x_scale = orig_shape[1] / MODEL_HEIGHT
        y_scale = orig_shape[0] / MODEL_WIDTH

        # filter strategy-----------------------------------------------------------------------------------------------

        # 1. in detect area
        opt_point1 = opt.area
        opt_point1 = opt_point1.split(',')
        toplx, toply = int(opt_point1[0]), int(opt_point1[1])
        toprx, topry = int(opt_point1[2]), int(opt_point1[3])
        bottomlx, bottomly = int(opt_point1[4]), int(opt_point1[5])
        bottomrx, bottomry = int(opt_point1[6]), int(opt_point1[7])
        point1 = [toplx, toply, bottomlx, bottomly, bottomrx, bottomry, toprx, topry]
        point1 = np.array(point1).reshape(4, 2)

        # 2.remove duplicate

        # real_box------------------------------------------------------------------------------------------------------
        coords = []
        coords_in_area = []
        for detect_result in real_box:
            # top_x = int((detect_result[0] - detect_result[2] / 2) * x_scale)
            # top_y = int((detect_result[1] - detect_result[3] / 2) * y_scale)
            # bottom_x = int((detect_result[0] + detect_result[2] / 2) * x_scale)
            # bottom_y = int((detect_result[1] + detect_result[3] / 2) * y_scale)

            top_x = int((detect_result[0]) * 608 * x_scale)
            top_y = int((detect_result[1]) * 608 * y_scale)
            bottom_x = int((detect_result[2]) * 608 * x_scale)
            bottom_y = int((detect_result[3]) * 608 * y_scale)

            # plan1-----------------------------------------------------------------------------------------------------
            point = [top_x, top_y, top_x, bottom_y, bottom_x, bottom_y, bottom_x, top_y]
            point = np.array(point).reshape(4, 2)
            inter_area = Cal_area_2poly(point1, point)
            pred_area = (bottom_x - top_x) * (bottom_y - top_y)
            iou_p1 = inter_area / pred_area if pred_area != 0 else 0

            # plan2-----------------------------------------------------------------------------------------------------
            if(iou_p1 >= 0.5):
                niou = 0
                for i, p in enumerate(point2[int(detect_result[4])]):
                    p = np.array(p).reshape(4, 2)
                    inter_area = Cal_area_2poly(point, p)
                    p_iou = inter_area / pred_area
                    print("iou", p_iou)
                    niou = niou + 1 if p_iou >= 0.9 else niou
                print("niou:", niou)
                if(niou < threshold):
                    print(point2[int(detect_result[4])])
                    print((top_x, top_y, bottom_x - top_x, bottom_y - top_y, detect_result[4], detect_result[5]))

            # cv2
            nf_thres = 0
            nf_thres = nf_thres + nf if nf_thres < 120 else nf_thres
            threshold = 1 if nf_thres < 120 and threshold == 1 else 2
            if(iou_p1 >= 0.5 and niou < 5):
                coords_in_area.append((top_x, top_y, bottom_x - top_x, bottom_y - top_y,
                                       detect_result[4], detect_result[5]))
            if(iou_p1 >= 0.5 and niou < threshold):
                cv2.rectangle(im0s, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 3)
                cv2.putText(im0s, labels[int(detect_result[4])] + " " + str(detect_result[5]), (top_x, top_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                coords.append((top_x, top_y, bottom_x - top_x, bottom_y - top_y, detect_result[4], detect_result[5]))

        # before push---------------------------------------------------------------------------------------------------
        if len(coords_in_area) > 0:
            # del-------------------------------
            for i, p2 in enumerate(point2):
                if(len(p2) >= threshold_box):
                    del p2[0:-threshold_box]
            # in---------------------------------
            for i, cor in enumerate(coords_in_area):
                point2[int(cor[4])].append([cor[0], cor[1], cor[0], cor[1] + cor[3],
                               cor[0] + cor[2], cor[1] + cor[3], cor[0] + cor[2], cor[1]])
            # in--------------------------------
            if(len(coords) > 0):
                for i, cor in enumerate(coords):
                    for j in np.arange(0, 2):
                        point2[int(cor[4])].append([cor[0], cor[1], cor[0], cor[1] + cor[3],
                                    cor[0] + cor[2], cor[1] + cor[3], cor[0] + cor[2], cor[1]])

        # push----------------------------------------------------------------------------------------------------------
        if(len(coords) > 0):
            print("push one")
            push(opt, im0s, coords)

        # wait key------------------------------------------------------------------------------------------------------

        # show detect area
        if(opt.show):
            point1 = point1.reshape((-1, 1, 2))
            cv2.polylines(im0s, [point1], True, (0, 255, 255))

            cv2.imshow("material", im0s)
            k = cv2.waitKey(10) & 0xFF
            if k == ord('q'):
                print('quit')
                cv2.destroyAllWindows()
                break

        # fps-----------------------------------------------------------------------------------------------------------
        vfps += 1


# Event  Post-----------------------------------------------------------------------------------------------------------
class Event(object):
    def __init__(self, cameraIp, timestamp,
                 roadId, roadName, code, subCode, dateTime, status, no, distance, picture,
                 targetType, xAxis, yAxis, height, width, prob,
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
            "coordinate": [
                {
                    "targetType": targetType,
                    "xAxis": xAxis,
                    "yAxis": yAxis,
                    "height": height,
                    "width": width,
                    "prob": prob
                }
            ],
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


def push(opt, frame, coords):
    # opt
    post_url = opt.post
    ponit_ip = opt.point

    # event ------------------------------------------------------------------------------------------------------------
    _, bi_frame = cv2.imencode('.jpg', frame)
    img = base64.b64encode(bi_frame)
    img = str(img)
    img = img[2:]

    coordinate = []
    for i, coord in enumerate(coords):
        coordinate.append(Coordinate("material", coord[0], coord[1], coord[2], coord[3], coord[4]))

    event = Event(ponit_ip, int(round(time.time() * 1000)),
                  1, "yzw1-dxcd", "peopleOrNoVehicles" if opt.om == "weights/highway-sim.om" else "throwThings", "", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 1, [1], 30, img,
                  "material", 0, 0, 0, 0, 0.75,
                  "", "",
                  "")
    event = json.dumps(event, default=lambda obj: obj.__dict__, sort_keys=True, indent=4)

    # post -------------------------------------------------------------------------------------------------------------
    url = post_url
    headers = {"content-type": "application/json"}
    ret = requests.post(url, data=event, headers=headers)
    print(ret.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 0,0,1920,0,0,1080,1920,1080

    # company
    # 752,555,1342,608,392,928,1304,1066
    # rtsp://admin:xsy12345@192.168.1.89:554/cam/realmonitor?channel=1&subtype=0
    # http://192.168.1.19:8080/v1/app/interface/uploadEvent
    # 10.17.1.20

    # xs
    # 885,480,1135,495,830,1059,1719,1065
    # rtsp://admin:xsy12345@33.64.78.178/h265/ch1/main/av_stream
    # http://33.64.78.180:8080/v1/app/interface/uploadEvent
    # 33.64.78.178
    # weights/highway-sim.om
    # data/highway.names
    parser.add_argument('--area', type=str, default='870,480,1135,495,830,1059,1719,1065', help='lt rt lb rb')
    parser.add_argument('--rtsp', type=str, default='test.mp4')
    parser.add_argument('--post', type=str, default='http://192.168.1.19:8080/v1/app/interface/uploadEvent')
    parser.add_argument('--point', type=str, default='10.17.1.20')
    parser.add_argument('--om', type=str, default='weights/deepsort_mars.om')
    parser.add_argument('--name', type=str, default='data/highway.names')
    parser.add_argument('--show', action='store_true')
    opt = parser.parse_args()

    # fps


    detect(opt)




