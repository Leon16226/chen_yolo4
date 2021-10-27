import argparse
import os
import shutil
import copy
import threading

import cv2
import numpy as np
from threading import Thread

from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource
from ydeepsort.utils import *
from ydeepsort.utils_deepsort import *
from ydeepsort.utils_deepsort import _preprocess
from ydeepsort.utils_deepsort import _xywh_to_xyxy
from ydeepsort.utils_deepsort import _xywh_to_tlwh
from ydeepsort.utils_deepsort import _tlwh_to_xyxy
from ydeepsort.utils_deepsort import filter_pool
from ydeepsort.push import *
from ydeepsort.sort.nn_matching import NearestNeighborDistanceMetric
from ydeepsort.sort.tracker import Tracker
from ydeepsort.sort.detection import Detection








y = readyaml()

out = "./inference"
SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]

MODEL_WIDTH = y['MODEL_WIDTH']
MODEL_HEIGHT = y['MODEL_HEIGHT']
NMS_THRESHOLD_CONST = y['NMS_THRESHOLD_CONST']
CLASS_SCORE_CONST = y['CLASS_SCORE_CONST']
MODEL_OUTPUT_BOXNUM = 10647
vfps = 0

# deepsort config
MAX_DIST = y['MAX_DIST']
MIN_CONFIDENCE = y['MIN_CONFIDENCE']
NMS_MAX_OVERLAP = y['NMS_MAX_OVERLAP']
MAX_IOU_DISTANCE = y['MAX_IOU_DISTANCE']
MAX_AGE = y['MAX_AGE']
N_INIT = y['N_INIT']
NN_BUDGET = y['NN_BUDGET']

# pool
id_thres = 20
car_id_pool = []
people_id_pool = []
material_id_pool = []
illdri_id_pool = []
lock = threading.Lock()


# fps
def showfps():
    print("rtsp success")
    global vfps
    while(True):
        time.sleep(1.0)
        print("fps:", vfps)
        vfps = 0

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
        thread = Thread(target=self.update, args=([cap,]), daemon=True)
        print('success (%gx%g at %.2f FPS).' % (w, h, fps))
        thread.start()

        thread_fps = Thread(target=showfps, args=(), daemon=True)
        thread_fps.start()

        print('')  # newline

    def update(self, cap):
        n = 0
        while cap.isOpened():
            n += 1
            ret = cap.grab()

            # 若没有帧返回，则重新刷新rtsp视频流
            while not ret:
                cap = cv2.VideoCapture(self.source)
                if not(cap):
                    continue
                ret = cap.grab()
                print("rtsp重新连接中---------------------------")
                time.sleep(0.5)

            # fps = 25
            if n == 2:
                _, self.imgs = cap.retrieve()
                n = 0


    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        print("get a img-----------------------------------------------------:", img0.shape)

        # resize
        img = cv2.resize(img0, self.img_size)
        img = img[np.newaxis, :]
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0

        return self.source, img, img0, self.cap

    def __len__(self):
        return 0


# Detect----------------------------------------------------------------------------------------------------------------
def detect(opt):
    # opt---------------------------------------------------------------------------------------------------------------
    rtsp = opt.rtsp
    MODEL_PATH = os.path.join(SRC_PATH, opt.om)
    MODEL_PATH_EX = os.path.join(SRC_PATH, opt.ex)
    print('rtsp:', rtsp)
    print("om:", MODEL_PATH)

    # Load labels-------------------------------------------------------------------------------------------------------
    names = opt.name
    labels = load_classes(names)
    print('labels:', labels)
    assert len(labels) > 0, "label file load fail"

    # init
    point2 = [[] for i in labels]
    threshold_box = 30
    global vfps
    nc = len(labels)

    # dir
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # pool 用于跟踪逻辑策略
    global car_id_pool
    global people_id_pool
    global material_id_pool

    # Load model--------------------------------------------------------------------------------------------------------
    acl_resource = AclResource()
    acl_resource.init()
    model = Model(MODEL_PATH)
    model_extractor = Model(MODEL_PATH_EX)

    # Load dataset------------------------------------------------------------------------------------------------------
    dataset = LoadStreams(rtsp, img_size=(MODEL_WIDTH, MODEL_HEIGHT))

    # init detect area--------------------------------------------------------------------------------------------------
    opt_point1 = opt.area
    opt_point1 = opt_point1.split(',')
    tlx1, tly1 = int(opt_point1[0]), int(opt_point1[1])
    trx1, try1 = int(opt_point1[2]), int(opt_point1[3])
    blx1, bly1 = int(opt_point1[4]), int(opt_point1[5])
    brx1, bry1 = int(opt_point1[6]), int(opt_point1[7])
    point1 = [tlx1, tly1, blx1, bly1, brx1, bry1, trx1, try1]
    point1 = np.array(point1).reshape(4, 2)

    # init road side----------------------------------------------------------------------------------------------------
    opt_point2 = opt.side
    opt_point2 = opt_point2.split(',')
    tlx2, tly2 = int(opt_point2[0]), int(opt_point2[1])
    trx2, try2 = int(opt_point2[2]), int(opt_point2[3])
    blx2, bly2 = int(opt_point2[4]), int(opt_point2[5])
    brx2, bry2 = int(opt_point2[6]), int(opt_point2[7])
    point2 = [tlx2, tly2, blx2, bly2, brx2, bry2, trx2, try2]
    point2 = np.array(point2).reshape(4, 2)

    # deepsort init-----------------------------------------------------------------------------------------------------
    max_cosine_distance = MAX_DIST
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=max_cosine_distance, max_age=MAX_AGE, n_init=N_INIT)

    limg = np.random.random([1, 3, 608, 608])
    # 开始取流检测--------------------------------------------------------------------------------------------------------
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):

        # 情况1：重复帧
        if np.sum(limg - img) == 0:
            print("xxxxxxxxxxxxxxxxx跳过这帧xxxxxxxxxxxxxxxx")
            continue
        limg = img


        # 模型推理-------------------------------------------------------------------------------------------------------
        infer_output = model.execute([img])
        assert infer_output[0].shape[1] > 0, "model no output, please check"
        infer_output_1 = infer_output[1].reshape((1, -1, 4))
        infer_output_2 = np.ones([1, infer_output_1.shape[1], 1])
        infer_output = np.concatenate((infer_output_1,
                                       infer_output_2,
                                       infer_output[0]), axis=2)

        # 模型输出box的数量
        MODEL_OUTPUT_BOXNUM = infer_output.shape[1]

        # 转换处理并根据置信度门限过滤box
        result_box = infer_output[:, :, 0:6].reshape((-1, 6)).astype('float32')
        list_class = infer_output[:, :, 5:5 + nc].reshape((-1, nc)).astype('float32')
        # class
        list_max = list_class.argmax(axis=1).reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 4] = list_max[:, 0]
        # conf
        list_max = list_class.max(axis=1).reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 5] = list_max[:, 0]
        all_boxes = result_box[result_box[:, 5] >= CLASS_SCORE_CONST]



        if all_boxes.shape[0] > 0:
            # 1.根据nms过滤box
            real_box = func_nms(all_boxes, NMS_THRESHOLD_CONST)
            print("real_box:", real_box.shape)

            # 2.scale
            orig_shape = im0s.shape[:2]  # (h, w, 3)
            x_scale = orig_shape[1] / MODEL_WIDTH
            y_scale = orig_shape[0] / MODEL_HEIGHT

            print('im0s:', im0s.shape)

            top_x = (real_box[:, 0] * MODEL_WIDTH * x_scale).astype(int)
            top_y = (real_box[:, 1] * MODEL_HEIGHT * y_scale).astype(int)
            bottom_x = (real_box[:, 2] * MODEL_WIDTH * x_scale).astype(int)
            bottom_y = (real_box[:, 3] * MODEL_HEIGHT * y_scale).astype(int)

            # 3.保留在检测区域内的box
            point = np.array([x for x in zip(top_x, top_y, top_x, bottom_y,
                                             bottom_x, bottom_y, bottom_x, top_y)]).reshape([-1, 4, 2])
            inter_area = np.array([Cal_area_2poly(point1, p) for p in point])
            det = real_box[inter_area > 5 * 5]  # gener [x1, y1, x2, y2, cls, confs]


            # 开始跟踪的处理-----------------------------------------------------------------------------------------------
            if det is not None and len(det):
                det[:, [0, 2]] = (det[:, [0, 2]] * MODEL_WIDTH * x_scale).round()
                det[:, [1, 3]] = (det[:, [1, 3]] * MODEL_HEIGHT * y_scale).round()
                print("det f:", det[:, :4].shape)

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 5]
                clss = det[:, 4]
                print("xywhs:", xywhs.shape)

                # 从原图im0s中截取目标区域，准备抽取特征---------------------------------------------------------------------
                height, width = orig_shape
                im_crops = []
                for i, box in enumerate(xywhs):
                    x1, y1, x2, y2 = _xywh_to_xyxy(box, height, width)
                    # ?
                    if x2 - x1 == 0:
                        x2 += 1
                    elif y2 - y1 == 0:
                        y2 += 1
                    im = im0s[y1:y2, x1:x2]
                    im_crops.append(im)

                # deepsort框架抽取特征------------------------------------------------------------------------------------
                if im_crops:
                    print("deepsort extractor")
                    im_batch = _preprocess(im_crops)
                    print("im_batch:", im_batch.shape)
                    # extractor-----------------------------------------------------------------------------------------
                    features = model_extractor.execute([im_batch, np.array(im_batch.shape)], 'deepsort')
                    print("features:", features[0].shape)
                    features = features[0][0:im_batch.shape[0], :]
                else:
                    features = np.array([])

                print("features:", np.array(features).shape)

                if features.shape[0] > 0:
                    # do track------------------------------------------------------------------------------------------
                    bbox_tlwh = _xywh_to_tlwh(xywhs)
                    detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs)]

                    # update tracker -----------------------------------------------------------------------------------
                    tracker.predict()
                    tracker.update(detections, clss)

                    # output bbox identities
                    outputs = []
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        box = track.to_tlwh()
                        x1, y1, x2, y2 = _tlwh_to_xyxy(box, height, width)
                        track_id = track.track_id
                        class_id = track.class_id
                        outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
                    # if len(outputs) > 0:
                    #     outputs = np.stack(outputs, axis=0)
                    #     confs = np.array(confs).reshape(-1, 1)
                    #     outputs = np.concatenate((outputs, confs), axis=1)


                    # # draw boxes for visualization----------------------------------------------------------------------
                    # if len(outputs) > 0:
                    #     for j, (output, conf) in enumerate(zip(outputs, confs)):
                    #         bboxes = output[0:4]
                    #         id = output[4]
                    #         cls = output[5]
                    #
                    #         c = int(cls)
                    #         label = f'{id} {labels[c]} {conf:.2f}'
                    #         color = compute_color_for_id(id)
                    #         plot_one_box(bboxes, im0s, label=label, color=color, line_thickness=2)


                    

                    # 新开一个线程取做处理---------------------------------------------------------------------------------
                    if len(outputs) > 0:
                        # 保持pool为一定大小否则内存溢出
                        car_id_pool = filter_pool(car_id_pool, id_thres)
                        people_id_pool = filter_pool(people_id_pool, id_thres)
                        material_id_pool = filter_pool(material_id_pool, id_thres)

                        # thread
                        thread_post = Thread(target=postprocess_track, args=(outputs,
                                                                             car_id_pool, people_id_pool,
                                                                             material_id_pool, illdri_id_pool,
                                                                             opt, im0s,
                                                                             lock,
                                                                             point2))
                        thread_post.start()
            else:
                tracker.increment_ages()

        # fps-----------------------------------------------------------------------------------------------------------
        vfps += 1

        # show----------------------------------------------------------------------------------------------------------
        if opt.show:
            point_s = point1.reshape((-1, 1, 2))
            cv2.polylines(im0s, [point_s], True, (0, 255, 255))

            cv2.namedWindow("deepsort", 0)
            cv2.resizeWindow("deepsort", 960, 540)
            cv2.imshow("deepsort", im0s)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                model.destroy()
                model_extractor.destroy()
                print('quit')
                break





if __name__ == '__main__':
    y = readyaml()

    parser = argparse.ArgumentParser()
    parser.add_argument('--area', type=str, default=y['AREA'], help='lt rt lb rb')
    parser.add_argument('--side', type=str, default=y['AREA_SIDE'], help='lt rt lb rb')
    parser.add_argument('--rtsp', type=str, default=y['RTSP'])
    parser.add_argument('--post', type=str, default=y['POST'])
    parser.add_argument('--point', type=str, default=y['POINT'])
    parser.add_argument('--om', type=str, default=y['OM'])
    parser.add_argument('--ex', type=str, default=y['EX'])
    parser.add_argument('--name', type=str, default=y['NAME'])
    parser.add_argument('--show', action='store_true')
    opt = parser.parse_args()

    detect(opt)




