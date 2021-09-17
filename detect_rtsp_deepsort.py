import argparse
import os
import shutil
import copy

import cv2
import numpy as np

from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource
from ydeepsort.utils import *
from ydeepsort.utils_deepsort import *
from ydeepsort.utils_deepsort import _preprocess
from ydeepsort.utils_deepsort import _xywh_to_xyxy
from ydeepsort.utils_deepsort import _xywh_to_tlwh
from ydeepsort.utils_deepsort import compute_color_for_id
from ydeepsort.utils_deepsort import plot_one_box
from ydeepsort.utils_deepsort import _tlwh_to_xyxy
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
MAX_DIST = 0.2
MIN_CONFIDENCE = 0.3
NMS_MAX_OVERLAP = 0.5
MAX_IOU_DISTANCE = 0.7
MAX_AGE = 70
N_INIT = 3
NN_BUDGET = 100

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
            time.sleep(0.01)

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


        return self.source, img, img0, self.cap

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


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

    # dir
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

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
    toplx, toply = int(opt_point1[0]), int(opt_point1[1])
    toprx, topry = int(opt_point1[2]), int(opt_point1[3])
    bottomlx, bottomly = int(opt_point1[4]), int(opt_point1[5])
    bottomrx, bottomry = int(opt_point1[6]), int(opt_point1[7])
    point1 = [toplx, toply, bottomlx, bottomly, bottomrx, bottomry, toprx, topry]
    point1 = np.array(point1).reshape(4, 2)

    # deepsort init-----------------------------------------------------------------------------------------------------
    max_cosine_distance = MAX_DIST
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=max_cosine_distance, max_age=MAX_AGE, n_init=N_INIT)


    # do
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):

        orig_shape = im0s.shape[:2]
        resized_img = img.astype(np.float32)
        resized_img /= 255.0

        # 模型推理-------------------------------------------------------------------------------------------------------
        infer_output = model.execute([resized_img])
        assert infer_output[0].shape[1] > 0, "model no output, please check"
        infer_output_0 = infer_output[0]
        infer_output_1 = infer_output[1].reshape((1, -1, 4))
        infer_output_2 = np.ones([1, infer_output_1.shape[1], 1])
        infer_output = np.concatenate((infer_output_1, infer_output_2, infer_output_0), axis=2)


        # postprocess---------------------------------------------------------------------------------------------------

        # thread_post = Thread(target=postprocess, args=(labels, copy.deepcopy(infer_output),
        #                                                CLASS_SCORE_CONST, NMS_THRESHOLD_CONST,
        #                                                orig_shape, MODEL_HEIGHT, MODEL_WIDTH,
        #                                                point2, copy.deepcopy(im0s), opt,
        #                                                threshold_box, point1, path,
        #                                                copy.deepcopy(img), MODEL_PATH_EX, model_extractor), )
        # thread_post.start()


        # Deepsort-----------------------------------------------------------------------------------------------------

        # init
        nc = len(labels)
        MODEL_OUTPUT_BOXNUM = infer_output.shape[1]

        # 1.process：-------------------------------------------------------------------------------------------------------
        result_box = infer_output[:, :, 0:6].reshape((-1, 6)).astype('float32')
        list_class = infer_output[:, :, 5:5 + nc].reshape((-1, nc)).astype('float32')
        list_max = list_class.argmax(axis=1)
        list_max = list_max.reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 5] = list_max[:, 0]
        # conf
        list_max = list_class.max(axis=1)
        list_max = list_max.reshape((MODEL_OUTPUT_BOXNUM, 1))
        result_box[:, 4] = list_max[:, 0]

        # 2.整合
        boxes = np.zeros(shape=(MODEL_OUTPUT_BOXNUM, 6), dtype=np.float32)
        boxes[:, :4] = result_box[:, :4]
        boxes[:, 4] = result_box[:, 5]
        boxes[:, 5] = result_box[:, 4]
        all_boxes = boxes[boxes[:, 5] >= CLASS_SCORE_CONST]


        # filter
        # only car---------------------------------------------------------------------------------------------------

        # all_boxes = all_boxes[(all_boxes[:, 4] == 0)]




        if all_boxes.shape[0] > 0:
            # 3.nms
            real_box = func_nms(np.array(all_boxes), NMS_THRESHOLD_CONST)
            print("real_box:", real_box.shape)

            # 4.scale
            x_scale = orig_shape[1] / MODEL_HEIGHT
            y_scale = orig_shape[0] / MODEL_WIDTH

            top_x = (real_box[:, 0] * 608 * x_scale).astype(int)
            top_y = (real_box[:, 1] * 608 * y_scale).astype(int)
            bottom_x = (real_box[:, 2] * 608 * x_scale).astype(int)
            bottom_y = (real_box[:, 3] * 608 * y_scale).astype(int)

            # if in detect area
            point = np.array([x for x in zip(top_x, top_y, top_x, bottom_y,
                                             bottom_x, bottom_y, bottom_x, top_y)]).reshape([-1, 4, 2])
            inter_area = np.array([Cal_area_2poly(point1, p) for p in point])
            in_area_box = real_box[inter_area > 5 * 5]


            # do deepsort-----------------------------------------------------------------------------------------------
            det = in_area_box  # gener [x1, y1, x2, y2, cls, confs]
            # im0 = im0s

            if det is not None and len(det):
                print("det:", det[:, :4])
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4] * 608, im0.shape).round()
                det[:, [0, 2]] = (det[:, [0, 2]] * 608 * x_scale).round()
                det[:, [1, 3]] = (det[:, [1, 3]] * 608 * y_scale).round()
                print("det f:", det[:, :4].shape)

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 5]
                clss = det[:, 4]
                print("xywhs:", xywhs.shape)

                # pass detections to deepsort---------------------------------------------------------------------------
                height, width = im0s.shape[:2]
                im_crops = []
                for box in xywhs:
                    x1, y1, x2, y2 = _xywh_to_xyxy(box, height, width)
                    im = im0s[y1:y2, x1:x2]
                    if (im.shape[0] != 0) and (im.shape[1] != 0):
                        print("im:", im.shape)
                        im_crops.append(im)

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


                if features.shape[0] > 0 :
                    # do track----------------------------------------------------------------------------------------------
                    bbox_tlwh = _xywh_to_tlwh(xywhs)
                    detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
                                    confs) if conf > MIN_CONFIDENCE]

                    # update tracker ---------------------------------------------------------------------------------------
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
                    if len(outputs) > 0:
                        outputs = np.stack(outputs, axis=0)


                    # draw boxes for visualization--------------------------------------------------------------------------
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)
                            label = f'{id} {labels[c]} {conf:.2f}'
                            color = compute_color_for_id(id)
                            plot_one_box(bboxes, im0s, label=label, color=color, line_thickness=2)
            else:
                tracker.increment_ages()


        # show
        if opt.show:
            point_s = point1.reshape((-1, 1, 2))
            cv2.polylines(im0s, [point_s], True, (0, 255, 255))

            cv2.imshow("deepsort", im0s)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                model.destroy()
                model_extractor.destroy()
                print('quit')
                break




        # fps-----------------------------------------------------------------------------------------------------------

        vfps += 1



if __name__ == '__main__':
    y = readyaml()

    parser = argparse.ArgumentParser()
    parser.add_argument('--area', type=str, default=y['AREA'], help='lt rt lb rb')
    parser.add_argument('--rtsp', type=str, default=y['RTSP'])
    parser.add_argument('--post', type=str, default=y['POST'])
    parser.add_argument('--point', type=str, default=y['POINT'])
    parser.add_argument('--om', type=str, default=y['OM'])
    parser.add_argument('--ex', type=str, default=y['EX'])
    parser.add_argument('--name', type=str, default=y['NAME'])
    parser.add_argument('--show', action='store_true')
    opt = parser.parse_args()

    detect(opt)




