import argparse
import os
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from threading import Thread
from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource

rtsp = "rtsp://admin:xsy12345@192.168.1.86:554/h264/ch1/main/av_stream"
SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
MODEL_PATH = os.path.join(SRC_PATH, "./weights/mask.om")

MODEL_WIDTH = 608
MODEL_HEIGHT = 608
NMS_THRESHOLD_CONST = 0.65  # nms
CLASS_SCORE_CONST = 0.6  # clss
MODEL_OUTPUT_BOXNUM = 10647
labels = ["on_mask", "mask"]


def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))


# load rtsp
class LoadStreams:
    def __init__(self, source='', img_size=640):
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
            # fps = 25
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
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    # init time
    t0, t1 = 0., 0.

    # Initialize
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # Load model
    acl_resource = AclResource()
    acl_resource.init()
    model = Model(MODEL_PATH)

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadStreams(rtsp, img_size=(MODEL_WIDTH, MODEL_HEIGHT))

    # iter
    for i, (path, img, im0s, vid_cap) in enumerate(dataset):
        orig_shape = im0s.shape[:2]
        resized_img = img.astype(np.float32)
        resized_img /= 255.0

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

        # 4.scale
        x_scale = orig_shape[1] / MODEL_HEIGHT
        y_scale = orig_shape[0] / MODEL_WIDTH

        # im0s -> h, w, n
        for detect_result in real_box:
            top_x = int((detect_result[0] - detect_result[2] / 2) * x_scale)
            top_y = int((detect_result[1] - detect_result[3] / 2) * y_scale)
            bottom_x = int((detect_result[0] + detect_result[2] / 2) * x_scale)
            bottom_y = int((detect_result[1] + detect_result[3] / 2) * y_scale)
            cv2.rectangle(im0s, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1)
            cv2.putText(im0s, labels[int(detect_result[4])], (top_x, top_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        print('im0s', im0s.shape)

        # save
        save_path = str(Path(out) / 'result.mp4')
        fourcc = 'mp4v'
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(fourcc, fps, w, h)
        if i == 0:
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))  # 2560 * 1536
        # vid_writer.write(im0s)

        cv2.imshow('mask', im0s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')
            break

    vid_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--weights', type=str, default='./weights/mask.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/mask.names', help='*.cfg path')
    parser.add_argument('--conf-thres', type=float, default=0.65, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    # save
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folders
    # process
    parser.add_argument('--view-img', action='store_true', help='display results')  # view image
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt') # save-txt
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')  # filter
    parser.add_argument('--update', action='store_true', help='update all models')

    opt = parser.parse_args()
    print(opt)

    detect()
