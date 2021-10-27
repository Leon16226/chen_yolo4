import cv2
import numpy
import numpy as np
import abc
import threading
import time
from .utils_deepsort import iou
from .utils_deepsort import calc_iou
from .push import push
from .utils_deepsort import compute_color_for_id
from .utils_deepsort import plot_one_box




class Strategy(metaclass=abc.ABCMeta):
    def __init__(self, boxes, pool, opt, im0s, threshold, lock):
        self.boxes = boxes
        self.pool = pool
        self.opt = opt
        self.im0s = im0s
        self.pbox = []
        self.threshold = threshold
        self.lock = lock

        # names
        names = opt.name
        with open(names, 'r') as f:
            names = f.read().split('\n')
        self.labels = list(filter(None, names))

    @abc.abstractmethod
    def do(in_area_box):
        pass

    # 画标签
    def draw(self):
        # draw boxes for visualization----------------------------------------------------------------------
        for i, box in enumerate(self.pbox):
            bboxes = box[0:4]
            id = box[4]
            cls = box[5]
            c = int(cls)
            # conf = box[6]

            # label = f'{id} {self.labels[c]}{conf:.2f}'
            label = f'{id} {self.labels[c]}'
            color = compute_color_for_id(id)
            plot_one_box(bboxes, self.im0s, label=label, color=color, line_thickness=2)


# 0.异常停车---------------------------------------------------------------------------------------------------------------
# 1. 判断是否异常停车
# 2. 已post的id不重复上报
class CarStrategy(Strategy):

    def do(self,):
        for j, box in enumerate(self.boxes):
            # 初始化参数
            id = box[4]
            bboxes = box[0:4]

            # lock
            self.lock.acquire()
            states = 0
            for i, p in enumerate(self.pool[::-1]):
                if id == p[0] and p[1] < self.threshold:
                    o = iou(bboxes, p[2:6])
                    print("当前p时间：", p[6])
                    print("iou:", o)
                    print("thread id:", threading.currentThread().ident)
                    states = p[1] + 1 if o > 0.95 else p[1]
                    break
                elif id == p[0] and p[1] >= self.threshold:
                    states = self.threshold + 1
                    break

            print("id:", id)
            print("当前状态为：", states)

            self.pool.append([box[4], states, box[0], box[1], box[2], box[3], int(round(time.time() * 1000))])
            self.lock.release()

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()
                push(self.opt, self.im0s, "illegalPark")


# 1. 行人检测---------------------------------------------------------------------------------------------------------------
class PeopleStrategy(Strategy):

    def do(self,):
        # init
        cars = self.boxes[self.boxes[:, 5] == 0]
        peoples = self.boxes[self.boxes[:, 5] == 8]
        self.boxes = peoples

        if self.boxes.size == 0:
            return None

        # 加一个空车
        if cars.size == 0:
            cars = np.array([[0, 0, 10, 10, -1, -1]])

        # iou
        ious = calc_iou(peoples[:, 0:4], cars[:, 0:4])
        print("people ious:", ious)

        for j, box in enumerate(self.boxes):
            # 参数初始化
            id = box[4]

            # lock
            self.lock.acquire()
            states = 0
            quadrant = -1
            for i, p in enumerate(self.pool[::-1]):
                if id == p[0] and p[1] < self.threshold:

                    pious = ious[j]
                    index = np.argmax(pious)
                    car = cars[index]

                    # 行人格外策略----------------------------------------------------------------------------------------
                    # 1. 人和车重叠iou > 0
                    if pious[index] > 0:
                        o = np.array([(car[0] + car[2])/2, (car[1] + car[3])/2])
                        x = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
                        y = x - o

                        # quadrant
                        # (-1, -1)  (1, -1)
                        # (-1,  1)  (1,  1)
                        if y[0] < 0 and y[1] < 0:
                            quadrant = 0
                        elif y[0] < 0 and y[1] > 0:
                            quadrant = 1
                        elif y[0] > 0 and y[1] > 0:
                            quadrant = 2
                        elif y[0] > 0 and y[1] < 0:
                            quadrant = 3

                        if p[2] != quadrant and quadrant != -1:
                            states = p[1] + 1
                    else:
                        states = p[1] + 1
                    break
                    # 行人格外策略----------------------------------------------------------------------------------------
                elif id == p[0] and p[1] >= self.threshold:
                    states = self.threshold + 1
                    break

            self.pool.append([box[4], states, quadrant])
            self.lock.release()

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()
                push(self.opt, self.im0s, "peopleOrNoVehicles")


# 2. 抛洒物--------------------------------------------------------------------------------------------------------------
class MaterialStrategy(Strategy):
    def do(self):
        for j, box in enumerate(self.boxes):
            # 初始化参数
            id = box[4]

            # lock
            self.lock.acquire()
            states = 0
            for i, p in enumerate(self.pool[::-1]):
                if id == p[0] and p[1] < self.threshold:
                    print("抛撒物状态加1")
                    states = p[1] + 1
                    break
                elif id == p[0] and p[1] >= self.threshold:
                    states = self.threshold + 1
                    break


            self.pool.append([box[4], states])
            self.lock.release()

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()
                push(self.opt, self.im0s, "throwThings")


# 3. 应急车道异常行驶--------------------------------------------------------------------------------------------------------
class illegalDriving(Strategy):

    def do(self,):
        for j, box in enumerate(self.boxes):
            # 初始化参数
            id = box[4]
            color = compute_color_for_id(id)

            # lock
            self.lock.acquire()
            states = 0
            points = ''
            for i, p in enumerate(self.pool[::-1]):
                if id == p[0] and p[1] < self.threshold:
                    states = p[1] + 1
                    points = p[2] + str(int((box[0] + box[2])/2)) + ',' + str(int((box[1] + box[3])/2)) + ','
                    break
                elif id == p[0] and p[1] >= self.threshold:
                    states = self.threshold + 1
                    break

            print("id:", id)
            print("当前状态：", states)
            self.pool.append([box[4], states, points])  # ponits会被程序释放掉
            self.lock.release()

            # post
            if states == self.threshold:
                self.pbox = [box]
                self.draw()

                # 画点
                points = points.split(',')
                mpoints = []
                for i, p in enumerate(points[0:-1]):
                    if i % 2 == 0:
                        mpoints.append((int(p), int(points[i + 1])))
                for point in mpoints:
                    print("画点")
                    cv2.circle(self.im0s, point, 5, color, -1)

                push(self.opt, self.im0s, "illegalDriving")



def todo(c_box, pool, opt, im0s, lock):

    thresholds = [10, 3, 3, 20]

    # 不同处理策略集合
    strategies = {
        # 0: CarStrategy(c_box[0], pool[0], opt, im0s, thresholds[0], lock) if c_box[0].size != 0 else 'no',
        # 1: PeopleStrategy(c_box[1], pool[1], opt, im0s, thresholds[1], lock) if c_box[1].size != 0 else 'no',
        # 2: MaterialStrategy(c_box[2], pool[2], opt, im0s, thresholds[2], lock) if c_box[2].size != 0 else 'no',
        3: illegalDriving(c_box[3], pool[3], opt, im0s, thresholds[3], lock) if c_box[3].size != 0 else 'no'
    }

    for k, v in strategies.items():
        if v != 'no':
            v.do()



