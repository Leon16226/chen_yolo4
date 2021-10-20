import numpy
import numpy as np
import abc
import threading
from .utils_deepsort import iou
from .utils_deepsort import calc_iou
from .push import push
from .utils_deepsort import compute_color_for_id
from .utils_deepsort import plot_one_box

R = threading.Lock()


class Strategy(metaclass=abc.ABCMeta):
    def __init__(self, boxes, pool, opt, im0s):
        self.boxes = boxes
        self.pool = pool
        self.opt = opt
        self.im0s = im0s
        self.pbox = []

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

    # 策略：同一个id连续20帧(假设帧率为5fps，4s)iou>0.95则认为是异常停车
    def do(self,):
        for j, box in enumerate(self.boxes):
            # 初始化参数
            id = box[4]
            bboxes = box[0:4]

            # states
            states = 0
            for i, p in enumerate(self.pool[::-1]):
                if(id == p[0] and p[0] < 20):
                    o = iou(bboxes, p[0:4])
                    states = p[1] + 1 if o > 0.95 else p[1]
                    break

            # lock
            R.acquire()
            self.pool.append([box[4], states])
            R.release()

            # post
            if box[5][1] == 20:
                self.pbox = box
                self.draw()
                push(self.opt, self.im0s, "illegalPark")


# 1. 行人检测---------------------------------------------------------------------------------------------------------------
class PeopleStrategy(Strategy):

    def do(self,):
        # init
        cars = self.boxes[self.boxes[:, 5] == 0]
        peoples = self.boxes[self.boxes[:, 5] == 8]
        ious = calc_iou(peoples, cars)
        self.boxes = peoples

        for j, box in enumerate(self.boxes):
            # 参数初始化
            id = box[4]

            # states
            states = 0
            quadrant = -1
            for i, p in enumerate(self.pool[::-1]):
                if(id == p[0] and p[1] < 3):
                    pious = ious[i]
                    index = np.argmax(pious)
                    car = cars[index]

                    # 如果当前行人box和有1辆车重叠
                    if(pious[index] > 0):
                        o = ((car[0] + car[2])/2, (car[1] + car[3])/2)
                        x = ((box[0] + box[2])/2, (box[1] + box[3])/2)
                        y = x - o

                        # quadrant
                        # (-1, -1)  (1, -1)
                        # (-1,  1)  (1,  1)
                        if(y[0] < 0 and y[1] < 0):
                            quadrant = 0
                        elif(y[0] < 0 and y[1] > 0):
                            quadrant = 1
                        elif(y[0] > 0 and y[1] > 0):
                            quadrant = 2
                        elif(y[0] > 0 and y[1] < 0):
                            quadrant = 3

                        if(p[2] != quadrant):
                            states = p[1] + 1
                    else:
                        states = p[1] + 1
                    break

            # lock
            R.acquire()
            self.pool.append([box[4], states, quadrant])
            R.release()

            # post
            if states == 3:
                self.pbox = box
                self.draw()
                push(self.opt, self.im0s, "peopleOrNoVehicles")






# 2. 抛洒物--------------------------------------------------------------------------------------------------------------
class MaterialStrategy(Strategy):
    def do(self):
        for j, box in enumerate(self.boxes):
            # 初始化参数
            id = box[4]

            # states
            states = 0
            for i, p in enumerate(self.pool[::-1]):
                if(id == p[0] and p[1] < 3):
                    states = p[1] + 1
                    break

            # lock
            R.acquire()
            self.pool.append([box[4], states])
            R.release()

            # post
            if states == 3:
                self.pbox = box
                self.draw()
                push(self.opt, self.im0s, "throwThings")



# 3. 应急车道异常行驶--------------------------------------------------------------------------------------------------------
class illegalDriving(Strategy):

    def do(self,):
        pass



def todo(c_box, pool, opt, im0s):

    # 不同处理策略集合
    strategies = {
        0: CarStrategy(np.array(c_box[0]), pool[0], opt, im0s),
        1: PeopleStrategy(np.array(c_box[1]), pool[1], opt, im0s),
        2: MaterialStrategy(np.array(c_box[2]), pool[2], opt, im0s)
    }

    # for v in strategies.values():
    #    v.do()

    for k, v in strategies.items():
        if c_box[k].size != 0:
            v.do()



