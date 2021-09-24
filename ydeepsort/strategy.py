import numpy as np
import abc
from ydeepsort.utils_deepsort import iou
from ydeepsort.push import push


class Strategy(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def do(in_area_box):
        pass

# Car Park--------------------------------------------------------------------------------------------------------------
class CarStrategy(Strategy):

    def do(self, boxes, pool, opt, im0s):

        # if post depend on iou
        for j, box in enumerate(boxes):
            # init
            id = box[4]
            bboxes = box[0:4]

            static_state = 0
            for x, p in enumerate(pool):
                if(id == p[4]):
                    o = iou(bboxes, p[0:4])
                    static_state = static_state + 1 if o > 0.9 else static_state

            pool.append(box)

            # post
            if static_state >= 2:
                push(opt, im0s, "illegalPark")
                break










class PeopleStrategy(Strategy):
    def do(in_area_box):
        # do nothing
        pass

class MaterialStrategy(Strategy):
    def do(in_area_box):
        # do nothing
        pass




def todo(c_box, pool, opt, im0s):

    strategies = {
        0: CarStrategy(),
        1: PeopleStrategy(),
        2: MaterialStrategy()
    }

    for k, v in strategies.items():
        if len(c_box[k]) != 0:
            v.do(np.array(c_box[k]), pool[k], opt, im0s)


