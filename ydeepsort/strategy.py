import numpy as np
import abc
from .utils_deepsort import iou
from .utils_deepsort import calc_iou
from .push import push
from .utils_deepsort import compute_color_for_id
from .utils_deepsort import plot_one_box



class Strategy(metaclass=abc.ABCMeta):
    def __init__(self, boxes, pool, opt, im0s):
        self.boxes = boxes
        self.pool = pool
        self.opt = opt
        self.im0s = im0s

        # names
        names = opt.name
        with open(names, 'r') as f:
            names = f.read().split('\n')
        self.labels = list(filter(None, names))


    @abc.abstractmethod
    def do(in_area_box):
        pass

    def draw(self):
        # draw boxes for visualization----------------------------------------------------------------------
        for i, box in enumerate(self.boxes):
            bboxes = box[0:4]
            id = box[4]
            cls = box[5]
            c = int(cls)
            # conf = box[6]

            # label = f'{id} {self.labels[c]}{conf:.2f}'
            label = f'{id} {self.labels[c]}'
            color = compute_color_for_id(id)
            plot_one_box(bboxes, self.im0s, label=label, color=color, line_thickness=2)


# Car Park--------------------------------------------------------------------------------------------------------------
class CarStrategy(Strategy):

    def do(self,):

        # if post depend on iou
        for j, box in enumerate(self.boxes):
            # init
            id = box[4]
            bboxes = box[0:4]

            static_state = 0
            for x, p in enumerate(self.pool):
                if(id == p[4]):
                    o = iou(bboxes, p[0:4])
                    static_state = static_state + 1 if o > 0.95 else static_state

            self.pool.append(box)

            # post
            if static_state == 3:
                self.draw()
                push(self.opt, self.im0s, "illegalPark")
                break


# People Detect---------------------------------------------------------------------------------------------------------
class PeopleStrategy(Strategy):

    def do(self,):
        # init----------------------------------------------------------------------------------------------------------
        boxes_people = self.boxes[self.boxes[5].astype('int') == 8]
        boxes_car = self.boxes[self.boxes[5].astype('int') == 0]

        # iou
        ious = calc_iou(boxes_car[:, 0:4], boxes_people[:, 0:4])
        b = ious > 0
        c = b.sum(axis=0)
        d = (c == 0)
        boxes_people = boxes_people[d]
        # angle

        # center_c = (boxes_car[:, 2:4] - boxes_car[:, 0:2]) / 2
        # center_p = (boxes_people[:, 2:4] - boxes_people[:, 0:2]) / 2
        #
        #
        # # if post depend on id
        # for j, (box, center) in enumerate(zip(boxes_people, center_p)):
        #     center_abs = np.abs(center_c - center)
        #     center_len = center_abs[:, 0] + center_abs[:, 1]
        #     len_max_index = np.argmax(center_len)
        #
        #     # angle
        #     tan = np.arctan(center_abs[len_max_index, 0] / center_abs[len_max_index, 1])
        #     angle = np.degrees(tan)
        #     car_id = boxes_car[len_max_index, 4]
        #
        #
        #     if box[4] in self.pool.keys():
        #         (p_angle, p_carid, p_post) = self.pool[box[4]]
        #         # not push yet
        #         if p_post == False:
        #             if((car_id == p_carid) and (np.abs(angle - p_angle) < 10)):
        #                 pass
        #             else:
        #                 print("people push:", box)
        #
        #                 self.draw()
        #                 push(self.opt, self.im0s, "peopleOrNoVehicles")
        #                 break
        #     else:
        #         self.pool[box[4]] = (angle, car_id, False)


        # if post depend on id------------------------------------------------------------------------------------------
        for j, box in enumerate(boxes_people):
            if box[4] not in self.pool:
                self.pool.append(box[4])
                print("people push:", box)

                self.draw()
                push(self.opt, self.im0s, "peopleOrNoVehicles")
                break


class MaterialStrategy(Strategy):
    def do(self):
        # do nothing
        pass




def todo(c_box, pool, opt, im0s):

    strategies = {
        0: CarStrategy(np.array(c_box[0]), pool[0], opt, im0s),
        1: PeopleStrategy(np.array(c_box[1]), pool[1], opt, im0s),
        2: MaterialStrategy(np.array(c_box[2]), pool[2], opt, im0s)
    }

    for v in strategies.values():
        v.do()


