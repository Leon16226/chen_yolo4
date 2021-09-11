import numpy as np
import abc


class Strategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def do(in_area_box):
        pass


class CarStrategy(Strategy):
    def do(in_area_box):
        # do nothing
        pass


class PeopleStrategy(Strategy):
    def do(in_area_box):
        # do nothing
        pass

class MaterialStrategy(Strategy):
    def do(in_area_box):
        # do nothing
        pass




def todo(c_box):
    strategies = {
        "car": CarStrategy,
        "people": PeopleStrategy,
        "material": MaterialStrategy
    }

    for k, v in strategies.items():
        if len(c_box[k]) != 0:
            v.do(np.array(c_box[k]))


