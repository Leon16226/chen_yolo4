#!/home/dai/anaconda3/bin/python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile
import numpy as np

# 根据自己的数据标签修改
classes = [ 'no_mask', 'with_mask' ]


Annota_Save_Root = './datasets/Mask/labels'

Annota_Get_Root = './datasets/Mask/mask/annotations'

Train_Name_Root = './datasets/Mask/mask/train.txt'



def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 输入id转变annotation，并且存储
def convert_annotation(image_id, sub_path='/train'):
    in_file = open(Annota_Get_Root + '/maksssksksss' + '%s.xml' % image_id)
    out_file = open(Annota_Save_Root + sub_path + '/maksssksksss%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()



if __name__ == "__main__":
    with open(Train_Name_Root, 'w') as f:
        for i in range(853):
            f.write(str(i) + '\n')
    f.close()

    with open(Train_Name_Root, 'r') as f:
        print("开始读取文件")
        one_item_array = {}
        lines = f.readlines()
        for line in lines:
          convert_annotation(line.strip(), '/train')
    f.close()


