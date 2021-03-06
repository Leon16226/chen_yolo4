#!/home/dai/anaconda3/bin/python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile
import numpy as np


# classes = ['Bag', 'Cup', 'Bottle']
classes = ['SLD', 'ZBZ', 'KQSP']
Annota_Save_Root = '../../datasets/Material/labels'   # label save
Image_Save_Root = '../../datasets/Material/images'
Label_Root = '../../datasets/Material/data/labels'  # label name
Image_Root = '../../datasets/Material/data/images'


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


def convert_label(id=0, filepath='', sub_path=''):
    in_file = open(filepath)
    out_file = open(Annota_Save_Root + sub_path + '/m%s.txt' % id, 'w')
    if_none = True

    # xml parse
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    # iter
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls.upper() not in classes:
            continue
        cls_id = classes.index(cls.upper())
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        if_none = False if len(bb) > 0 else if_none
        # write---------------------------------------------------------------------------------------------------------
        temp = 0
        for b in bb:
            if b > 1:
                temp = 1
        if temp == 0 and not if_none:
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


    in_file.close()
    out_file.close()


def convert_image(id=0, filepath='', sub_path=''):
    copyfile(filepath, Image_Save_Root + sub_path + '/m%s.jpg' % id)


if __name__ == "__main__":
    nt = 0
    nv = 0
    for id, filename in enumerate(os.listdir(Label_Root)):
        rand = random.randint(1, 10)
        name = filename.split('.')[0]
        # --------------------------------------------------------------------------------------------------------------
        imagepath = os.path.join(Image_Root, name + '.jpg')
        if not os.path.exists(imagepath):
            imagepath = os.path.join(Image_Root, name + '.JPG')
        filepath = os.path.join(Label_Root, name + '.xml')
        # --------------------------------------------------------------------------------------------------------------
        if os.path.exists(imagepath):
            if rand <= 8:
                nt += 1
                convert_label(nt, filepath, '/train')
                convert_image(nt, imagepath, '/train')
            else:
                nv += 1
                convert_label(nv, filepath, '/val')
                convert_image(nv, imagepath, '/val')


