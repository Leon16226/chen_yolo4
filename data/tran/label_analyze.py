#!/home/dai/anaconda3/bin/python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile
import numpy as np


classes = ['SLD', 'ZBZ', 'KQSP', 'ZHZ', 'PMX']
Label_Root = '../../datasets/Material/data/labels_road'
sesult = [0 for i in np.arange(0, len(classes))]
wh = [[0, 0, 0] for i in np.arange(0, len(classes))]

def convert_label(filepath=''):
    in_file = open(filepath)

    # xml parse
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    # iter--------------------------------------------------------------------------------------------------------------
    for obj in root.iter('object'):

        cls = obj.find('name').text
        if cls.upper() not in classes:
            continue
        cls_id = classes.index(cls.upper())
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))

        x = (b[0] + b[1]) / 2.0
        y = (b[2] + b[3]) / 2.0
        ws = b[1] - b[0]
        hs = b[3] - b[2]

        sesult[cls_id] += 1
        pixel = max(ws / w * 608, hs / h * 608)
        if(pixel <= 32):
            wh[cls_id][0] += 1
        elif(pixel <= 96):
            wh[cls_id][1] += 1
        else:
            wh[cls_id][2] += 1


    in_file.close()






if __name__ == "__main__":


    for id, filename in enumerate(os.listdir(Label_Root)):
        rand = random.randint(1, 10)
        name = filename.split('.')[0]
        filepath = os.path.join(Label_Root, name + '.xml')
        convert_label(filepath)

    print(sesult)
    print(wh)




