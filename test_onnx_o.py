import os, sys

sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import time
import math



class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        # input or output
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores


def to_numpy(tensor):
    print(tensor.device)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


labels = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
r_model_path = "resnet50----.onnx"

# load
time_start1 = time.time()
rnet1 = ONNXModel(r_model_path)
time_end2 = time.time()
print('load model cost', time_end2 - time_start1)

# 测时间
for i in range(5):
    # reprocess
    time_start = time.time()
    img_ori = cv2.imread(r"C:\Users\59593\Desktop\pytorch_classification-master\samples\rolled-in_scale_155.jpg")
    img = cv2.resize(img_ori, (224, 224), interpolation=cv2.INTER_CUBIC)
    img_input = img[..., ::-1] # BGR to RGB
    img_input = (np.float32(img)/255.0-[0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_input = img_input.transpose((2, 0, 1))
    img_input = torch.from_numpy(img_input).unsqueeze(0)
    img_input = img_input.type(torch.FloatTensor)

    # infer
    out = rnet1.forward(to_numpy(img_input))

    print(out)
    print(labels[np.argmax(out[0][0])])

    time_end=time.time()
    print('infer cost',time_end-time_start)

    # cv2.putText(img_ori, labels[np.argmax(out[0][0])], (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)
    # cv2.imshow("1", img_ori)
    # cv2.waitKey(0)
