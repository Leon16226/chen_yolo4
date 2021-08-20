import os

import cv2
import numpy as np


# 全局阈值
def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    print("阈值：", ret)
    cv2.imshow("binary", binary)


# 局部阈值
def local_threshold(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    cv2.imshow("binary ", binary)


def custom_threshold(image):
    # gray--------------------------------------------------------------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])
    mean = m.sum()/(w*h)
    # binary------------------------------------------------------------------------------------------------------------
    ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    # erode-------------------------------------------------------------------------------------------------------------
    k = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, k, iterations=1)
    return binary


if __name__ == "__main__":
    # files-------------------------------------------------------------------------------------------------------------
    path = "./in"
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.exists(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
            img = custom_threshold(img)
            cv2.imwrite("./out/" + file, img)


