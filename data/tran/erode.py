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
    binary = cv2.erode(binary, k, iterations=3)
    cv2.imshow("binary ", binary)


if __name__ == "__main__":
    img = cv2.imread("./m1.jpg")
    cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input image", img)
    # threshold---------------------------------------------------------------------------------------------------------
    custom_threshold(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
