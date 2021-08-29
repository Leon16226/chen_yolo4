import torch
import random
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms.functional as torchtransform
import numpy as np
import torch.nn.functional as F
import math
import cv2
from utils.general import plot_images

# ada sampling----------------------------------------------------------------------------------------------------------
def fill_duck(data):
    try:
        img, annos, roadmap = data
        img = torch.tensor(img)
        roadmap = torch.tensor(roadmap)

        # constrain-----------------------------------------------------------------------------------------------------
        ch, cw= roadmap.shape
        sh = int(ch * 0.35)
        sw = int(cw * 0.3)
        roadmap[0:ch, 0:sw] = 0
        roadmap[0:ch, cw-sw:cw] = 0
        roadmap[0:sh, 0:cw] = 0
        # cv2.imwrite("/home/chen/Desktop/dd/" + str(annos[0][1]) + 'ss.jpg', roadmap.numpy())

        # I. Get valid area.--------------------------------------------------------------------------------------------
        valid_idx = roadmap.view(-1)
        idx = torch.nonzero(valid_idx).view(-1)
        if idx.size(0) == 0:
            return img, annos

        # valid xy
        xs = idx % roadmap.size(1)
        ys = idx // roadmap.size(1)
        coor = torch.stack((xs, ys), dim=1)

        # ann
        annos_n = [s for s in annos]

        # II Calculate scale -------------------------------------------------------------------------------------------
        scale_factor = [1.25, 0.75]
        (h, w, _) = img.shape
        for i, an in enumerate(annos):
            tx, ty = int((an[1] - an[3]/2) * w), int((an[2] - an[4]/2) * h)
            bx, by = int((an[1] + an[3]/2) * w), int((an[2] + an[4]/2) * h)
            img_an = img[ty:by, tx:bx, :]
            idxs = torch.randint(low=0, high=coor.shape[0], size=(2,))

            for j, scale in enumerate(scale_factor):
                (ah, aw, _) = img_an.shape
                rh, rw = int(ah*scale), int(aw*scale)
                img_scale = cv2.resize(img_an.numpy(), (rw, rh), interpolation=cv2.INTER_LINEAR)
                coorxy = coor[idxs[j]]
                rtx, rty = int(coorxy[0].numpy()), int(coorxy[1].numpy())
                rbx, rby = rtx + rw, rty + rh

                try:
                    img[rty:rby, rtx:rbx, :] = torch.from_numpy(img_scale)
                    annos_n.append([an[0], (rtx + rbx) / 2.0 / w, (rty + rby) / 2.0 / h, rw/w, rh/h])
                except:
                    continue

        # print("roadmap sucess")

        # ss = img.numpy()
        # for i, n in enumerate(annos_n):
        #     tx, ty = int((n[1] - n[3] / 2) * w), int((n[2] - n[4] / 2) * h)
        #     bx, by = int((n[1] + n[3] / 2) * w), int((n[2] + n[4] / 2) * h)
        #     cv2.rectangle(ss, (tx, ty), (bx, by), (0, 255, 0), 1)
        # cv2.imwrite("/home/chen/Desktop/ss/" + str(annos_n[0][1]) + 'ss.jpg', ss)
        return img.numpy(), np.array(annos_n)
    except Exception as e:
        return data[0], data[1]

def fill_duck_normal(data):
    try:
        img, annos, roadmap = data
        img = torch.tensor(img)
        roadmap = torch.tensor(roadmap)

        # constrain-----------------------------------------------------------------------------------------------------
        ch, cw = roadmap.shape
        sh = int(ch * 0.2)
        sw = int(cw * 0.2)
        roadmap[0:ch, 0:sw] = 0
        roadmap[0:ch, cw-sw:cw] = 0
        roadmap[0:sh, 0:cw] = 0
        # cv2.imwrite("/home/chen/Desktop/dd/" + str(annos[0][1]) + 'ss.jpg', roadmap.numpy())

        # I. Get valid area.--------------------------------------------------------------------------------------------
        valid_idx = roadmap.view(-1)
        idx = torch.nonzero(valid_idx).view(-1)
        if idx.size(0) == 0:
            return img, annos

        # valid xy
        xs = idx % roadmap.size(1)
        ys = idx // roadmap.size(1)
        coor = torch.stack((xs, ys), dim=1)

        # ann
        annos_n = [s for s in annos]

        # II Calculate scale -------------------------------------------------------------------------------------------
        scale_factor = [1.25, 0.75]
        (h, w, _) = img.shape
        for i, an in enumerate(annos):
            tx, ty = int((an[1] - an[3]/2) * w), int((an[2] - an[4]/2) * h)
            bx, by = int((an[1] + an[3]/2) * w), int((an[2] + an[4]/2) * h)
            img_an = img[ty:by, tx:bx, :]
            idxs = torch.randint(low=0, high=coor.shape[0], size=(2,))

            for j, scale in enumerate(scale_factor):
                (ah, aw, _) = img_an.shape
                rh, rw = int(ah*scale), int(aw*scale)
                img_scale = cv2.resize(img_an.numpy(), (rw, rh), interpolation=cv2.INTER_LINEAR)
                coorxy = coor[idxs[j]]
                rtx, rty = int(coorxy[0].numpy()), int(coorxy[1].numpy())
                rbx, rby = rtx + rw, rty + rh

                try:
                    img[rty:rby, rtx:rbx, :] = torch.from_numpy(img_scale)
                    annos_n.append([an[0], (rtx + rbx) / 2.0 / w, (rty + rby) / 2.0 / h, rw/w, rh/h])
                except:
                    continue

        # print("roadmap sucess")

        # ss = img.numpy()
        # for i, n in enumerate(annos_n):
        #     tx, ty = int((n[1] - n[3] / 2) * w), int((n[2] - n[4] / 2) * h)
        #     bx, by = int((n[1] + n[3] / 2) * w), int((n[2] + n[4] / 2) * h)
        #     cv2.rectangle(ss, (tx, ty), (bx, by), (0, 255, 0), 1)
        # cv2.imwrite("/home/chen/Desktop/ss/" + str(annos_n[0][1]) + 'ss.jpg', ss)
        return img.numpy(), np.array(annos_n)
    except Exception as e:
        return data[0], data[1]

# yolov5 albumentation--------------------------------------------------------------------------------------------------
class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A

            self.transform = A.Compose([
                A.Blur(p=0.1),
                A.MedianBlur(p=0.1),
                A.ToGray(p=0.01)],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            print(', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            print(f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments

def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])



# synthesize
def to_image_list_synthesize_4(transposed_info, size_divisible=0):
    tensors = transposed_info[0]
    if isinstance(tensors, (tuple, list)):
        pass
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


if __name__ == '__main__':
    img = cv2.imread("/home/chen/chen_p/chen_yolo4/datasets/Material/roadmap/road1.jpg")
    ann = [[0, 0.45598958333333334, 0.4203703703703704, 0.06614583333333333, 0.037037037037037035]]
    roadmap = cv2.imread("/home/chen/chen_p/chen_yolo4/datasets/Material/roadmap/m688.png", cv2.IMREAD_GRAYSCALE)
    data = (img, ann, roadmap)
    imgs, annos = fill_duck(data)

    # ------------------------------------------------------------------------------------------------------------------
    print(annos)
    cv2.imshow("ss", imgs.numpy())
    cv2.waitKey(0)
