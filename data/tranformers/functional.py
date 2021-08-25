import torch
import random
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms.functional as torchtransform
import numpy as np
import torch.nn.functional as F
import math
import cv2

# ada sampling----------------------------------------------------------------------------------------------------------
def fill_duck(data):
    try:
        img, annos, roadmap = data
        img = torch.tensor(img)
        roadmap = torch.tensor(roadmap)

        # I. Get valid area.--------------------------------------------------------------------------------------------
        valid_idx = roadmap.view(-1)
        idx = torch.nonzero(valid_idx).view(-1)
        if idx.size(0) == 0:
            return img, annos

        # valid xy
        ys = idx % roadmap.size(1)
        xs = idx // roadmap.size(1)
        coor = torch.stack((xs, ys), dim=1)

        # ann
        annos_n = [s for s in annos]

        # II Calculate scale -------------------------------------------------------------------------------------------
        scale_factor = [1.25, 0.75, 0.5]
        (h, w, _) = img.shape
        for i, an in enumerate(annos):
            tx, ty = int((an[1] - an[3]/2) * w), int((an[2] - an[4]/2) * h)
            bx, by = int((an[1] + an[3]/2) * w), int((an[2] + an[4]/2) * h)
            img_an = img[ty:by, tx:bx, :]
            idxs = torch.randint(low=0, high=coor.shape[0], size=(3,))

            for j, scale in enumerate(scale_factor):
                (ah, aw, _) = img_an.shape
                rh, rw = int(ah*scale), int(aw*scale)
                img_scale = cv2.resize(img_an.numpy(), (rw, rh), interpolation=cv2.INTER_LINEAR)
                coorxy = coor[idxs[j]]
                rtx, rty = int(coorxy[0].numpy()), int(coorxy[1].numpy())
                rbx, rby = rtx + rw, rty + rh

                try:
                    img[rty:rby, rtx:rbx, :] = torch.from_numpy(img_scale)
                    annos_n.append([an[0], (rtx + rbx) / 2 / w, (rty + rby) / 2 / h, rw/w, rh/h])
                except:
                    continue
        # cv2.imwrite("/home/chen/Desktop/ss/" + str(annos_n[0][1]) + 'ss.jpg', img.numpy())
        return img.numpy(), np.array(annos_n)
    except Exception as e:
        return data[0], data[1]


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
