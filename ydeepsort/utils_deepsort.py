import cv2
import numpy as np

# rescale boxes
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2   # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2   # y center
    y[:, 2] = (x[:, 2] - x[:, 0])   # width
    y[:, 3] = (x[:, 3] - x[:, 1])   # height
    return y


def _xyxy_to_tlwh(bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h


def _xywh_to_xyxy(bbox_xywh, height, width):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2


# preprocess
size = (64, 128)
# norm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])

def norm(im):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # output = im.copy()
    for i in np.arange(0, 3):
        im[:, :, i] = (im[:, :, i] - mean[i]) / std[i]
    return im


def _preprocess(im_crops):
    """
    TODO:
        1. to float with scale from 0 to 1
        2. resize to (64, 128) as Market1501 dataset did
        3. concatenate to a numpy array
        3. to torch Tensor
        4. normalize
    """

    def _resize(im, size):
        return cv2.resize(im.astype(np.float32) / 255., size)

    im_batch = np.concatenate([np.expand_dims(norm(_resize(im, size)).transpose(2, 0, 1), axis=0) for im in im_crops], axis=0)
    # im_batch = [np.expand_dims(norm(_resize(im, size)), axis=0) for im in im_crops]
    return im_batch