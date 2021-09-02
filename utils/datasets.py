import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy, torch_distributed_zero_first

from yrrnet.datasets.transforms import *
from PIL import Image
from data.tranformers.functional import *
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
from itertools import repeat
from utils.datatools import *

help_url = ''
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
NUM_THREADS = min(8, os.cpu_count())

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

# Dataloader------------------------------------------------------------------------------------------------------------
def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      local_rank=-1, world_size=1, prefix=''):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(local_rank):

        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                        augment=augment,  # augment images
                                        hyp=hyp,  # augmentation hyperparameters
                                        rect=rect,  # rectangular training
                                        cache_images=cache,
                                        single_cls=opt.single_cls,
                                        stride=int(stride),
                                        pad=pad,
                                        prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if local_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             sampler=train_sampler,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


# Dataset---------------------------------------------------------------------------------------------------------------
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):

        # load image file-----------------------------------------------------------------------------------------------
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))  # os-agnostic
                parent = str(Path(p).parent) + os.sep
                if os.path.isfile(p):  # file
                    with open(p, 'r') as t:
                        t = t.read().splitlines()
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                elif os.path.isdir(p):  # folder
                    f += glob.iglob(p + os.sep + '*.*')
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted(
                [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats])
        except Exception as e:
            raise Exception('Error loading data from %s: %s\nSee %s' % (path, e, help_url))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index

        # init ---------------------------------------------------------------------------------------------------------
        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.mosaic = self.augment
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.albumentations = Albumentations() if augment else None

        # Define labels
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in
                            self.img_files]

        # Roadmap labels
        self.roadmap_files = [x.replace('train', 'roadmap').replace(os.path.splitext(x)[-1], '.png') for x in
                              self.img_files]

        # 1.label cache---------------------------------------------------------------------------------------------------
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'
        cache = self.cache_labels(cache_path)
        # self.label_files = img2label_paths(self.img_files)  # labels
        # cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        # cache, exists = self.cache_labels_v5(cache_path, prefix), False

        # 2.Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # 3.check labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            l = self.labels[i]  # label
            if l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                if single_cls:
                    l[:, 0] = 0
                self.labels[i] = l
                nf += 1  # file found

            else:
                ne += 1
                print('empty labels for image %s' % self.img_files[i])  # file empty
                os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s



        # Cache images into memory for faster training------------------------------------------------------------------
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)


    # cache label-------------------------------------------------------------------------------------------------------
    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = None
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def cache_labels_v5(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap_unordered(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            print('\n'.join(msgs))
        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}.')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = 0.4  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            print().info(f'{prefix}New cache created: {path}')
        except Exception as e:
            print.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable

        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp

        # if mosaic-----------------------------------------------------------------------------------------------------
        #if hyp['sp'] == 0:
        #     self.mosaic = False
        #else:
        #      self.mosaic = True

        # mosaic--------------------------------------------------------------------------------------------------------

        if self.mosaic and (True if np.random.randint(1, 11) <= 8 else False):
            img, labels = load_mosaic(self, index)
            shapes = None

            # mixup-----------------------------------------------------------------------------------------------------
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            # for test -------------------------------------------------------------------------------------------------
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

            if self.augment:
                if not self.mosaic:
                    img, labels = random_perspective(img, labels,
                                                     degrees=hyp['degrees'],
                                                     translate=hyp['translate'],
                                                     scale=hyp['scale'],
                                                     shear=hyp['shear'],
                                                     perspective=hyp['perspective'])

        # copy-paste----------------------------------------------------------------------------------------------------
        # index_c = np.random.randint(0, labels.shape[0], size=(3,))
        # x = np.array([labels[l] for l in index_c])
        # h0, w0 = img.shape[:2]  # orig hw
        # roadmap = np.random.randint(0, 2, (h0, w0))
        # r = self.img_size / max(h0, w0)
        # if r != 1:
        #     roadmap = np.random.randint(0, 2, (int(h0 * r), int(w0 * r)))
        #
        # data = (img, x, roadmap)
        # img, x = fill_duck_normal(data)
        # for xx in x:
        #     labels.append(xx) # ??????




        # nl------------------------------------------------------------------------------------------------------------
        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        # flip----------------------------------------------------------------------------------------------------------
        if self.augment:
            img, labels = self.albumentations(img, labels)

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert-------------------------------------------------------------------------------------------------------
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


# load mosaic ----------------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]

    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path, cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

        return img, (h0, w0), img.shape[:2] # img, hw_original, hw_resized
    else:

        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized

def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    yc, xc = s, s  # mosaic center x, y
    # yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)
        x = self.labels[index]


        # roadmap ----------------------------------------------------------------------------------------------------
        roadmappath = self.roadmap_files[index]
        if os.path.exists(roadmappath):
            # resize----------------------------------------------------------------------------------------------------
            roadmap = cv2.imread(roadmappath, cv2.IMREAD_GRAYSCALE)
            h0, w0 = roadmap.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                roadmap = cv2.resize(roadmap, (int(w0 * r), int(h0 * r)), interpolation=interp)

            data = (img, x, roadmap)
            img, x = fill_duck(data)

        elif len(x) <= 3:
            # copy and paste in own image if label < 3

            h0, w0 = img.shape[:2]  # orig hw
            roadmap = np.random.randint(0, 2, (h0, w0))
            r = self.img_size / max(h0, w0)
            if r != 1:
                roadmap = np.random.randint(0, 2, (int(h0 * r), int(w0 * r)))

            data = (img, x, roadmap)
            img, x = fill_duck_normal(data)


        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        # reszie--------------------------------------------------------------------------------------------------------
        # cv2.resize(img4, (self.img_size, self.img_size))
        # h, w, _ = img4.shape

        # Labels--------------------------------------------------------------------------------------------------------
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels-----------------------------------------------------------------------------------------------
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment-----------------------------------------------------------------------------------------------------------
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove


    return img4, labels4

def load_mosaic_v5(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)
        x = self.labels[index]

        # roadmap ----------------------------------------------------------------------------------------------------
        roadmappath = self.roadmap_files[index]
        if os.path.exists(roadmappath):
            # resize----------------------------------------------------------------------------------------------------
            roadmap = cv2.imread(roadmappath, cv2.IMREAD_GRAYSCALE)
            h0, w0 = roadmap.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                roadmap = cv2.resize(roadmap, (int(w0 * r), int(h0 * r)), interpolation=interp)

            data = (img, x, roadmap)
            img, x = fill_duck(data)

        elif len(x) <= 3:
            # copy and paste in own image if label < 3

            h0, w0 = img.shape[:2]  # orig hw
            roadmap = np.random.randint(0, 2, (h0, w0))
            r = self.img_size / max(h0, w0)
            if r != 1:
                roadmap = np.random.randint(0, 2, (int(h0 * r), int(w0 * r)))

            data = (img, x, roadmap)
            img, x = fill_duck_normal(data)



        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = x.copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


# other loader----------------------------------------------------------------------------------------------------------
class LoadImages:  # for inference
    def __init__(self, path, img_size=640):
        # path
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        # files
        if '*' in p:
            files = sorted(glob.glob(p))  # glob, return a list of paths
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        # images & videos
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        # init
        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        # from count =0
        path = self.files[self.count]

        # video
        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read() # read
            # if false
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')
        else:
            # Read image
            self.count += 1
            # img0 = cv2.imread(path,)  # opencv -> BGR
            img0 = cv2.imread(path, cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]
        # img = cv2.resize(img0, [self.img_size, self.img_size])

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    # load a video through opencv
    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # the number of all frames

    def __len__(self):
        return self.nf  # number of files

class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=640):
        self.img_size = img_size

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0

# load img from rtsp -> fps
class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        # it's can have a lot of sources
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        # num of sources
        n = len(sources)
        # save all frames every 4 grab
        self.imgs = [None] * n
        self.sources = sources
        # iter
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')  # the order of sources
            cap = cv2.VideoCapture(0 if s == '0' else s)  # opencv
            assert cap.isOpened(), 'Failed to open %s' % s
            # width & height
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps -> frame per second
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            # read -> bool,frame
            _, self.imgs[i] = cap.read()  # guarantee first frame

            # thread
            # target -> while thread.run(), process target
            # args -> the tuple contained with args push to targets
            # daemon -> if it is a background thread -> the b thread exit randomly, while the main thread end
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            # run()
            thread.start()

        print('')  # newline

        # check for common shapes -> if rect infer
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    # daemon thread's target -> read every 4th frame
    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # grab the next frame
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    # in order to iter
    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        # the way of quit
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack -> batch
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        # convert to contiguous array
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

