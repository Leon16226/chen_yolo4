from yrrnet.datasets import make_dataloader
from yrrnet.datasets.drones_det import DronesDET
from yrrnet.configs.rrnet_config import Config
import cv2

if __name__ == '__main__':

    train_loader, val_loader = make_dataloader(Config, DronesDET.collate_fn())
    for i, (image, annotation, roadmap, name) in enumerate(train_loader):
        cv2.imshow("ss", image)
        cv2.waitKey(100)
