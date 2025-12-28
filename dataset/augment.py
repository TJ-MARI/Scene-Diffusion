import cv2

import numpy as np


class Augment_pipeline(object):
    def __init__(self, augment_params):
        self.active = len(augment_params) >= 1
        self.Transforms = []
        if "flip_p" in augment_params.keys():
            self.Transforms.append(Flip_transform(augment_params["flip_p"]))
        if "crop_p" in augment_params.keys():
            self.Transforms.append(Crop_transform(augment_params["crop_p"], augment_params["crop_min_scale"]))

    def __call__(self, imgs):
        if self.active:
            for trans in self.Transforms:
                imgs = trans(imgs)
        return imgs


class Flip_transform(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, imgs):
        if not isinstance(imgs, dict):
            imgs = {"img": imgs}
        if np.random.rand() < self.p:
            for k in imgs.keys():
                imgs[k] = np.fliplr(imgs[k]).copy()
        return imgs


class Crop_transform(object):
    def __init__(self, p, min_scale):
        self.p = p
        self.min_scale = min_scale

    def __call__(self, imgs):
        if not isinstance(imgs, dict):
            imgs = {"img": imgs}
        if np.random.rand() < self.p:
            h, w = imgs[list(imgs.keys())[0]].shape[:2]
            scale = np.random.rand() * (1.0 - self.min_scale)  + self.min_scale
            start_y = int(np.random.rand() * (1.0 - scale) * h)
            start_y = max(start_y, 0)
            start_x = int(np.random.rand() * (1.0 - scale) * w)
            start_x = max(start_x, 0)
            end_y = int(start_y + h * scale)
            end_y = min(end_y, h)
            end_x = int(start_x + w * scale)
            end_x = min(end_x, w)
            for k in imgs.keys():
                imgs[k] = imgs[k][start_y:end_y, start_x:end_x, :]
                imgs[k] = cv2.resize(imgs[k], (h, w)).copy()
        return imgs