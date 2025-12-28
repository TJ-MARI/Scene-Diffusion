import os
import cv2
import copy
import numpy as np

from dataset.augment import Augment_pipeline
from torch.utils.data import Dataset, DataLoader


class Furniture_dataset(Dataset):
    def __init__(self, data_root, split="train", subset=None,
                 augment_params={}):

        # target
        self.tgt_root = os.path.join(data_root, "scene")

        # control
        self.ctrl_root = os.path.join(data_root, "control")

        # control mask
        self.ctrl_mask_root = os.path.join(data_root, "mask")

        # data
        self.is_train = (split == "train")

        if subset == 'all':
            _subset_iter = ["bed", "sofa", "shelf", "table", "chair"]
        else:
            _subset_iter = [subset]

        self.data = []
        for _subset in _subset_iter:
            _subset_split = split + "_" + _subset

            _data = []
            for file in os.listdir(os.path.join(self.ctrl_root, _subset_split)):
                _data.append(_subset_split + "/" + file.split(".")[0])

            _data.sort()

            self.data.extend(_data)

        # prompt
        self.pmt_dict = {}
        self.pmt_dict[split] = np.load(os.path.join(data_root, "prompt/prompt_{}.npy".format(split)), allow_pickle=True).item()

        # augmentation
        self.augment = Augment_pipeline(augment_params)

    def __len__(self):
        return len(self.data)

    # 所有图片数据统一的通道顺序是：CHW, 范围是[0, 1]
    def _load_img_as_rgb(self, file_name):
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # [0, 1]
        return img

    def _scale_and_channel_alignment(self, imgs):
        for k in imgs.keys():
            imgs[k] = np.transpose(imgs[k], (2, 0, 1))
        return imgs

    def __getitem__(self, idx):
        item = self.data[idx]
        split = "train" if self.is_train else "test"
        pmt = self.pmt_dict[split][item.split("/")[1].split("_")[0]]
        rts = {"item": item, "pmt": pmt}

        tgt_file = os.path.join(os.path.join(self.tgt_root, split), item.split("/")[1].split("_")[0] + ".jpg")
        tgt = self._load_img_as_rgb(tgt_file)

        ctrl_file = os.path.join(self.ctrl_root, item + ".png")
        ctrl = self._load_img_as_rgb(ctrl_file)

        _imgs = {"tgt": tgt, "ctrl": ctrl, "tgt_rgb": copy.copy(tgt), "ctrl_rgb": copy.copy(ctrl)}
        if self.is_train:
            ctrl_mask_file = os.path.join(self.ctrl_mask_root, item + ".png")  # 3通道 [0, 255]
            ctrl_mask = self._load_img_as_rgb(ctrl_mask_file)
            _imgs.update({"ctrl_mask": ctrl_mask})

        _imgs = self.augment(_imgs)
        _imgs = self._scale_and_channel_alignment(_imgs)
        rts.update(_imgs)

        return rts


def get_dataloader(dataset, batch_size):
    if dataset.is_train:
        return DataLoader(dataset, pin_memory=True, num_workers=8, batch_size=batch_size, persistent_workers=True, shuffle=True)
    else:
        return DataLoader(dataset, pin_memory=True, num_workers=8, batch_size=batch_size, persistent_workers=True, shuffle=False)