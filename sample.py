import os
import cv2
import numpy as np

from omegaconf import OmegaConf

import pytorch_lightning as pl

from cldm.model import create_model, load_state_dict
from torch.utils.data import Dataset, DataLoader


def get_last_ckpt(resume_path):
    resume_path = os.path.join(resume_path, "checkpoints") if not resume_path.endswith("checkpoints") else resume_path
    _max_epoch, _last_ckpt = 0, ""
    for f in os.listdir(resume_path):
        if f.endswith(".ckpt"):
            if int(f.split("=")[1].split("-")[0]) > _max_epoch:
                _last_ckpt = f
                _max_epoch = int(f.split("=")[1].split("-")[0])
    return os.path.join(resume_path, _last_ckpt)


class Single_Image_Dataset(Dataset):
    def __init__(self, img_path, prompt, sample_num=1, save_path="./sample_log"):
        self.controls = [img_path] * sample_num
        self.prompts = ["High resolution photography interior design, " + prompt + ", 8k, photorealistic, realistic light."] * sample_num
        self.neg_prompts = ["semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts"] * sample_num

        self.sample_num = sample_num
        self.save_path = save_path

    def _load_img_as_numpy(self, file_name):
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0   # [0, 1]
        return img

    def _channel_alignment(self, imgs):
        for k in imgs.keys():
            imgs[k] = np.transpose(imgs[k], (2, 0, 1))
        return imgs

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        rts = {"pmt": self.prompts[idx], "neg_pmt": self.neg_prompts[idx]}
        img = self._load_img_as_numpy(self.controls[idx])
        rts["img"] = self._channel_alignment({"img": img})
        rts.update({"save_path": os.path.join(self.save_path, f"sample_{idx}.png")})
        return rts


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="./example/bed.png")
    parser.add_argument("--prompt", type=str, default="a bed, an arm chair, a corner table")
    parser.add_argument("--sample_num", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="./sample_log")
    args = parser.parse_args()

    config_path = './configs/cldm_v21_512_latctrl_mrcoltrans.yaml'
    ckpt_path = './model/scene-diffusion.ckpt'

    # First use cpu to load configs. Pytorch Lightning will automatically move it to GPUs.
    print(f'Loaded model config from [{config_path}]')
    config = OmegaConf.load(config_path)

    # You may need to manually download openai/clip-vit-large-patch14
    ###
    # open_clip_ckpt = "/Data/hanx/OC_ckpt/open_clip_pytorch_model.bin"
    # if "cldm_v21" in config_path :
    #     config.model.params.cond_stage_config.params.version = open_clip_ckpt  # for sd21
    ###

    model = create_model(config).cpu()
    model.load_state_dict(load_state_dict(get_last_ckpt(ckpt_path), location='cpu'))
    model.sd_locked = True
    model.only_mid_control = False

    # Misc
    dataset = Single_Image_Dataset(img_path=args.img_path, prompt=args.prompt, sample_num=args.sample_num, save_path=args.save_path)
    dataloader = DataLoader(dataset, pin_memory=True, num_workers=1, batch_size=1, persistent_workers=True, shuffle=False)
    trainer = pl.Trainer(gpus=1, precision=32, max_epochs=1)

    # Test!
    trainer.test(model, dataloader)
