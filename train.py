import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.furniture import Furniture_dataset,get_dataloader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


if __name__ == '__main__':

    # Configs
    config_path = 'configs/cldm_v21_512_latctrl_mridcoltrans.yaml'
    resume_path = './ckpt/control_sd21_latctrl_mrcoltrans_ini.ckpt'
    gpus = 1 # set your gpu number
    batch_size = 1 # set your batch size
    learning_rate = 1e-5

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
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = True
    model.only_mid_control = False

    # Misc
    dataset = Furniture_dataset(data_root="./data", split="train",
                                augment_params={"flip_p": 0.5, "crop_p": 0.8, "crop_min_scale": 0.85})
    dataloader = get_dataloader(dataset, batch_size)

    img_logger = ImageLogger(batch_frequency=1000)
    ckpt_logger = ModelCheckpoint(every_n_train_steps=25000)

    trainer = pl.Trainer(gpus=gpus, precision=32, callbacks=[img_logger, ckpt_logger], max_epochs=20)

    # Train!
    trainer.fit(model, dataloader)
    # Resume !
    # trainer.fit(model, dataloader, ckpt_path="./lightning_logs/version_0/checkpoints/epoch=7-step=34631.ckpt")
