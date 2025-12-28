# Scene Diffusion

This is official implementation of MM'24 oral paper *Scene Diffusion: Text-driven Scene Image Synthesis Conditioning on a Single 3D Model*.
Conditioned only on simple renderings of a single 3D model, our method can complete the background and generate high-quality scene images of the model in the specified scene. The generated results not only achieve photorealistic visual effects but also faithfully preserve the texture details of the original 3D model. Our method demonstrates excellent performance in scenarios involving single objects, multiple objects, diverse text descriptions, and various viewing angles.

![The results of Scene Diffusion](./asset/showcase.png)  

Our current open-source content includes:
- [x] Complete training and testing datasets
- [x] Training code and required preprocessing scripts
- [x] Sampling code and pretrained models


### Environment

1. Clone the repository

```bash
git clone https://github.com/TJ-MARI/Scene-Diffusion.git
cd Scene-Diffusion
```

2. Install dependencies
- We provide a complete environment configuration file, and you can directly create a conda environment with it. This project is built on the codebase of ControlNet and Stable Diffusion v2.1 (without using diffusers). If you already have an environment that can run them, you can skip this step.

```bash
conda env create -f environment.yaml
conda activate scene-diffusion
```


### Sampling

1. Download the checkpoints

- You can download our [ckpt](https://huggingface.co/hikkilover/scene-diffusion) from Hugging Face and place it in the ./ckpt directory
- You may need to manually download the OpenCLIP model from `openai/clip-vit-large-patch14` and place `open_clip_pytorch_model.bin` in the ./ckpt directory

2. Generate scene images

```bash
python sample.py --img_path ./example/bed.png --prompt "a bed, an arm chair, a corner table" --sample_num 4 --save_path ./sample_log
```

3. Sample with your own furniture model 
- For testing on your own cases, you need to prepare the input renderings yourself. We recommend using Blender and ensure that the background color is consistent with the example images we provide.


### Training

1. Download Dataset 

- You can download our [dataset](https://huggingface.co/datasets/hikkilover/3D-FUTURE-FCO) from Hugging Face and place it in the ./data directory
- The directory structure of the dataset is shown below

```
./data/
├── control/                # Condition images
│   ├── train_{category}/
│   └── test_{category}/
├── mask/                   # Mask image
│   ├── train_{category}/
│   └── test_{category}/
├── prompt/                 # Text prompt of scene images
│   ├── prompt_train.npy
│   └── prompt_test.npy
└── scene/                  # Scene images (Target images)
    ├── train/
    └── test/
```

2. Prepare the initial checkpoint

- Download the pretrained Stable Diffusion v2.1 model `stabilityai` and place the `v2-1_512-ema-pruned.ckpt` in the ./ckpt directory.
- run the `tool_add_control.py` script to add ControlNet to the Stable Diffusion v2.1 model.
- We found that the repository of stabilityai/stable-diffusion-2-1-base is currently unavailable. Therefore, we provide a processed initial checkpoint on [Hugging Face](https://huggingface.co/hikkilover/scene-diffusion), and you can directly download `control_sd21_latctrl_mrcoltrans_ini.ckpt` from it.

3. Train the model

```bash
python train.py
```

## Citation

If you use this project for research, please cite the relevant paper.

```
@inproceedings{han2024scene,
  title={Scene Diffusion: Text-driven Scene Image Synthesis Conditioning on a Single 3D Model},
  author={Han, Xuan and Zhao, Yihao and You, Mingyu},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)},
  pages={7862--7870},
  year={2024}
}
```

