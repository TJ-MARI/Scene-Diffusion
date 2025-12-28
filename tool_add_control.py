import os
from omegaconf import OmegaConf
import torch
from cldm.model import create_model

from cldm.hack import disable_verbosity, enable_sliced_attention
disable_verbosity()
save_memory = False
if save_memory:
    enable_sliced_attention()
def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


config_path = 'configs/cldm_v21_512_latctrl_mrcoltrans.yaml'

input_path = "./ckpt/v2-1_512-ema-pruned.ckpt"
output_path = "./ckpt/control_sd21_latctrl_mrcoltrans_ini.ckpt"

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'


print(f'Loaded model config from [{config_path}]')
config = OmegaConf.load(config_path)

# You may need to manually download openai/clip-vit-large-patch14
###
# open_clip_ckpt = "/Data/hanx/OC_ckpt/open_clip_pytorch_model.bin"
# if "cldm_v21" in config_path :
#     config.model.params.cond_stage_config.params.version = open_clip_ckpt  # for sd21
###

model = create_model(config)

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
