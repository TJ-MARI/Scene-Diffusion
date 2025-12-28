import os
import numpy as np
from PIL import Image

import torch
import torch as th
import torch.nn as nn
import einops
from einops import rearrange, repeat
import torchvision
from torchvision.utils import make_grid

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from .coltrans_loss import multi_resolution_coltrans_loss


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []

        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        unet_feat = [x.detach() for x in hs]
        unet_feat.append(h.detach())

        ctrl_pointer, hs_pointer = -1, -1
        # assert (len(control) - len(hs)) == 1
        if control is not None:
            h += control[ctrl_pointer]
            ctrl_pointer -= 1

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs[hs_pointer]], dim=1)
                hs_pointer -= 1
            else:
                h = torch.cat([h, hs[hs_pointer] + control[ctrl_pointer]], dim=1)
                ctrl_pointer -= 1
                hs_pointer -= 1

            h = module(h, emb, context)

        h = h.type(x.dtype)

        return self.out(h), unet_feat


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            latent_control=False,
            noise_control=False
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.latent_control = latent_control
        self.noise_control = noise_control
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        if self.latent_control:
            self.input_hint_block = TimestepEmbedSequential(
                    zero_module(conv_nd(dims, hint_channels, model_channels, 3, padding=1))
                )
        else:
            self.input_hint_block = TimestepEmbedSequential(
                conv_nd(dims, hint_channels, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),  # 256 -> 128
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),  # 128 -> 64
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),  # 64 -> 32
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
            )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels  # use_spatial_transformer=True
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if not self.noise_control:
            guided_hint = self.input_hint_block(hint, emb, context)
            h = x.type(self.dtype)
        else:
            guided_hint = None
            h = hint.type(self.dtype)

        outs = []
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))  # 分支先预测，把所有的outs都出来

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control,
                 l_coltrans_weight=0.0, l_mrcoltrans_weight=0.0,
                 test_ddim_steps=100, test_guidance_scale=9.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        self.l_coltrans_weight = l_coltrans_weight
        self.l_mrcoltrans_weight = l_mrcoltrans_weight
        if self.l_mrcoltrans_weight > 0:
            self.control_loss = multi_resolution_coltrans_loss()

        self.test_ddim_steps = test_ddim_steps
        self.test_guidance_scale = test_guidance_scale

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = control.to(memory_format=torch.contiguous_format).float()

        if self.control_model.latent_control:
            control = (control * 2.0) - 1.0
            control = self.encode_first_stage(control).mean

        ctrl_loss_params = {}
        if self.l_mrcoltrans_weight > 0 and "ctrl_mask" in batch.keys():
            mask = batch["ctrl_mask"]
            if bs is not None:
                mask = mask[:bs]
            mask = mask.to(self.device)
            ctrl_loss_params["ctrl_mask"] = mask
            tgt_rgb = batch["tgt"]
            if bs is not None:
                tgt_rgb = tgt_rgb[:bs]
            tgt_rgb = tgt_rgb.to(self.device)
            ctrl_loss_params["tgt_rgb"] = tgt_rgb
            ctrl_rgb = batch["ctrl"]
            if bs is not None:
                ctrl_rgb = ctrl_rgb[:bs]
            ctrl_rgb = ctrl_rgb.to(self.device)
            ctrl_loss_params["ctrl_rgb"] = ctrl_rgb

        return x, dict(c_crossattn=[c], c_concat=[control]), ctrl_loss_params

    def shared_step(self, batch):
        x, c, ctrl_loss_params = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, ctrl_loss_params)
        return loss

    def forward(self, x, c, ctrl_loss_params, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:  # text prompt
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        if self.control_model.noise_control:   # harmful
            ctrl_noise = torch.randn_like(c["c_concat"][0])
            c["c_concat"] = [self.q_sample(x_start=i, t=t, noise=ctrl_noise) for i in c["c_concat"]]

        output = self.apply_model(x_noisy, t, c)

        if isinstance(output, tuple):
            model_output, unet_feat, ctrl_feat = output
        else:
            model_output = output

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss = 0

        loss_simple = self.get_loss(model_output, noise, mean=False).mean([1, 2, 3])
        _logvar_t = self.logvar[t].to(self.device)
        loss_simple = loss_simple / torch.exp(_logvar_t) + _logvar_t
        loss += self.l_simple_weight * loss_simple.mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        loss_vlb = self.get_loss(model_output, noise, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        if self.l_mrcoltrans_weight > 0:
            x0_pred = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
            x0_pred_rgb = self.decode_first_stage(x0_pred)
            x0_pred_rgb = torch.clamp(x0_pred_rgb, -1., 1.)
            x0_pred_rgb = (x0_pred_rgb + 1.0) / 2.0

            loss_mrcoltrans = self.control_loss(t, x0_pred_rgb,
                                                mask=ctrl_loss_params["ctrl_mask"],
                                                tgt=ctrl_loss_params["tgt_rgb"],
                                                ctrl=ctrl_loss_params["ctrl_rgb"],)

            loss += (self.l_mrcoltrans_weight * loss_mrcoltrans)
            loss_dict.update({f'{prefix}/loss_coltrans': loss_mrcoltrans})

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        if cond['c_concat'] is None:
            eps, _ = self.model.diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
            return eps, None, None
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps, unet_feat = self.model.diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            return eps, unet_feat, control


    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        use_ddim = self.test_ddim_steps is not None
        ddim_eta = 0.0

        images = dict()
        z, c, _ = self.get_input(batch, self.first_stage_key)  # 这个 z 我们是不用的
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        N = z.shape[0]
        # images["target"] = batch[self.first_stage_key]
        # images["control"] = batch[self.control_key]
        # images["prompt"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        # samples
        if "neg_pmt" not in batch.keys():
            uc_cross = self.get_unconditional_conditioning(N)
        else:
            uc_cross = self.get_learned_conditioning(batch["neg_pmt"])
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        sample_latent, intermediates = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                       batch_size=N, ddim=use_ddim,
                                                       ddim_steps=self.test_ddim_steps, eta=ddim_eta,
                                                       unconditional_guidance_scale=self.test_guidance_scale,
                                                       unconditional_conditioning=uc_full,
                                                       log_every_t= 100)
        sample = self.decode_first_stage(sample_latent)
        sample = (sample + 1.0) / 2.0  # 记住：First Stage 的输入输出都是 [-1, 1] 的区间
        images["sample"] = torch.clamp(sample, 0., 1.)

        sample = images["sample"].detach().cpu()
        sample = torchvision.utils.make_grid(sample, nrow=4)
        sample = sample.transpose(0, 1).transpose(1, 2).squeeze(-1)
        sample = (sample.numpy() * 255).astype(np.uint8)
        os.makedirs(os.path.split(batch["save_path"])[0], exist_ok=True)
        Image.fromarray(sample).save(batch["save_path"])


    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, _ = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["target"] = batch[self.first_stage_key]
        log["control"] = batch[self.control_key]
        log["prompt"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            x_samples_cfg = (x_samples_cfg + 1.0) / 2.0
            log["sample"] = torch.clamp(x_samples_cfg, 0., 1.)

        if plot_diffusion_rows:  # 默认不开，加不同程度的噪声，直接解码，主要反映自编码器对图像的熟悉程度
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:  # 默认不开，直接根据当前的 prompt 和 control 进行采样，没有用 classifier-free guidance
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        for k in log:
            if isinstance(log[k], torch.Tensor):
                log[k] = log[k].detach().cpu()

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        # b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, self.first_stage_model.encoder.resolution // 8, self.first_stage_model.encoder.resolution // 8)
        # shape = (self.channels, 32, 32)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
