import math
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


class coltrans_loss(nn.Module):
    def __init__(self):
        super(coltrans_loss, self).__init__()
        _rgb2lms_mat = torch.tensor([[0.3811, 0.5783, 0.0402],
                                     [0.1967, 0.7244, 0.0782],
                                     [0.0241, 0.1288, 0.8444]],
                                    dtype=torch.float32)
        self.register_buffer("rgb2lms_mat", _rgb2lms_mat)
        _lms2lab_mat = torch.tensor([[1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)],
                                     [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)],
                                     [1 / math.sqrt(2), -1 / math.sqrt(2), 0]],
                                    dtype=torch.float32)
        self.register_buffer("lms2lab_mat", _lms2lab_mat)

    def rgb2lab(self, rgb):  # 处理的是 flatten 后的 rgb
        bs = rgb.shape[0]
        _rgb2lms_mat = repeat(self.rgb2lms_mat, 'x y -> b x y', b=bs)

        lms = torch.bmm(_rgb2lms_mat, rgb)

        # 后续有对数操作，没法处理黑色像素点, 所以要事先滤除，这样对数操作以后，这些点也就是 0
        # 取对数的操作，为了安全要截断
        lms[lms == 0] = 1
        lms = torch.log10(lms)
        lms = torch.clamp_(lms, min=1e-4)

        _lms2lab_mat = repeat(self.lms2lab_mat, 'x y -> b x y', b=bs)
        lab = torch.bmm(_lms2lab_mat, lms)
        return lab

    def shift_color(self, x0_pred_lab, tgt_lab_mean, tgt_lab_std, ctrl_lab_mean, ctrl_lab_std):
        ps = x0_pred_lab.shape[-1]
        tgt_lab_mean = repeat(tgt_lab_mean, 'b c -> b c p', p=ps)
        tgt_lab_std = repeat(tgt_lab_std, 'b c -> b c p', p=ps)
        ctrl_lab_mean = repeat(ctrl_lab_mean, 'b c -> b c p', p=ps)
        ctrl_lab_std = repeat(ctrl_lab_std, 'b c -> b c p', p=ps)
        tgt_lab_std = torch.clamp(tgt_lab_std, 1e-4)
        return (x0_pred_lab - tgt_lab_mean) / tgt_lab_std * ctrl_lab_std + ctrl_lab_mean

    def _mean_in_mask(self, x, mask):
        return torch.sum(x * mask, dim=2) / torch.clamp(torch.sum(mask, dim=2), min=1)

    def _std_in_mask(self, x, x_mean, mask):
        ps = x.shape[-1]
        _rpt_mean = repeat(x_mean, 'b c -> b c p', p=ps)
        return torch.sqrt(torch.sum(torch.pow((x - _rpt_mean), 2) * mask, dim=2) / torch.clamp(torch.sum(mask, dim=2) - 1, min=1))

    def forward(self, x0_pred_rgb, mask, tgt, ctrl):

        h, w = x0_pred_rgb.shape[-2], x0_pred_rgb.shape[-1]
        x0_pred_rgb = rearrange(x0_pred_rgb, 'b c h w -> b c (h w)')
        tgt = rearrange(tgt, 'b c h w -> b c (h w)')
        ctrl = rearrange(ctrl, 'b c h w -> b c (h w)')
        x0_pred_lab = self.rgb2lab(x0_pred_rgb)
        tgt_lab = self.rgb2lab(tgt)
        ctrl_lab = self.rgb2lab(ctrl)

        # 注意 mean 和 std 的统计范围
        _flatten_mask = rearrange(mask, 'b c h w -> b c (h w)')
        tgt_lab_mean = self._mean_in_mask(tgt_lab, _flatten_mask)
        tgt_lab_std = self._std_in_mask(tgt_lab, tgt_lab_mean, _flatten_mask)

        ctrl_lab_mean = self._mean_in_mask(ctrl_lab, _flatten_mask)
        ctrl_lab_std = self._std_in_mask(ctrl_lab, ctrl_lab_mean, _flatten_mask)

        shifted = self.shift_color(x0_pred_lab, tgt_lab_mean, tgt_lab_std, ctrl_lab_mean, ctrl_lab_std)
        shifted = rearrange(shifted, 'b c (h w) -> b c h w', h=h, w=w)
        ctrl_lab = rearrange(ctrl_lab, 'b c (h w) -> b c h w', h=h, w=w)
        loss = F.mse_loss(shifted, ctrl_lab, reduce=False)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss


class multi_resolution_coltrans_loss(coltrans_loss):
    def __init__(self):
        super(multi_resolution_coltrans_loss, self).__init__()
        _gs_kernel = torch.Tensor([[[[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                                   [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                                   [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                    [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]]
                                   ])  # [3,3,3,3]
        self.register_buffer("gs_kernel", _gs_kernel)

    def coltrans_loss_single_sample(self, x0_pred_rgb, mask, tgt, ctrl):
        h, w = x0_pred_rgb.shape[-2], x0_pred_rgb.shape[-1]
        x0_pred_rgb = rearrange(x0_pred_rgb, 'b c h w -> b c (h w)')
        tgt = rearrange(tgt, 'b c h w -> b c (h w)')
        ctrl = rearrange(ctrl, 'b c h w -> b c (h w)')
        x0_pred_lab = self.rgb2lab(x0_pred_rgb)
        tgt_lab = self.rgb2lab(tgt)
        ctrl_lab = self.rgb2lab(ctrl)

        # 注意 mean 和 std 的统计范围
        _flatten_mask = rearrange(mask, 'b c h w -> b c (h w)')
        tgt_lab_mean = self._mean_in_mask(tgt_lab, _flatten_mask)
        tgt_lab_std = self._std_in_mask(tgt_lab, tgt_lab_mean, _flatten_mask)

        ctrl_lab_mean = self._mean_in_mask(ctrl_lab, _flatten_mask)
        ctrl_lab_std = self._std_in_mask(ctrl_lab, ctrl_lab_mean, _flatten_mask)

        shifted = self.shift_color(x0_pred_lab, tgt_lab_mean, tgt_lab_std, ctrl_lab_mean, ctrl_lab_std)
        shifted = rearrange(shifted, 'b c (h w) -> b c h w', h=h, w=w)
        ctrl_lab = rearrange(ctrl_lab, 'b c (h w) -> b c h w', h=h, w=w)
        loss = F.mse_loss(shifted, ctrl_lab, reduce=False)
        loss = torch.sum(loss * mask) / torch.clamp(torch.sum(mask), min=1)
        return loss

    def gaussian_blur_downsample(self, img):
        img = F.conv2d(img, self.gs_kernel, stride=1, padding=1)
        img = F.avg_pool2d(img, kernel_size=(2,2))
        return img

    def forward(self, ts, x0_pred_rgb, mask, tgt, ctrl):
        loss = 0
        for _t, _x0_pred_rgb, _mask, _tgt, _ctrl in zip(ts, x0_pred_rgb, mask, tgt, ctrl):
            _x0_pred_rgb = _x0_pred_rgb.unsqueeze(0)
            _tgt = _tgt.unsqueeze(0)
            _ctrl = _ctrl.unsqueeze(0)
            _mask = _mask.unsqueeze(0)
            if _t >= 400:
                _x0_pred_rgb = torch.clamp(self.gaussian_blur_downsample(_x0_pred_rgb), 0, 1)
                _tgt = torch.clamp(self.gaussian_blur_downsample(_tgt), 0, 1)
                _ctrl = torch.clamp(self.gaussian_blur_downsample(_ctrl), 0, 1)
                _mask = torch.clamp(F.interpolate(_mask, scale_factor=0.5), 0, 1)
            if _t >= 700:
                _x0_pred_rgb = torch.clamp(self.gaussian_blur_downsample(_x0_pred_rgb), 0, 1)
                _tgt = torch.clamp(self.gaussian_blur_downsample(_tgt), 0, 1)
                _ctrl = torch.clamp(self.gaussian_blur_downsample(_ctrl), 0, 1)
                _mask = torch.clamp(F.interpolate(_mask, scale_factor=0.5), 0, 1)
            loss += self.coltrans_loss_single_sample(_x0_pred_rgb, _mask, _tgt, _ctrl)
        return loss
