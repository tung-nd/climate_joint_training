from math import log
from typing import List, Tuple, Union

import torch
from torch import nn
from .cnn_blocks import PeriodicConv2D, DownBlock, UpBlock, MiddleBlock, Downsample, Upsample
from src.datamodules import CONSTANTS, NAME_TO_VAR

# Large based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        history,
        hidden_channels=64,
        activation="leaky",
        out_channels=None,
        norm: bool = True,
        dropout: float = 0.1,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels * history
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.3)
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = self.in_channels
        n_channels = hidden_channels
        # Project image into feature map
        self.image_proj = PeriodicConv2D(insize, n_channels, kernel_size=7, padding=3)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, has_attn=mid_attn, activation=activation, norm=norm, dropout=dropout)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        dropout=dropout,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels, out_channels, has_attn=is_attn[i], activation=activation, norm=norm, dropout=dropout
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.BatchNorm2d(n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.out_channels
        self.final = PeriodicConv2D(in_channels, out_channels, kernel_size=7, padding=3)
        
    def get_constant_ids(self, all_ids, vars):
        constant_ids = []
        for id in all_ids:
            var = vars[id]
            if var in NAME_TO_VAR:
                if NAME_TO_VAR[var] in CONSTANTS:
                    constant_ids.append(id)
        return constant_ids

    def predict(self, inp, in_vars, out_vars):
        # inp: B, T, C, H, W
        x = inp.flatten(1, 2)
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        pred = self.final(self.activation(self.norm(x)))
        # do not predict constant variables
        out_var_id_constants = self.get_constant_ids(range(pred.shape[-3]), out_vars)
        if len(out_var_id_constants) > 0:
            in_var_id_constants = self.get_constant_ids(range(inp.shape[-3]), in_vars)
            assert len(in_var_id_constants) == len(out_var_id_constants)
            pred[:, out_var_id_constants] = inp[:, 0, in_var_id_constants].to(pred.dtype)
        return pred

    def forward(self, x: torch.Tensor, y: torch.Tensor, variables, out_variables, metric, lat, return_pred=False):
        # B, C, H, W
        pred = self.predict(x, variables, out_variables)
        if return_pred:
            return [m(pred, y, out_variables, lat) for m in metric], x, pred
        else:
            return [m(pred, y, out_variables, lat) for m in metric], x

    def evaluate(self, x, y, variables, out_variables, transform, metrics, lat, clim, log_postfix):
        pred = self.predict(x, variables, out_variables)
        return [m(pred, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]

# model = Unet(in_channels=2, out_channels=2).cuda()
# x = torch.randn((64, 2, 32, 64)).cuda()
# y = model.predict(x)
# print (y.shape)
