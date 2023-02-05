from math import log
import torch
from torch import nn
from .cnn_blocks import PeriodicConv2D, ResidualBlock, Upsample
from src.datamodules import CONSTANTS, NAME_TO_VAR

# Large based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        history,
        hidden_channels=128,
        activation="leaky",
        out_channels=None,
        upsampling=1,
        norm: bool = True,
        dropout: float = 0.1,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels * history
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.upsampling = upsampling

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

        insize = self.in_channels
        # Project image into feature map
        self.image_proj = PeriodicConv2D(insize, hidden_channels, kernel_size=7, padding=3)

        blocks = []
        for i in range(n_blocks):
            blocks.append(
                ResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    norm=True,
                    dropout=dropout
                )
            )
        
        if upsampling > 1:
            n_upsamplers = int(log(upsampling, 2))
            for i in range(n_upsamplers - 1):
                blocks.append(Upsample(hidden_channels))
                blocks.append(self.activation)
            blocks.append(Upsample(hidden_channels))

        self.blocks = nn.ModuleList(blocks)

        if norm:
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.out_channels
        self.final = PeriodicConv2D(hidden_channels, out_channels, kernel_size=7, padding=3)
        
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

        for m in self.blocks:
            x = m(x)

        pred = self.final(self.activation(self.norm(x))) # B, C, H, W
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

    def upsample(self, x, y, out_vars, transform, metric):
        with torch.no_grad():
            pred = self.predict(x)
        return [m(pred, y, transform, out_vars) for m in metric], pred

# model = ResNet(in_channels=2, out_channels=2).cuda()
# x = torch.randn((64, 2, 32, 64)).cuda()
# y = model.predict(x)
# print (y.shape)
