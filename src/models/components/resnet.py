from torch import nn
from .cnn_blocks import PeriodicConv2D, ResidualBlock
from .base import BaseModel

# Large based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License


class ResNet(BaseModel):
    def __init__(
        self,
        in_channels,
        history,
        hidden_channels=128,
        activation="leaky",
        out_channels=None,
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

        self.blocks = nn.ModuleList(blocks)

        if norm:
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.out_channels
        self.final = PeriodicConv2D(hidden_channels, out_channels, kernel_size=7, padding=3)

    def predict(self, inp, in_vars, out_vars):
        # inp: B, T, C, H, W
        x = inp.flatten(1, 2)
        x = self.image_proj(x)

        for m in self.blocks:
            x = m(x)

        pred = self.final(self.activation(self.norm(x))) # B, C, H, W
        return self.replace_constant_variables(inp, pred, in_vars, out_vars)

# model = ResNet(in_channels=2, out_channels=2).cuda()
# x = torch.randn((64, 2, 32, 64)).cuda()
# y = model.predict(x)
# print (y.shape)
