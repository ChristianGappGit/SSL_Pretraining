"""
ResNetAutoEnc
ResNet + 3D Conv Transp. + 3D Conv Transp.

analogously to ViTAutoEnc from https://github.com/Project-MONAI/tutorials/tree/main/self_supervised_pretraining/vit_unetr_ssl
code ViTAutoEnc:
https://docs.monai.io/en/0.8.1/_modules/monai/networks/nets/vitautoenc.html

"""


import torch
import torch.nn as nn

import math

from monai.networks.nets import ResNet as ResNetMonai
from monai.networks.layers import Conv


__all__ = ["ResNetAutoEnc"]



class ResNetAutoEnc(nn.Module):
    """
    ResNet-based autoencoder with 3D convolutional layers.
    encoder: ResNetMonai
    decoder: ConvTranspose3d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        resnet_block = "basic",
        layers = [3, 4, 6, 3],
        block_inplanes = [64, 128, 256, 512],
        conv1_t_size=7,
        conv1_t_stride=2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ):
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input
            out_channels: number of output channels.
            ResNet setup (block, layers, block_inplanes, conv1_t_size, conv1_t_stride).
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
        """

        super().__init__()

        self.encoder = ResNetMonai(
                block = resnet_block,
                layers = layers,
                block_inplanes = block_inplanes,
                spatial_dims = spatial_dims,
                n_input_channels=in_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=False,
                shortcut_type="B",
                widen_factor= 1.0,
                num_classes = None,
                feed_forward = False,
                bias_downsample = True,  # for backwards compatibility (also see PR #5477)
                #act is relu inplace standart
        )

        hidden_size = block_inplanes[-1]
        deconv_chns = hidden_size // 8 # 8 = 2**3

        if hidden_size % 8 != 0:
            raise ValueError(
                f"The last element of block_inplanes (hidden_size={hidden_size}) must be divisible by 2**3 = 8 "
                "to properly compute deconv channels."
            )

        conv_trans = Conv[Conv.CONVTRANS, spatial_dims]

        # First upsample: stride=4, kernel=4 (7 -> 28)
        self.conv3d_transpose1 = conv_trans(hidden_size, hidden_size // 2, kernel_size=4, stride=4)
        # Second upsample: stride=4, kernel=4 (28 -> 112)
        self.conv3d_transpose2 = conv_trans(hidden_size // 2, hidden_size // 4, kernel_size=4, stride=4)
        # Third upsample: stride=2, kernel=2 (112 -> 224)
        self.conv3d_transpose3 = conv_trans(hidden_size // 4, deconv_chns, kernel_size=2, stride=2)
        # Final conv to output channels
        self.conv3d_out = Conv[Conv.CONV, spatial_dims](deconv_chns, out_channels, kernel_size=1)


    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """

        #x: [B, 1, 224, 224, 320]

        hidden_states_out = []

        # Initial conv and maxpool
        x = self.encoder.conv1(x)
        #x: [B, 64, 112, 112, 160]
        x = self.encoder.bn1(x)
        #x: [B, 64, 112, 112, 160]
        x = self.encoder.act(x)
        #x: [B, 64, 112, 112, 160]
        if not self.encoder.no_max_pool:
            x = self.encoder.maxpool(x)
            #x: [B, 64, 56, 56, 80]

        # Pass through ResNet layers and store intermediate outputs
        x = self.encoder.layer1(x)
        #x: [B, 64, 56, 56, 80]
        hidden_states_out.append(x)
        x = self.encoder.layer2(x)
        #x: [B, 128, 28, 28, 40]
        hidden_states_out.append(x)
        x = self.encoder.layer3(x)
        #x: [B, 256, 14, 14, 20]
        hidden_states_out.append(x)
        x = self.encoder.layer4(x)
        #x: [B, 512, 7, 7, 10]
        hidden_states_out.append(x)

        #x shape is [B, C, D', H', W'] with C = 512

        # Decode
        #x: [B, 512, 7, 7, 10]
        x = self.conv3d_transpose1(x)
        #x: [B, 256, 28, 28, 40]
        x = self.conv3d_transpose2(x)
        #x: [B, 128, 112, 112, 160]
        x = self.conv3d_transpose3(x)
        #x: [B, 64, 224, 224, 320]
        x = self.conv3d_out(x)
        #x: [B, 1, 224, 224, 320]
        return x, hidden_states_out