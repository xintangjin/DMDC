# import os
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
from timm.models.layers import DropPath, to_2tuple

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class MsakNet(nn.Module):
    def __init__(self, mask_base, in_c=3, out_c=1, nf=16):
        super(MsakNet, self).__init__()
        """mask_base = [w, h]"""
        self.mask_base = mask_base
        self.Phi = Parameter(torch.zeros(256, 256), requires_grad=True)

        self.block0 = Conv2dBlock(in_c, nf, 7, 2, padding=3, norm='in', pad_type='reflect')
        self.block1 = Conv2dBlock(nf, nf*2, 3, 2, padding=1, norm='in', pad_type='reflect')
        self.block2 = Conv2dBlock(nf*2, nf*4, 3, 2, padding=1, norm='in', pad_type='reflect')

        self.block3 = Conv2dBlock(nf * 6, nf*2, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block4 = Conv2dBlock(nf * 3, nf, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.block5 = Conv2dBlock(nf + in_c, nf // 3, 3, 1, padding=1, norm='in', pad_type='reflect')
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')

        self.fea = Conv2dBlock(nf // 3, out_c, 3, 1, padding=1, activation='tanh', pad_type='reflect')
        self.out = nn.Conv2d(2, 1, 1, 1, bias=False)

        self.activation = nn.Sigmoid()

    def forward(self, RGB):
        """
        :param RGB: [b, 3, w, h]
        :return: [b, w, h]
        """
        b = RGB.shape[0]
        mask_base = torch.repeat_interleave(self.mask_base.unsqueeze(0), b, dim=0)
        mask_base = mask_base.unsqueeze(1)
        conv1 = self.block0(RGB)
        conv2 = self.block1(conv1)
        conv3 = self.block2(conv2)

        conv4 = self.block3(torch.cat([self.up3(conv3), conv2], 1))
        conv5 = self.block4(torch.cat([self.up4(conv4), conv1], 1))
        conv6 = self.block5(torch.cat([self.up5(conv5), RGB], 1))
        fea = self.fea(conv6)
        out = self.out(torch.cat([fea, mask_base], dim=1))

        out = out.squeeze(1)
        return self.activation(out)

# mask_base = torch.randn(256, 256).cuda()
# input = torch.randn(2,3, 256, 256).cuda()
# a = MsakNet(mask_base).cuda()
# b = a(input)
# print(b.shape)