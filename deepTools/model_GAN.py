#!/usr/bin/env python
# model_GAN.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from torch.nn.utils import spectral_norm

class LinformerAttention3D(nn.Module):
    def __init__(self, channels, reduce_ratio=4, r=64):
        super(LinformerAttention3D, self).__init__()
        self.channels = channels
        self.reduce_ratio = reduce_ratio
        self.r = r
        self.query = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.key   = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.value = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.scale = channels ** 0.5

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(B, C, N).permute(0, 2, 1)
        k = k.view(B, C, N).permute(0, 2, 1)
        v = v.view(B, C, N).permute(0, 2, 1)
        k_t = k.permute(0, 2, 1)
        v_t = v.permute(0, 2, 1)
        k_proj = self.k_proj(k_t)
        v_proj = self.v_proj(v_t)
        max_r = min(self.r, k_proj.size(2))
        k_proj = k_proj[:, :, :max_r]
        v_proj = v_proj[:, :, :max_r]
        q_div = q / self.scale
        attn_scores = torch.einsum('b n c, b c r -> b n r', q_div, k_proj)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('b n r, b c r -> b n c', attn_weights, v_proj)
        attn_output = attn_output.permute(0, 2, 1).contiguous().view(B, C, D, H, W)
        return x + attn_output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(g_channels, inter_channels, kernel_size=1),
            nn.InstanceNorm3d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(x_channels, inter_channels, kernel_size=1),
            nn.InstanceNorm3d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, reduce_ratio=4, r=64):
        super(UNetWithAttention, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(512, 1024))
        self.linear_attention = LinformerAttention3D(1024, reduce_ratio=reduce_ratio, r=r)
        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(512, 512, 256)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(256, 256, 128)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(64, 64, 32)
        self.up_conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv3d(64, out_channels, kernel_size=1)

    def downsample_conditionally(self, input_tensor, layer):
        D, H, W = input_tensor.shape[2:]
        if D <= 4 or H <= 4 or W <= 4:
            return layer[1](input_tensor)
        else:
            return layer(input_tensor)

    def forward(self, x):
        x1 = checkpoint.checkpoint(lambda inp: self.inc(inp), x)
        x2 = checkpoint.checkpoint(lambda inp: self.downsample_conditionally(inp, self.down1), x1)
        x3 = checkpoint.checkpoint(lambda inp: self.downsample_conditionally(inp, self.down2), x2)
        x4 = checkpoint.checkpoint(lambda inp: self.downsample_conditionally(inp, self.down3), x3)
        x5 = checkpoint.checkpoint(lambda inp: self.downsample_conditionally(inp, self.down4), x4)
        x5 = self.linear_attention(x5)
        d1 = self.up1(x5)
        d1 = self._match_size(d1, x4)
        x4_att = self.att1(g=d1, x=x4)
        d1 = torch.cat([d1, x4_att], dim=1)
        d1 = self.up_conv1(d1)
        d2 = self.up2(d1)
        d2 = self._match_size(d2, x3)
        x3_att = self.att2(g=d2, x=x3)
        d2 = torch.cat([d2, x3_att], dim=1)
        d2 = self.up_conv2(d2)
        d3 = self.up3(d2)
        d3 = self._match_size(d3, x2)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat([d3, x2_att], dim=1)
        d3 = self.up_conv3(d3)
        d4 = self.up4(d3)
        d4 = self._match_size(d4, x1)
        x1_att = self.att4(g=d4, x=x1)
        d4 = torch.cat([d4, x1_att], dim=1)
        d4 = self.up_conv4(d4)
        out = self.outc(d4)
        return out

    def _match_size(self, decoder_feature, encoder_feature):
        diff_D = encoder_feature.size(2) - decoder_feature.size(2)
        diff_H = encoder_feature.size(3) - decoder_feature.size(3)
        diff_W = encoder_feature.size(4) - decoder_feature.size(4)
        decoder_feature = F.pad(decoder_feature,
                                [diff_W // 2, diff_W - diff_W // 2,
                                 diff_H // 2, diff_H - diff_H // 2,
                                 diff_D // 2, diff_D - diff_D // 2])
        if decoder_feature.size() != encoder_feature.size():
            encoder_feature = self._crop_to_match(encoder_feature, decoder_feature.size())
        return decoder_feature

    def _crop_to_match(self, tensor, target_size):
        _, _, D, H, W = tensor.size()
        target_D, target_H, target_W = target_size[2], target_size[3], target_size[4]
        d_start = (D - target_D) // 2
        h_start = (H - target_H) // 2
        w_start = (W - target_W) // 2
        tensor = tensor[:, :, d_start:d_start + target_D,
                        h_start:h_start + target_H,
                        w_start:w_start + target_W]
        return tensor

class Discriminator3D(nn.Module):
    def __init__(self, in_channels=1, base_filters=64):
        super(Discriminator3D, self).__init__()
        # Apply spectral normalization to each convolution.
        self.conv1 = spectral_norm(nn.Conv3d(in_channels, base_filters, kernel_size=4, stride=2, padding=1))
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = spectral_norm(nn.Conv3d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1))
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=base_filters * 2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = spectral_norm(nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1))
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=base_filters * 4)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = spectral_norm(nn.Conv3d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1))
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=base_filters * 8)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_filters * 8, 1)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.gn2(self.conv2(x)))
        x = self.lrelu3(self.gn3(self.conv3(x)))
        x = self.lrelu4(self.gn4(self.conv4(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # Raw logits
