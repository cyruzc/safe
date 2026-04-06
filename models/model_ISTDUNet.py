"""
ISTDU-Net: Improved Small Target Detection U-Net
Adapted from BasicIRSTD: https://github.com/XinyiYing/BasicIRSTD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== External Attention Module ==============
class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''

    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)

        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=-1)

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        x = self.linear_1(attn)

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        return x


# ============== Split-Attention Conv2d ==============
class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d"""

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob

        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                              groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)

        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
        else:
            splited = [x]
        gap = sum(splited)

        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


def _pair(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


# ============== ResNet Components ==============
class Bottleneck(nn.Module):
    """ResNet Bottleneck"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            nn.init.zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetCt(nn.Module):
    """ResNet for ISTDU-Net"""

    def __init__(self, block, layers, radix=2, groups=4, bottleneck_width=16,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=8, avg_down=True,
                 rectified_conv=False, rectify_avg=False,
                 avd=True, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d,
                 inp_num=1, layer_parms=[16, 32, 64, 128]):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNetCt, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg

        self.layer1 = self._make_layer(block, layer_parms[0], layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, layer_parms[1], layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, layer_parms[2], layers[2], stride=2,
                                       norm_layer=norm_layer,
                                       dropblock_prob=dropblock_prob)
        self.layer4 = self._make_layer(block, layer_parms[3], layers[3], stride=2,
                                       norm_layer=norm_layer,
                                       dropblock_prob=dropblock_prob)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                ceil_mode=True, count_include_pad=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            radix=self.radix, cardinality=self.cardinality,
                            bottleneck_width=self.bottleneck_width,
                            avd=self.avd, avd_first=self.avd_first,
                            dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                            rectify_avg=self.rectify_avg,
                            norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                            last_gamma=self.last_gamma))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


# ============== ISTDU-Net Components ==============
class Down(nn.Module):
    def __init__(self,
                 inp_num=1,
                 layers=[1, 2, 4, 8],
                 channels=[8, 16, 32, 64],
                 bottleneck_width=16,
                 stem_width=8,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 **kwargs):
        super(Down, self).__init__()

        stemWidth = int(8)
        self.stem = nn.Sequential(
            normLayer(1, affine=False),
            nn.Conv2d(1, stemWidth * 2, kernel_size=3, stride=1, padding=1, bias=False),
            normLayer(stemWidth * 2),
            activate()
        )
        self.down = ResNetCt(Bottleneck, layers, inp_num=inp_num,
                             radix=2, groups=4, bottleneck_width=bottleneck_width,
                             deep_stem=True, stem_width=stem_width, avg_down=True,
                             avd=True, avd_first=False, layer_parms=channels, **kwargs)

    def forward(self, x):
        x = self.stem(x)
        x = self.down(x)
        return x


class UPCt(nn.Module):
    def __init__(self, channels=[],
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU
                 ):
        super(UPCt, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(channels[0],
                      channels[1],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[1]),
            activate()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[1],
                      channels[2],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[2]),
            activate()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[2],
                      channels[3],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[3]),
            activate()
        )

    def forward(self, x):
        x1, x2, x3, x4 = x
        out = self.up1(x4)
        out = x3 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.up2(out)
        out = x2 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.up3(out)
        out = x1 + F.interpolate(out, scale_factor=2, mode='bilinear')
        return out


class Head(nn.Module):
    def __init__(self, inpChannel, oupChannel,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 ):
        super(Head, self).__init__()
        interChannel = inpChannel // 4
        self.head = nn.Sequential(
            nn.Conv2d(inpChannel, interChannel,
                      kernel_size=3, padding=1,
                      bias=False),
            normLayer(interChannel),
            activate(),
            nn.Conv2d(interChannel, oupChannel,
                      kernel_size=1, padding=0,
                      bias=True)
        )

    def forward(self, x):
        return self.head(x)


class EDN(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        super(EDN, self).__init__()
        self.X1 = External_attention(channels[0])
        self.X2 = External_attention(channels[1])
        self.X3 = External_attention(channels[2])
        self.X4 = External_attention(channels[3])

    def forward(self, x):
        x1, x2, x3, x4 = x
        x1 = self.X1(x1)
        x2 = self.X2(x2)
        x3 = self.X3(x3)
        x4 = self.X4(x4)
        return [x1, x2, x3, x4]


# ============== ISTDU-Net Main Model ==============
class ISTDUNet(nn.Module):
    """
    ISTDU-Net: Improved Small Target Detection U-Net

    Args:
        in_channels (int): Number of input channels (default: 1 for infrared)
        out_channels (int): Number of output channels (default: 1)
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(ISTDUNet, self).__init__()

        self.down = Down(channels=[16, 32, 64, 128])
        self.up = UPCt(channels=[512, 256, 128, 64])

        self.headDet = Head(inpChannel=64, oupChannel=1)
        self.headSeg = Head(inpChannel=64, oupChannel=1)

        self.DN = EDN(channels=[64, 128, 256, 512])

    def forward(self, x):
        x = self.down(x)
        x = self.DN(x)
        x = self.up(x)

        det_out = torch.sigmoid(self.headDet(x))
        seg_out = torch.sigmoid(self.headSeg(x))

        # Return only detection output for compatibility
        return det_out


# Alias for consistency
ISTDU_Net = ISTDUNet


if __name__ == '__main__':
    # Test the model
    x = torch.rand((2, 1, 256, 256))
    model = ISTDUNet()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ISTDU-Net Total Params: {total_params / 1e6:.2f}M")

    out = model(x)
    print(f"Output shape: {out.shape}")
