import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import torchvision.models as models
from morpholayers import *
from unet_part import *


class ResNet50BottomS(nn.Module):
    def __init__(self, model_name, pretrained):
        super(ResNet50BottomS, self).__init__()
        if model_name == 'ResNet50':
          original_model = models.resnet50(pretrained=pretrained)
        self.features1 = nn.Sequential(*list(original_model.children())[:5])
        self.features21 = nn.Sequential(*list(original_model.children())[:6])
        self.features22 = nn.Sequential(*list(list(original_model.children())[6].children())[:5])
        self.features23 = nn.Sequential(*list(list(list(original_model.children())[6].children())[5].children())[:4])

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features23(self.features22(self.features21(x)))
        return x2,x1



class ASPP(nn.Module):
    def __init__(self, in_channel):
        super(ASPP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.conv3 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.conv4 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channel)
        self.conv5 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn5 = nn.BatchNorm2d(in_channel)
        self.conv6 = nn.Conv2d(in_channel * 5, in_channel, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn6 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        batch, _, h, w = x.size()

        if batch > 1:
            x1 = self.relu(self.bn1(self.conv1(self.pool(x))))
        else:
            x1 = self.relu(self.conv1(self.pool(x)))
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear')
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv5(x)))
        x5 = self.relu(self.bn5(self.conv5(x)))

        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.drop(self.relu(self.bn6(self.conv6(x))))

        return x

class MASPP(nn.Module):
    def __init__(self, in_channel):
        super(MASPP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        n = in_channel
        self.grad1 = Gradient2d(n, 1, soft_max=True,dilation = 1)
        self.grad2 = Gradient2d(n, 1, soft_max=True,dilation = 1)
        self.grad3 = Gradient2d(n, 3, soft_max=True,dilation = 6)
        self.grad4 = Gradient2d(n, 3, soft_max=True,dilation = 12)
        self.grad5 = Gradient2d(n, 3, soft_max=True,dilation = 18)

        self.conv = nn.Conv2d(in_channel, n, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(n)

        self.bn1 = nn.BatchNorm2d(n)
        self.bn2 = nn.BatchNorm2d(n)
        self.bn3 = nn.BatchNorm2d(n)
        self.bn4 = nn.BatchNorm2d(n)
        self.bn5 = nn.BatchNorm2d(n)

        self.conv6 = nn.Conv2d(n*5, n, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn6 = nn.BatchNorm2d(n)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        batch, _, h, w = x.size()
        x = self.relu(self.bn(self.conv(x)))

        if batch > 1:
            x1 = self.relu(self.bn1(self.grad1(self.pool(x))))
        else:
            x1 = self.relu(self.grad1(self.pool(x)))
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear')
        x2 = self.relu(self.bn2(self.grad2(x)))
        x3 = self.relu(self.bn3(self.grad3(x)))
        x4 = self.relu(self.bn4(self.grad4(x)))
        x5 = self.relu(self.bn5(self.grad5(x)))

        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.drop(self.relu(self.bn6(self.conv6(x))))

        return x   


class DeepLabV3PlusS(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, os=16):
        super(DeepLabV3PlusS, self).__init__()
        self.resnet = ResNet50BottomS('ResNet50',pretrained)
        self.aspp = ASPP(256)
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0)
        # self.sigmoid = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H= x.shape[-2]
        W = x.shape[-1]
        x, low_level_feature = self.resnet(x)
        # print(x.shape,low_level_feature.shape)
        x = self.aspp(x)

        low_level_feature = self.relu(self.bn1(self.conv1(low_level_feature)))
        x = F.interpolate(x, size=low_level_feature.size()[2:], mode='bilinear')
        x = torch.cat((x, low_level_feature), dim=1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, size=(H,W), mode='bilinear')
        x = self.sigmoid(self.conv4(x))

        return x




class MorphoGradNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, os=16):
        super(MorphoGradNet, self).__init__()
        self.resnet = ResNet50BottomS('ResNet50',pretrained)
        self.maspp = MASPP(256)
        # self.conv1 = nn.Conv2d(256, 48, kernel_size=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0)
        # self.sigmoid = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        # self.conv = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        # self.bn = nn.BatchNorm2d(256)
        self.maspp2 = MASPP(256)

    def forward(self, x):
        H= x.shape[-2]
        W = x.shape[-1]
        x, low_level_feature = self.resnet(x)
        # print(x.shape,low_level_feature.shape)
        # x = self.relu(self.bn(self.conv(x)))
        x = self.maspp(x)

        # low_level_feature = self.relu(self.bn1(self.conv1(low_level_feature)))
        low_level_feature = self.maspp2(low_level_feature)

        x = F.interpolate(x, size=low_level_feature.size()[2:], mode='bilinear')
        x = torch.cat((x, low_level_feature), dim=1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = F.interpolate(x, size=(H,W), mode='bilinear')

        x = self.sigmoid(self.conv4(x))

        return x

class myunet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(myunet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)



class MorphoGradUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(MorphoGradUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (myDoubleConv(n_channels, 16))
        # self.maspp1 = MASPP(16)
        self.down1 = (Down(16,32))
        # self.maspp2 = MASPP(32)
        self.down2 = (Down(32, 64))
        # self.maspp3 = MASPP(64)
        self.down3 = (Down(64, 128))
        self.maspp = MASPP(256)
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConv(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        # x1 = self.maspp1(x)
        x2 = self.down1(x1)
        # x2 = self.maspp2(x)
        x3 = self.down2(x2)
        # x3 = self.maspp3(x)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.maspp(x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)

