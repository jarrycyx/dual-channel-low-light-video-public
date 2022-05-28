import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

from Net.Utils import *


class Guided_Fusion(nn.Module):

    def __init__(self, channel=128):
        super(Guided_Fusion, self).__init__()

        self.conv1 = conv(False, channel*2, channel, kernel_size=3, stride=1)
        self.conv2 = conv(False, channel, channel, kernel_size=3, stride=1)
        self.conv3 = conv(False, channel, channel, kernel_size=3, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data, a=0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, main_channel, guided_channel):
        data = torch.concat([main_channel, guided_channel], dim=1)
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = data + main_channel # residual connection
        return data



class Channel_Merge_Decoder_Fusion(nn.Module):

    def __init__(self, batchNorm=False, input_size=[240, 416], mode="train", frame_num=7):
        super(Channel_Merge_Decoder_Fusion, self).__init__()
        self.batchNorm = batchNorm
        self.mode = mode
        self.frame_num = frame_num

        self.deconv4 = deconv(1024, 512)
        self.deconv3 = deconv(1024, 256)
        self.deconv2 = deconv(512, 128)
        self.deconv1 = deconv(256, 64)
        
        self.gf4 = Guided_Fusion(channel=1024)
        self.gf3 = Guided_Fusion(channel=512)
        self.gf2 = Guided_Fusion(channel=256)
        self.gf1 = Guided_Fusion(channel=128)
        self.gf0 = Guided_Fusion(channel=64)

        self.upsampled_image4_to_3 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False)  # 8_16
        self.upsampled_image3_to_2 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False)  # 16-32
        self.upsampled_image2_to_1 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False)  # 32-64
        self.upsampled_image1_to_finally = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False)  # 64-128

        self.output1 = conv(self.batchNorm, 128, 64, kernel_size=3, stride=1)
        self.output2 = conv(self.batchNorm, 64, 64, kernel_size=3, stride=1)
        self.output3 = conv_no_lrelu(
            self.batchNorm, 64, 3, kernel_size=3, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data, a=0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, main_channel, guided_channel):
        
        l4 = self.gf4(main_channel[4], guided_channel[4])
        out_deconv4 = crop_like(self.deconv4(l4), main_channel[3])
        
        if self.mode=="debug":
            print("main_channel[3] size: ", main_channel[3].shape)

        l3 = self.gf3(main_channel[3], guided_channel[3])
        concat3 = torch.cat((l3, out_deconv4), dim=1)
        out_deconv3 = crop_like(self.deconv3(concat3), main_channel[2])

        l2 = self.gf2(main_channel[2], guided_channel[2])
        concat2 = torch.cat((l2, out_deconv3), dim=1)
        out_deconv2 = crop_like(self.deconv2(concat2), main_channel[1])

        l1 = self.gf1(main_channel[1], guided_channel[1])
        concat1 = torch.cat((l1, out_deconv2), dim=1)
        out_deconv1 = crop_like(self.deconv1(concat1), main_channel[0])

        l0 = self.gf0(main_channel[0], guided_channel[0])
        concat0 = torch.cat([l0, out_deconv1], dim=1)
        image_out = self.output1(concat0)
        image_out2 = self.output2(image_out)
        image_finally = self.output3(image_out2)
        
        if self.mode=="train" or self.mode=="debug" or self.mode=="test":
            return image_finally
        else:
            print("Mode settings incorrect!")

