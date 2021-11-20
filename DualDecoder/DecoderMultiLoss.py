import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

from Net.Utils import *



class Channel_Merge_Decoder_MultiLoss(nn.Module):

    def __init__(self, batchNorm=False, input_size=[240, 416], mode="train", frame_num=7):
        super(Channel_Merge_Decoder_MultiLoss, self).__init__()
        self.batchNorm = batchNorm
        self.mode = mode
        self.frame_num = frame_num

        self.deconv4 = deconv(1024, 512)
        self.deconv3 = deconv(1027, 256)
        self.deconv2 = deconv(515, 128)
        self.deconv1 = deconv(259, 64)

        self.predict_image4 = predict_image(1024)
        self.predict_image3 = predict_image(1027)
        self.predict_image2 = predict_image(515)
        self.predict_image1 = predict_image(259)

        self.upsampled_image4_to_3 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False)  # 8_16
        self.upsampled_image3_to_2 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False)  # 16-32
        self.upsampled_image2_to_1 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False)  # 32-64
        self.upsampled_image1_to_finally = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False)  # 64-128

        self.output1 = conv(self.batchNorm, 131, 64, kernel_size=3, stride=1)
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
        
        l4 = main_channel[4] + guided_channel[4]
        image_4 = self.predict_image4(l4)
        image_4_up = crop_like(self.upsampled_image4_to_3(image_4), main_channel[3])
        out_deconv3 = crop_like(self.deconv4(l4), main_channel[3])
        
        if self.mode=="debug":
            print("self.upsampled_image4_to_3(image_4) size: ", self.upsampled_image4_to_3(image_4).shape)
            print("main_channel[3] size: ", main_channel[3].shape)

        l3 = main_channel[3] + guided_channel[3]
        concat3 = torch.cat((l3, out_deconv3, image_4_up), dim=1)
        image_3 = self.predict_image3(concat3)
        image_3_up = crop_like(self.upsampled_image3_to_2(image_3), main_channel[2])
        out_deconv2 = crop_like(self.deconv3(concat3), main_channel[2])

        l2 = main_channel[2] + guided_channel[2]
        concat2 = torch.cat((l2, out_deconv2, image_3_up), dim=1)
        image_2 = self.predict_image2(concat2)
        image_2_up = crop_like(self.upsampled_image2_to_1(image_2), main_channel[1])
        out_deconv2 = crop_like(self.deconv2(concat2), main_channel[1])

        l1 = main_channel[1] + guided_channel[1]
        concat1 = torch.cat((l1, out_deconv2, image_2_up), dim=1)
        image_1 = self.predict_image1(concat1)
        image_1_up = crop_like(
            self.upsampled_image1_to_finally(image_1), main_channel[0])
        out_deconv1 = crop_like(self.deconv1(concat1), main_channel[0])

        l0 = main_channel[0] + guided_channel[0]
        concat0 = torch.cat([l0, out_deconv1, image_1_up], dim=1)
        image_out = self.output1(concat0)
        image_out2 = self.output2(image_out)
        image_finally = self.output3(image_out2)

        if self.mode=="train" or self.mode=="debug":
            return [image_4, image_3, image_2, image_1, image_finally]
        elif self.mode=="test":
            return image_finally
        else:
            print("Mode settings incorrect!")
