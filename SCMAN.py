import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from DualDecoder.DecoderMultiLoss import Channel_Merge_Decoder_MultiLoss
from DualDecoder.DecoderAdd import Channel_Merge_Decoder_Add
from DualDecoder.DecoderConcat import Channel_Merge_Decoder_Concat
from Utils import *
from LSTM.BiConvLSTM import BiConvLSTM
from torch.nn.init import kaiming_normal_
import torch.nn as nn
import torch
import sys


class Channel_Encoder(nn.Module):

    def __init__(self, batchNorm=False, input_size=[240, 416], frame_num=7,
                 mode='train', lstm_layer=1):
        super(Channel_Encoder, self).__init__()
        self.batchNorm = batchNorm
        self.mode = mode
        self.lstm_layer = lstm_layer

        self.frame_num = frame_num
        pre_convs = []
        for i in range(self.frame_num):
            preconv = nn.ModuleList([conv(self.batchNorm, 3, 64,  kernel_size=3,  stride=1),
                                     conv(self.batchNorm, 64, 64, kernel_size=3,  stride=1)])
            pre_convs.append(preconv)

        self.pre_convs = nn.ModuleList(pre_convs)

        self.biconvlstm = BiConvLSTM(input_size=(
            input_size[0], input_size[1]), input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=self.lstm_layer)

        self.LSTM_out = conv(self.batchNorm, 128, 64,
                             kernel_size=1,  stride=1)

        self.conv1 = conv(self.batchNorm,   64, 128,
                          kernel_size=7,  stride=2)  # 64
        self.conv1_1 = conv(self.batchNorm,   128, 128)  # 128*128 ->64*64
        self.conv2 = conv(self.batchNorm,   128, 256,
                          kernel_size=3,  stride=2)  # 64 ->32
        self.conv2_1 = conv(self.batchNorm,   256, 256)  # 128*128 ->64*64
        self.conv3 = conv(self.batchNorm,   256, 512,
                          kernel_size=3,  stride=2)  # 32->16
        self.conv3_1 = conv(self.batchNorm,   512, 512)
        self.conv4 = conv(self.batchNorm,   512, 1024,
                          kernel_size=3,  stride=2)  # 16->8
        self.conv4_1 = conv(self.batchNorm,   1024, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data, a=0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, frames):

        CNN_seq = []
        half_num = (self.frame_num-1)//2

        for i in range(self.frame_num):
            pre_conv_this_frame = self.pre_convs[i][0](frames[:, i])
            pre_conv_1_this_frame = self.pre_convs[i][1](pre_conv_this_frame)
            CNN_seq.append(pre_conv_1_this_frame)

        CNN_seq_out = torch.stack(CNN_seq, dim=1)
        CNN_seq_feature_maps = self.biconvlstm(CNN_seq_out)

        CNN_concat_input = torch.cat(
            [CNN_seq_out[:, half_num, ...], CNN_seq_feature_maps[:, half_num, ...]], dim=1)

        LSTM_out = self.LSTM_out(CNN_concat_input)  # 128*128*64

        out_conv1 = self.conv1_1(self.conv1(LSTM_out))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        return [LSTM_out, out_conv1, out_conv2, out_conv3, out_conv4]



class Channel_Decoder(nn.Module):

    def __init__(self, batchNorm=False, input_size=[240, 416], mode="train", frame_num=7):
        super(Channel_Decoder, self).__init__()
        self.batchNorm = batchNorm
        self.mode = mode
        self.frame_num = frame_num

        self.deconv4 = deconv(1024, 512)
        self.deconv3 = deconv(1024, 256)
        self.deconv2 = deconv(512, 128)
        self.deconv1 = deconv(256, 64)

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

    def forward(self, main_channel):
        
        concat3 = main_channel[4]
        out_deconv3 = crop_like(self.deconv4(concat3), main_channel[3])
        
        if self.mode=="debug":
            print("main_channel[3] size: ", main_channel[3].shape)

        concat3 = torch.cat((main_channel[3], out_deconv3), dim=1)
        out_deconv2 = crop_like(self.deconv3(concat3), main_channel[2])

        concat2 = torch.cat((main_channel[2], out_deconv2), dim=1)
        out_deconv2 = crop_like(self.deconv2(concat2), main_channel[1])

        concat1 = torch.cat((main_channel[1], out_deconv2), dim=1)
        out_deconv1 = crop_like(self.deconv1(concat1), main_channel[0])

        concat0 = torch.cat([main_channel[0], out_deconv1], dim=1)
        image_out = self.output1(concat0)
        image_out2 = self.output2(image_out)
        image_finally = self.output3(image_out2)
        
        if self.mode=="train" or self.mode=="debug" or self.mode=="test":
            return image_finally
        else:
            print("Mode settings incorrect!")
            

class Single_Channel_Attention_Net(nn.Module):

    def __init__(self, batchNorm=False, input_size=[240, 416], mode="train",
                 frame_num=7, lstm_layer=1):
        super(Single_Channel_Attention_Net, self).__init__()
        self.batchNorm = batchNorm
        self.mode = mode
        self.frame_num = frame_num
        self.lstm_layer = lstm_layer

        self.encoder = Channel_Encoder(batchNorm=batchNorm, input_size=input_size,
                                                 frame_num=frame_num, lstm_layer=self.lstm_layer)

        self.decoder = Channel_Decoder(batchNorm=batchNorm, input_size=input_size,frame_num=frame_num, mode=mode)


    def forward(self, channel_a):
        codes_a = self.encoder(channel_a)

        output_a = self.decoder(codes_a)
        return output_a


if __name__ == "__main__":
    test_a = torch.zeros([1, 7, 3, 120, 240])
    test_b = torch.zeros([1, 7, 3, 120, 240])

    net = Single_Channel_Attention_Net(input_size=[120, 240], mode='debug')
