import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings("ignore")

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(Conv2d(in_channel, out_channel, 1))

        self.branch1 = nn.Sequential(Conv2d(in_channel, out_channel, 1),
                                     Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
                                     Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
                                     Conv2d(out_channel, out_channel, 3, padding=3, dilation=3))

        self.branch2 = nn.Sequential(Conv2d(in_channel, out_channel, 1),
                                     Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
                                     Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
                                     Conv2d(out_channel, out_channel, 3, padding=5, dilation=5))

        self.branch3 = nn.Sequential(Conv2d(in_channel, out_channel, 1),
                                     Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
                                     Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
                                     Conv2d(out_channel, out_channel, 3, padding=7, dilation=7))
        self.conv_cat = Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv_upsample1 = Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = Conv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = Conv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class DarkraNet(nn.Module):
    # Darknet53 based encoder Parnet based decoder
    def __init__(self, channel=32):
        super(DarkraNet, self).__init__()
        # ---- Darknet53 Backbone ----
        self.darknet = darknet53()
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(256, channel)
        self.rfb3_1 = RFB_modified(512, channel)
        self.rfb4_1 = RFB_modified(1024, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        #  ---- reverse attention branch 4 ----
        self.ra4_conv1 = Conv2d(1024, 256, kernel_size=1)
        self.ra4_conv2 = Conv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = Conv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = Conv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = Conv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = Conv2d(512, 64, kernel_size=1)
        self.ra3_conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = Conv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = Conv2d(256, 64, kernel_size=1)
        self.ra2_conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.darknet.conv1(x)                   # bs, 32, 352, 352
        x = self.darknet.conv2(x)                   # bs, 64, 176, 176
        x = self.darknet.residual_block1(x)         # bs, 64, 176, 176
        x = self.darknet.conv3(x)                   # bs, 128, 88, 88
        # ---- low-level features ----
        x1 = self.darknet.residual_block2(x)        # bs, 128, 88, 88

        x2 = self.darknet.conv4(x1)                 # bs, 256, 44, 44
        x2 = self.darknet.residual_block3(x2)       # bs, 256, 44, 44

        x3 = self.darknet.conv5(x2)                 # bs, 512, 22, 22
        x3 = self.darknet.residual_block4(x3)       # bs, 512, 22, 22

        x4 = self.darknet.conv6(x3)                 # bs, 1024, 11, 11
        x4 = self.darknet.residual_block5(x4)       # bs, 1024, 11, 11
        # x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rfb2_1(x2)        # bs, 32, 44, 44
        x3_rfb = self.rfb3_1(x3)        # bs, 32, 22, 22
        x4_rfb = self.rfb4_1(x4)        # bs, 32, 11, 11

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bicubic')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        '''
        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bicubic')
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bicubic')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        '''
        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(ra5_feat, scale_factor=0.5, mode='bicubic')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bicubic')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 256, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bicubic')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_3, lateral_map_2


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

def darknet53(num_classes=1000):

    model = Darknet53(DarkResidualBlock, num_classes)
    weights = torch.load('model_best.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(weights["state_dict"])
    return model # 1*num_classes

if __name__ == '__main__':

    model = DarkraNet().to(DEVICE)
    input_tensor = torch.randn(1, 3, 352, 352).to(DEVICE)
    out = model(input_tensor)
    print(out[0].shape) # bs*1*352*352
    print(out[1].shape)
    print(out[2].shape)
