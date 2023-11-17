"""
This File has been abandoned!!!
Please goto UNet
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from config.extern_var import settings
from config.settings import *
from torchvision import transforms
import math


class ParallelCode_UNetBlock(nn.Module):
    """该部分为Unet的平行变化过程模块"""

    def __init__(self, in_channels, mid_channels, out_channels, padding=True):
        super(ParallelCode_UNetBlock, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels,
                               kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels,
                               kernel_size=3, padding=padding)
        self.batch_norm1 = nn.BatchNorm2d(self.mid_channels)
        self.batch_norm2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x, inplace=True)
        return x


class CustomUNet(nn.Module):
    """该Unet模型的期待输入为[N, 3, 432, 432], 输出为[N, 3, 48, 48]"""

    def __init__(self, in_channels=settings[Config_Item.UNet_train_thickness],
                 out_channels=settings[Config_Item.UNet_train_thickness], padding=True):
        super(CustomUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.total_layer = 4
        self.size_x = [settings[Config_Item.UNet_train_input_final_size] * 2]
        self.down_code = [ 0 for _ in range(0, self.total_layer + 1)]
        self.max_pool = [ 0 for _ in range(0, self.total_layer + 1)]

        ## 输入的批量归一化
        self.input_bn = nn.BatchNorm2d(self.in_channels)

        ## 编码器部分

        self.down_code1 = ParallelCode_UNetBlock(in_channels=self.in_channels, mid_channels=64, out_channels=64,
                                                 padding=padding)
        self.size_x1 -= 4
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.size_x2 = self.size_x1 // 2
        self.down_code2 = ParallelCode_UNetBlock(in_channels=64, mid_channels=128, out_channels=128)
        self.size_x2 -= 4
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.size_x3 = self.size_x2 // 2
        self.down_code3 = ParallelCode_UNetBlock(in_channels=128, mid_channels=256, out_channels=256)
        self.size_x3 -= 4
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.size_x4 = self.size_x3 // 2
        self.down_code4 = ParallelCode_UNetBlock(in_channels=256, mid_channels=512, out_channels=512)
        self.size_x4 -= 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        ## 桥接部分
        self.size_bridge = self.size_x4 // 2
        self.brige_code = ParallelCode_UNetBlock(in_channels=512, mid_channels=1024, out_channels=1024)
        self.size_bridge -= 4

        ## 解码器部分
        self.up_sample4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.size_y4 = self.size_bridge * 2
        self.transform4 = transforms.CenterCrop(self.size_y4)
        self.up_code4 = ParallelCode_UNetBlock(in_channels=1024, mid_channels=512, out_channels=512)
        self.size_y4 -= 4

        self.up_sample3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.size_y3 = self.size_y4 * 2
        self.transform3 = transforms.CenterCrop(self.size_y3)
        self.up_code3 = ParallelCode_UNetBlock(in_channels=512, mid_channels=256, out_channels=256)
        self.size_y3 -= 4

        ## 最终输出部分
        self.final_code = ParallelCode_UNetBlock(in_channels=256, mid_channels=128, out_channels=64)
        self.size_y3 -= 4
        self.end_code = nn.Conv2d(64, self.out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """权重初始化函数"""
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x1):

        ## 编码器部分
        x1 = self.down_code1(x1)
        x2 = self.maxpool1(x1)
        x2 = self.down_code2(x2)
        x3 = self.maxpool2(x2)
        x3 = self.down_code3(x3)
        x4 = self.maxpool3(x3)
        x4 = self.down_code4(x4)
        bridge = self.maxpool4(x4)

        ## 桥接部分
        bridge = self.brige_code(bridge)

        ## 解码器部分
        y4 = self.up_sample4(bridge)
        x4 = self.transform4(x4)
        y4 = torch.cat([x4, y4], dim=1)
        y4 = self.up_code4(y4)

        y3 = self.up_sample3(y4)
        x3 = self.transform3(x3)
        y3 = torch.cat([x3, y3], dim=1)
        y3 = self.up_code3(y3)

        output = self.final_code(y3)
        output = self.end_code(output)
        output = F.sigmoid(output)

        return output  # 返回经过softmax处理后的结果。


if __name__ == "__main__":
    model_tmp = CustomUNet()
    print(model_tmp)
