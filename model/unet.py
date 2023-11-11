import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomUNet(nn.Module):
    def __init__(self):
        super(CustomUNet, self).__init__()

        # 编码器部分
        self.conv_down1 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 桥接部分
        self.conv_bridge = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 解码器部分
        self.upsample = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 输出层
        self.conv_output = nn.Conv2d(64, 64, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        x1 = self.conv_down1(x)
        x1_pool = self.maxpool1(x1)

        x2 = self.conv_down2(x1_pool)
        x2_pool = self.maxpool2(x2)

        # 桥接部分
        x_bridge = self.conv_bridge(x2_pool)

        # 解码器部分
        x_up = self.upsample(x_bridge)
        x_up = torch.cat([x_up, x1_pool], dim=1)  # 特征图拼接
        x_up = self.conv_up(x_up)

        # 输出层，并添加softmax激活函数，这里假设需要输出的是多分类概率图，所以使用softmax。
        output = self.conv_output(x_up)  # 得到[5, 64, 64]的输出，未经softmax处理。
        output = F.softmax(output, dim=1)  # 在通道维度上添加softmax激活函数。
        return output  # 返回经过softmax处理后的结果。




if __name__ == "__main__":
    model_tmp = CustomUNet()
    print(model_tmp)
