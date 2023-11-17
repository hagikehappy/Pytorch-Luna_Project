"""用于封装unet的最终函数"""


from unet import UNet
import torch.nn as nn
import math
from config.extern_var import settings
from config.settings import Config_Item


class UNet_Wrapper(nn.Module):
    """对于UNet在此处的封装起器"""

    def __init__(self, in_channels=settings[Config_Item.UNet_train_thickness],
                 out_channels=settings[Config_Item.UNet_train_thickness],
                 depth=5, wf=6, padding=True, batch_norm=False, up_mode="upconv"):
        """初始化模型"""

        ## 初始化模型结构
        super(UNet_Wrapper, self).__init__()
        self.input_bn = nn.BatchNorm2d(in_channels)
        self.unet = UNet(in_channels=in_channels, n_classes=out_channels, depth=depth,
                         wf=wf, padding=padding, batch_norm=batch_norm, up_mode=up_mode)
        self.final_layer = nn.Sigmoid()

        ## 初始化模型权重
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
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        """前向传播"""

        x = self.input_bn(x)
        x = self.unet(x)
        x = self.final_layer(x)

        return x


