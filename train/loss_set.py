"""该文件记录了所有人工的损失函数"""


import torch.nn as nn
import torch
from config.extern_var import settings
from config.settings import Config_Item


def Dice_Loss(input_t, label_t, smooth):
    """骰子损失"""
    # 计算交集
    intersection = (input_t * label_t).sum()
    # 计算并集，通过取预测值和目标值的平均值来避免0除问题
    union = input_t.sum() + label_t.sum() + smooth
    # 计算骰子损失，并使用1减去骰子系数作为损失值
    dice_loss = 1 - (2.0 * intersection + smooth) / union
    return dice_loss


class Batch_Dice_Loss(nn.Module):
    """骰子损失"""

    def __init__(self, augment=settings[Config_Item.Dice_Loss_Rate]):
        super(Batch_Dice_Loss, self).__init__()
        self.smooth = settings[Config_Item.Dice_Loss_Smooth]
        self.augment = augment

    def forward(self, input_g, label_g):
        predict_loss = Dice_Loss(input_g, label_g, self.smooth)
        fn_loss = Dice_Loss(input_g * label_g, label_g * label_g, self.smooth)
        return predict_loss.mean() + self.augment * fn_loss.mean()
