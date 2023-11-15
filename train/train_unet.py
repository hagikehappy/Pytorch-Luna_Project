import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config.extern_var import settings
from config.settings import *
from data_coding.data_cache import *
from model.custom_unet import CustomUNet as UNet
from tools.tool import *
from data_coding.data_transport_from_ct import *


class Dice_Loss(nn.Module):
    """骰子损失"""

    def __init__(self, smooth=1.):
        super(Dice_Loss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, target):
        # 将预测值和目标值进行flatten操作，并确保它们在同一设备上
        inputs = inputs.view(-1)
        target = target.view(-1)

        # 计算交集
        intersection = (inputs * target).sum()

        # 计算并集，通过取预测值和目标值的平均值来避免0除问题
        union = inputs.sum() + target.sum() + self.smooth

        # 计算骰子损失，并使用1减去骰子系数作为损失值
        dice_loss = 1 - (2. * intersection + self.smooth) / union

        return dice_loss


class UNet_Train_Dataset(Dataset):
    def __init__(self):
        """初始化"""
        self.length = settings['annotated_data_num']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ct_input = Get_CT_Annoted_INPUT(index)
        ct_label = Get_CT_Annoted_LABEL(index)

        return ct_input, ct_label


def train():
    """训练器"""
    ## 初始化模型、优化器和损失函数
    device = torch.device(settings['device'])
    model = UNet().to(device)
    if settings['optimizer_type'] == Optimizer_Type.SGD.name:
        optimizer = optim.SGD(model.parameters(),
                              lr=settings['learning_rate'], momentum=settings['optimizer_parameter']['momentum'],
                              dampening=settings['optimizer_parameter']['dampening'],
                              weight_decay=settings['optimizer_parameter']['weight_decay'])
    elif settings['optimizer_type'] == Optimizer_Type.Adam.name:
        optimizer = optim.Adam(model.parameters(),
                               lr=settings['learning_rate'], betas=settings['optimizer_parameter']['betas'])
    else:
        print("\nWrong Optimizer!!!\n")
        exit()
    if settings['loss_type'] == Loss_Type.Dice_Loss.name:
        criterion = Dice_Loss()
    else:
        print("\nWrong Loss Type!!!\n")
        exit()
    ## 加载数据集
    Flush_All_CT_Data_To_Mem()
    train_dataset = UNet_Train_Dataset()
    batch_size = settings['train_batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## 模型保存路径
    save_path = settings['UNet_save_path']
    save_file = save_path + "unet_model.pth"

    ## 训练循环
    num_epochs = settings['UNet_total_epochs']
    counter = DynamicCounter(num_epochs, "Training_Progress", 1)
    for epoch in range(num_epochs):
        train_one_epoch(epoch=epoch, n_epochs=num_epochs, length=train_dataset.length, device=device,
                        model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)
        counter.increment()

    ## 保存模型权重
    torch.save(model.state_dict(), save_file)
    print(f"Model saved to {save_file}")
    print("Training completed.")
    return model  # 你可以选择只返回训练后的模型，不包含损失值。损失值在这里仅用于示例。


def train_one_epoch(epoch, n_epochs, length, device, model, train_loader, criterion, optimizer):
    """一个训练循环"""
    model.train()
    mean_loss = 0
    counter = DynamicCounter(116, "One_Epoch", 1)
    for inputs, labels in train_loader:
        ## 前向传播
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        mean_loss += loss

        ## 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        counter.increment()

    mean_loss /= length
    ## 打印训练信息
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, n_epochs, mean_loss))
