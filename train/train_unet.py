import os.path
import time
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from config.extern_var import settings
from config.settings import *
from data_coding.data_cache import *
from model.unet_wrapper import UNet_Wrapper as UNet
from utils.timer import *
import utils.abort as abort
from data_coding.data_transport_from_ct import *
from train.loss_set import Batch_Dice_Loss
import torchvision.transforms.functional as F


class Train_UNet:
    """UNet的训练器封装类"""

    def __init__(self):
        """训练器初始化"""
        torch.manual_seed(int(time.time()))

        self.device = torch.device(settings[Config_Item.device])
        self.cpu = torch.device("cpu")

        if settings[Config_Item.UNet_loss_type] == Loss_Type.Dice_Loss.name:
            self.criterion = Batch_Dice_Loss()
        else:
            raise abort.TrainAbort("Wrong Loss Type!!!")

        self.model = UNet().to(self.device)
        if settings[Config_Item.UNet_train_from_exist] is True:
            saved_state_dict = torch.load(settings[Config_Item.UNet_train_load_path], map_location=self.device)
            self.model.load_state_dict(saved_state_dict)

        if settings[Config_Item.UNet_optimizer_type] == Optimizer_Type.SGD.name:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=settings[Config_Item.UNet_learning_rate],
                                       momentum=settings[Config_Item.UNet_optimizer_para]['momentum'],
                                       dampening=settings[Config_Item.UNet_optimizer_para]['dampening'],
                                       weight_decay=settings[Config_Item.UNet_optimizer_para]['weight_decay'])
        elif settings[Config_Item.UNet_optimizer_type] == Optimizer_Type.Adam.name:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=settings['learning_rate'], betas=settings['optimizer_parameter']['betas'])
        else:
            raise abort.TrainAbort("Wrong Optimizer!!!")

        self.total_epochs = settings[Config_Item.UNet_total_epochs]

        self.train_dataset = UNet_Dataset(mode=True)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=settings[Config_Item.UNet_train_batch_size], shuffle=True)
        self.eval_dataset = UNet_Dataset(mode=False)
        self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=settings[Config_Item.UNet_train_batch_size], shuffle=True)

        now_date = datetime.now()
        formatted_date = now_date.strftime("%Y-%m-%d-%H-%M-%S")
        self.model_save_path = os.path.join(settings[Config_Item.UNet_save_path], formatted_date)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        else:
            raise abort.TrainAbort("Model Save Dir EXIST!!!")

        try:
            shutil.copy2(settings[Config_Item.config_json_path], self.model_save_path)
        except FileNotFoundError:
            raise abort.TrainAbort("The config.json doesn't EXIST!!!")

        self.best_eval_score = None
        self.best_model_path = None

        ## 配置tensorboard监控
        self.monitor_path = os.path.join(self.model_save_path, settings[Config_Item.monitor_dir_name])
        os.makedirs(self.monitor_path)
        self.monitor = SummaryWriter(log_dir=self.monitor_path)
        ## 可视化结构模型
        self.monitor.add_graph(self.model,
                               Get_CT_Cache(0, dataset_cache_type.eval_UNet_label).unsqueeze(0).to(self.device))

    def train_all_epochs(self):
        """全序数训练器"""

        Flush_CT_Data_To_Mem(
            [dataset_cache_type.train_UNet_input, dataset_cache_type.train_UNet_label,
             dataset_cache_type.eval_UNet_input, dataset_cache_type.eval_UNet_label])
        counter_t = DynamicCounter(self.total_epochs, "Training_Progress", 1)
        for epoch in range(self.total_epochs):
            self.train_one_epoch(epoch)
            counter_t.increment()

        self.monitor.close()

    def train_one_epoch(self, epoch):
        """一个训练循环"""

        ## 训练模式
        total_train_loss = 0
        self.model.train()
        for inputs, labels in self.train_dataloader:
            ## 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            total_train_loss += loss

            ## 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        total_train_loss /= self.train_dataset.length

        ## 评估模式
        total_eval_loss = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.eval_dataloader:
                ## 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_eval_loss += loss
                # counter_t.increment()

        total_eval_loss /= self.eval_dataset.length
        ## 打印训练信息
        print("Epoch [{}/{}], Train Loss: {:.4f}, Eval Loss: {:.4f}".format(epoch + 1, self.total_epochs,
                                                                            total_train_loss, total_eval_loss))

        ## 保存模型参数
        if self.best_eval_score is None or self.best_eval_score > total_eval_loss:
            save_name = \
                "UNet_epoch_{}_eval_loss_{:.4f}_train_loss_{:.4f}.pth".format(
                    epoch + 1, total_eval_loss, total_train_loss)
            save_path = os.path.join(self.model_save_path, save_name)
            torch.save(self.model.state_dict(), save_path)
            self.best_eval_score = total_eval_loss
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            self.best_model_path = save_path

        ## 添加监视
        self.monitor.add_scalar('Train loss', total_train_loss, global_step=epoch)
        self.monitor.add_scalar('Eval  loss', total_eval_loss, global_step=epoch)
        self.monitor.add_image('Input', inputs.to(self.cpu), global_step=epoch, dataformats="NCHW")
        self.monitor.add_image('Label', labels.to(self.cpu), global_step=epoch, dataformats="NCHW")


class UNet_Dataset(Dataset):
    def __init__(self, mode):
        """初始化，mode为True时表示训练数据，为False时表示评估数据"""
        super(UNet_Dataset, self).__init__()
        if mode is True:
            self.input_cache_type = dataset_cache_type.train_UNet_input
            self.label_cache_type = dataset_cache_type.train_UNet_label
            self.length = settings[Config_Item.train_UNet_dataset_num]
        elif mode is False:
            self.input_cache_type = dataset_cache_type.eval_UNet_input
            self.label_cache_type = dataset_cache_type.eval_UNet_label
            self.length = settings[Config_Item.eval_UNet_dataset_num]
        else:
            raise abort.TrainAbort("Wrong UNet Dataset Mode (Not in True or False)!!!")
        self.device = torch.device(settings[Config_Item.device])

        self.transform_list = []
        if mode is True:
            if settings[Config_Item.UNet_data_RandomHorizontalFlip] is True:
                self.transform_list.append(transforms.RandomHorizontalFlip())
            if settings[Config_Item.UNet_data_RandomVerticalFlip] is True:
                self.transform_list.append(transforms.RandomVerticalFlip())
            if settings[Config_Item.UNet_data_RandomRotation] is True:
                self.transform_list.append(transforms.RandomRotation(
                    degrees=settings[Config_Item.UNet_data_RandomRotation_para]["degrees"]))
        self.transform_list.append(transforms.RandomCrop(settings[Config_Item.UNet_train_input_final_size]))
        self.transform = transforms.Compose(self.transform_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ct_input = Get_CT_Cache(index, self.input_cache_type).to(self.device).to(self.device)
        ct_label = Get_CT_Cache(index, self.label_cache_type).to(self.device).to(self.device)
        ct_merged = torch.cat((ct_input, ct_label), dim=0)
        ct_merged = self.transform(ct_merged)
        ct_input, ct_label = torch.split(ct_merged, settings[Config_Item.UNet_train_thickness], dim=0)
        ct_input, ct_label = ct_input.contiguous(), ct_label.contiguous()

        # CT_Transform.show_one_ct_tensor(ct_input, settings[Config_Item.UNet_train_thickness_half], (-1, 1))
        # CT_Transform.show_one_ct_tensor(ct_label, settings[Config_Item.UNet_train_thickness_half], (0, 1))
        # print()

        return ct_input, ct_label



