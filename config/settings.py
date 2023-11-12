from pathlib import Path
import json
from enum import Enum


class Optimizer_Type(Enum):
    SGD = 1
    Adam = 2


class Loss_Type(Enum):
    Dice_Loss = 1


class Settings:
    """设置类"""

    ## 设置项
    settings_item = {'UNet_total_epochs', 'UNet_save_path', 'unet_dataset_cache_path',
                     'unet_annoted_dataset_cache_path', 'unet_unannoted_dataset_cache_path','annoted_data_num',
                     'unannoted_data_num', 'train_batch_size', 'learning_rate', 'device', 'optimizer_type',
                     'optimizer_parameter', 'loss_type', 'model_save_path', 'UNet_save_path'}

    def __getitem__(self, key):
        """重载[]，直接返回项"""
        try:
            return self.config[key]
        except KeyError:
            print(f"\nThe setting '{key}' does not exist in the Settings!!!\n")
            exit()

    def write_config(self):
        """刷新配置"""
        self.config['UNet_total_epochs'] = 100
        self.config['UNet_save_path'] = "save/model/UNet"
        self.config['UNet_dataset_cache_path'] = "dataset/Cache/data_for_unet/"
        self.config['UNet_annoted_dataset_cache_path'] = self.config['UNet_dataset_cache_path'] + "annoted_slices/"
        self.config['UNet_unannoted_dataset_cache_path'] = self.config['UNet_dataset_cache_path'] + "unannoted_slices/"
        with open(self.config['UNet_annoted_dataset_cache_path'] + "note", "r") as f:
            self.config['annoted_data_num'] = int(f.readline())
        with open(self.config['UNet_unannoted_dataset_cache_path'] + "note", "r") as f:
            self.config['unannoted_data_num'] = int(f.readline())
        self.config['train_batch_size'] = 10
        self.config['learning_rate'] = 10
        self.config['device'] = "cuda"
        self.config['optimizer_type'] = Optimizer_Type.SGD.name
        if self.config['optimizer_type'] == Optimizer_Type.SGD.name:
            self.config['optimizer_parameter'] = {'momentum': 0.99, 'dampening': 0.5, 'weight_decay': 0.001}
        elif self.config['optimizer_type'] == Optimizer_Type.Adam.name:
            self.config['optimizer_parameter'] = {'betas': (0.9, 0.999)}
        self.config['loss_type'] = Loss_Type.Dice_Loss.name
        self.config['model_save_path'] = "save/model/"
        self.config['UNet_save_path'] = self.config['model_save_path'] + "UNet/"

        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def __init__(self, update=False):
        """初始化设置"""
        self.config_path = Path('config/config.json')
        self.config = {}

        if self.config_path.exists() and not update:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            self.check_config()
        else:
            self.write_config()

        print("\nConfigure:")
        for conf, value in self.config.items():
            print(f"{conf}: {value}")
        print("\n")

    def check_config(self):
        """检查配置是否齐全"""
        config_set = set(self.config.keys())
        if config_set == Settings.settings_item:
            pass
        else:
            self.write_config()


