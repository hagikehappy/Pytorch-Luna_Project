from pathlib import Path
import json
from enum import Enum
import enum
import os
import utils.abort as abort


class Optimizer_Type(Enum):
    SGD = 1
    Adam = 2


class Loss_Type(Enum):
    Dice_Loss = 1


class Config_Item(Enum):
    """存储了所有的config项"""

    ## Dataset: 1 - 10


    ## Model Save Path: 11 - 20
    model_save_path = 11
    UNet_save_path = 12

    ## Dataset Cache Path: 21 - 40
    dataset_cache_path = 21
    ## Train
    train_dataset_cache_path = 22
    train_UNet_dataset_cache_path = 23
    train_UNet_input_dataset_cache_path = 24
    train_UNet_label_dataset_cache_path = 25
    train_type_dataset_cache_path = 26
    train_type_annotated_dataset_cache_path = 27
    train_type_unannotated_dataset_cache_path = 28
    ## Eval
    eval_dataset_cache_path = 29
    eval_UNet_dataset_cache_path = 30
    eval_UNet_input_dataset_cache_path = 31
    eval_UNet_label_dataset_cache_path = 32
    eval_type_dataset_cache_path = 33
    eval_type_annotated_dataset_cache_path = 34
    eval_type_unannotated_dataset_cache_path = 35
    ## Predict
    predict_dataset_cache_path = 36
    predict_raw_dataset_cache_path = 36
    predict_raw_annotated_dataset_cache_path = 37
    predict_raw_unannotated_dataset_cache_path = 38
    predict_UNet_dataset_cache_path = 39
    predict_UNet_annotated_dataset_cache_path = 40
    predict_UNet_unannotated_dataset_cache_path = 41

    ## Dataset Parameter: 51 - 60
    train_UNet_dataset_num = 51
    train_type_annotated_dataset_num = 52
    train_type_unannotated_dataset_num = 53
    eval_UNet_dataset_num = 54
    eval_type_annotated_dataset_num = 55
    eval_type_unannotated_dataset_num = 56
    predict_annotated_dataset_num = 57
    predict_unannotated_dataset_num = 58
    cache_maxsize = 59

    ## Global Train Configure: 61 - 70
    device = 61

    ## UNet Train Configure: 71 -
    UNet_total_epochs = 71
    UNet_train_batch_size = 72
    UNet_learning_rate = 73
    UNet_optimizer_type = 75
    UNet_optimizer_parameter = 76
    UNet_loss_type = 77
    UNet_loss_parameter = 78


class Settings:
    """设置类"""

    def write_config(self):
        """刷新配置"""

        ## Reserved

        ## Model Save Path
        self.config_dict[Config_Item.model_save_path.name] = "save/model/"
        self.config_dict[Config_Item.UNet_save_path.name] = \
            os.path.join(self.config_dict[Config_Item.model_save_path.name], "UNet/")

        ## Dataset Cache Path
        self.config_dict[Config_Item.dataset_cache_path.name] = "dataset/Cache/"
        ## Train
        self.config_dict[Config_Item.train_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.dataset_cache_path.name], "train/")
        self.config_dict[Config_Item.train_UNet_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.train_dataset_cache_path.name], "UNet/")
        self.config_dict[Config_Item.train_UNet_input_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.train_UNet_dataset_cache_path.name], "input/")
        self.config_dict[Config_Item.train_UNet_label_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.train_UNet_dataset_cache_path.name], "label/")
        self.config_dict[Config_Item.train_type_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.train_dataset_cache_path.name], "type/")
        self.config_dict[Config_Item.train_type_annotated_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.train_type_dataset_cache_path.name], "annotated/")
        self.config_dict[Config_Item.train_type_unannotated_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.train_type_dataset_cache_path.name], "unannotated/")
        ## Eval
        self.config_dict[Config_Item.eval_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.dataset_cache_path.name], "eval/")
        self.config_dict[Config_Item.eval_UNet_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.eval_dataset_cache_path.name], "UNet/")
        self.config_dict[Config_Item.eval_UNet_input_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.eval_UNet_dataset_cache_path.name], "input/")
        self.config_dict[Config_Item.eval_UNet_label_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.eval_UNet_dataset_cache_path.name], "label/")
        self.config_dict[Config_Item.eval_type_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.eval_dataset_cache_path.name], "type/")
        self.config_dict[Config_Item.eval_type_annotated_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.eval_type_dataset_cache_path.name], "annotated/")
        self.config_dict[Config_Item.eval_type_unannotated_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.eval_type_dataset_cache_path.name], "unannotated/")
        ## Predict
        self.config_dict[Config_Item.predict_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.dataset_cache_path.name], "predict/")
        self.config_dict[Config_Item.predict_raw_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.predict_dataset_cache_path.name], "raw/")
        self.config_dict[Config_Item.predict_raw_annotated_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.predict_raw_dataset_cache_path.name], "annotated/")
        self.config_dict[Config_Item.predict_raw_unannotated_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.predict_raw_dataset_cache_path.name], "unannotated/")
        self.config_dict[Config_Item.predict_UNet_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.predict_dataset_cache_path.name], "UNet/")
        self.config_dict[Config_Item.predict_UNet_annotated_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.predict_UNet_dataset_cache_path.name], "annotated/")
        self.config_dict[Config_Item.predict_UNet_unannotated_dataset_cache_path.name] = \
            os.path.join(self.config_dict[Config_Item.predict_UNet_dataset_cache_path.name], "unannotated/")

        ## Dataset Parameter
        self._read_note(Config_Item.train_UNet_dataset_cache_path.name,
                        Config_Item.train_UNet_dataset_num.name)
        self._read_note(Config_Item.train_type_annotated_dataset_cache_path.name,
                        Config_Item.train_type_annotated_dataset_num.name)
        self._read_note(Config_Item.train_type_unannotated_dataset_cache_path.name,
                        Config_Item.train_type_unannotated_dataset_num.name)
        self._read_note(Config_Item.eval_UNet_dataset_cache_path.name,
                        Config_Item.eval_UNet_dataset_num.name)
        self._read_note(Config_Item.eval_type_annotated_dataset_cache_path.name,
                        Config_Item.eval_type_annotated_dataset_num.name)
        self._read_note(Config_Item.eval_type_unannotated_dataset_cache_path.name,
                        Config_Item.eval_type_unannotated_dataset_num.name)
        self._read_note(Config_Item.predict_raw_annotated_dataset_cache_path.name,
                        Config_Item.predict_annotated_dataset_num.name)
        self._read_note(Config_Item.predict_raw_unannotated_dataset_cache_path.name,
                        Config_Item.predict_unannotated_dataset_num.name)
        self.config_dict[Config_Item.cache_maxsize.name] = self.total_cache_num

        ## Global Train Configure
        self.config_dict[Config_Item.device.name] = "cuda"

        ## UNet Train Configure
        self.config_dict[Config_Item.UNet_total_epochs.name] = 10
        self.config_dict[Config_Item.UNet_train_batch_size.name] = 10
        self.config_dict[Config_Item.UNet_learning_rate.name] = 0.001
        self.config_dict[Config_Item.UNet_optimizer_type.name] = Optimizer_Type.SGD.name
        if self.config_dict[Config_Item.UNet_optimizer_type.name] == Optimizer_Type.SGD.name:
            self.config_dict[Config_Item.UNet_optimizer_parameter.name] = \
                {'momentum': 0.99, 'dampening': 0.5, 'weight_decay': 0.001}
        elif self.config_dict[Config_Item.UNet_optimizer_type.name] == Optimizer_Type.Adam.name:
            self.config_dict[Config_Item.UNet_optimizer_parameter.name] = \
                {'betas': (0.9, 0.999)}
        else:
            raise abort.SettingsAbort(f"There is no such optimizer as {Config_Item.UNet_optimizer_type.name}!!!")
        self.config_dict[Config_Item.UNet_loss_type.name] = Loss_Type.Dice_Loss.name
        if self.config_dict[Config_Item.UNet_loss_type.name] == Loss_Type.Dice_Loss.name:
            self.config_dict[Config_Item.UNet_loss_parameter.name] = \
                {}
        else:
            raise abort.SettingsAbort(f"There is no such Loss as {Config_Item.UNet_loss_type.name}!!!")

        with open(self.config_path, "w") as f:
            json.dump(self.config_dict, f, indent=4)

    def _read_note(self, dataset_name, num_name):
        """读取dataset的数值"""
        try:
            with open(os.path.join(self.config_dict[dataset_name], "note"), "r") as f:
                self.config_dict[num_name] = int(f.readline())
        except FileNotFoundError:
            self.config_dict[num_name] = 0
        except ValueError:
            raise abort.SettingsAbort(f"Number Of {dataset_name} Cache ERROR!!!")
        self.total_cache_num += self.config_dict[num_name]

    def __init__(self, update=False):
        """初始化设置"""
        self.config_path = Path('config/config.json')
        self.config_dict = {}
        self.config_set = set()
        self.config_list = [0 for _ in range(100)]
        self.total_cache_num = 0

        if self.config_path.exists() and not update:
            with open(self.config_path, "r") as f:
                self.config_dict = json.load(f)
            self.check_config()
        else:
            self.write_config()
        for key, value in self.config_dict.items():
            self.config_set.add(key)
            self.config_list[Config_Item[key].value] = value
        print(type(self.config_list))

        print("\nConfigure:")
        for conf, value in self.config_dict.items():
            print(f"{conf}: {value}")
        print("\n")

    def __getitem__(self, key):
        """重载[]，直接返回项"""
        if type(key) == int:
            try:
                if self.config_list[key] == 0:
                    raise IndexError
                return self.config_list[key]
            except IndexError:
                raise abort.SettingsAbort(f"The setting '{key}' does not exist in the Settings!!!")
        elif type(key) == str:
            try:
                return self.config_dict[key]
            except KeyError:
                raise abort.SettingsAbort(f"The setting '{key}' does not exist in the Settings!!!")
        elif type(key) == Config_Item:
            try:
                return self.config_list[key.value]
            except AttributeError:
                raise abort.SettingsAbort(f"The setting '{key}' does not exist in the Settings!!!")
        else:
            raise abort.SettingsAbort(f"The setting '{key}' does not exist in the Settings!!!")

    def check_config(self):
        """检查配置是否齐全"""
        config_set = set(self.config_dict.keys())
        if config_set == self.config_list:
            pass
        else:
            self.write_config()


if __name__ == "__main__":
    print(type(Config_Item.predict_raw_annotated_dataset_cache_path) == Config_Item)
