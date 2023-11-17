
import random, time
from config.extern_var import settings
import torch.nn as nn
import torch
import utils.abort as abort
from model.unet_wrapper import UNet_Wrapper as UNet
from data_coding.data_cache import *
from data_coding.data_transport_from_ct import CT_Transform


def UNet_Predict(index=None):
    """用于使用UNet来生成处理图片的结果"""
    if index is None:
        random.seed(int(time.time()))
        index = random.randint(0, settings[Config_Item.eval_UNet_dataset_num])
    elif type(index) != int:
        raise abort.PredictAbort("Wrong Index Input !!!")
    device = torch.device(settings[Config_Item.device])
    cpu = torch.device("cpu")
    unet_model = UNet()
    saved_state_dict = torch.load(settings[Config_Item.UNet_best_path])
    unet_model.load_state_dict(saved_state_dict)
    unet_model.to(device)
    unet_model.eval()
    input_t = Get_CT_Cache(index, dataset_cache_type.eval_UNet_input)
    label_t = Get_CT_Cache(index, dataset_cache_type.eval_UNet_label)
    with torch.no_grad():
        result_t = unet_model(input_t.unsqueeze(0).to(device))
    CT_Transform.show_one_ct_tensor(
        input_t, settings[Config_Item.UNet_train_thickness_half])
    CT_Transform.show_one_ct_tensor(
        label_t, settings[Config_Item.UNet_train_thickness_half], (0, 1))
    CT_Transform.show_one_ct_tensor(
        result_t.squeeze(0).to(cpu), settings[Config_Item.UNet_train_thickness_half], (0, 1))



