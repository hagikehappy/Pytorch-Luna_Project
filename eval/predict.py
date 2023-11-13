

from config.extern_var import settings
import torch.nn as nn
import torch
from model.custom_unet import CustomUNet as UNet
from data_coding.data_cache import *
from data_coding.data_transport_from_ct import CT_Transform


def UNet_Predict(index):
    """用于使用UNet来生成处理图片的结果"""
    device = torch.device(settings['device'])
    cpu = torch.device("cpu")
    unet_model = UNet()
    saved_state_dict = torch.load(settings['UNet_save_path'] + 'unet_model.pth')
    unet_model.load_state_dict(saved_state_dict)
    unet_model.to(device)
    unet_model.eval()
    raw = Get_CT_Annoted_RAW(index).unsqueeze(0).to(device)
    label = Get_CT_Annoted_LABEL(index)
    result = unet_model(raw)
    CT_Transform.show_one_ct_tensor(raw.squeeze(0).to(cpu), 2)
    CT_Transform.show_one_ct_tensor(label, 2)
    CT_Transform.show_one_ct_tensor(result.squeeze(0).to(cpu), 2)



