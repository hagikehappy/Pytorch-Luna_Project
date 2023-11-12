from model.custom_unet import *
from data_coding.data_cache import *
from data_coding.data_transport_from_ct import *


def test_unet_basic_use():
    """该函数用于测试unet的使用正确"""
    unet = CustomUNet()
    index = 17
    annoted_raw = Get_CT_Annoted_RAW(index)
    annoted_input = Get_CT_Annoted_INPUT(index)
    annoted_label = Get_CT_Annoted_LABEL(index)
    CT_Transform.show_one_ct_tensor(annoted_raw, 2)
    CT_Transform.show_one_ct_tensor(annoted_input, 2)
    CT_Transform.show_one_ct_tensor(annoted_label, 2)

    unet.eval()
    result_train = unet(annoted_input)
    result_eval = unet(annoted_raw)
    print(result_train.shape)
    CT_Transform.show_one_ct_tensor(result_train, 2)
    CT_Transform.show_one_ct_tensor(result_eval, 2)




