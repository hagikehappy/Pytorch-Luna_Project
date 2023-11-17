from data_coding.data_cache import *
from data_coding.data_transport_from_ct import *


def test_unet_basic_use():
    """该函数用于测试unet的使用正确"""
    unet = CustomUNet()
    index = 17
    annotated_raw = Get_CT_Annoted_RAW(index)
    annotated_input = Get_CT_Annoted_INPUT(index)
    annotated_label = Get_CT_Annoted_LABEL(index)
    CT_Transform.show_one_ct_tensor(annotated_raw, 2)
    CT_Transform.show_one_ct_tensor(annotated_input, 2)
    CT_Transform.show_one_ct_tensor(annotated_label, 2)

    unet.eval()
    result_train = unet(annotated_input.unsqueeze(0))
    result_eval = unet(annotated_raw.unsqueeze(0))
    print(result_train.shape)
    CT_Transform.show_one_ct_tensor(result_train.squeeze(0), 2)
    CT_Transform.show_one_ct_tensor(result_eval.squeeze(0), 2)




