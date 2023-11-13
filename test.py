from test.test_data_coding.test_data_cache import test_data_cache
from test.test_data_coding.test_data_transport_from_ct import test_data_transport_from_ct
from test.test_data_coding.test_data_transport_from_ct import test_combination_of_ct_graphics_and_csv
from test.test_data_coding.test_data_transport_from_ct import test_tensor_cache
from test.test_data_coding.test_data_transport_from_ct import test_cache_mem
from test.test_model.test_unet import test_unet_basic_use
from test.test_train.test_unet_train import test_train_unet
from test.test_eval.test_predict import test_unet_predict


if __name__ == '__main__':
    print()
    # test_data_cache()
    # test_data_transport_from_ct()
    # test_combination_of_ct_graphics_and_csv()
    test_tensor_cache()
    # test_cache_mem()
    # test_unet_basic_use()
    # test_train_unet()
    # test_unet_predict()

