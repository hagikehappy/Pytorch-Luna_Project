from data_coding.data_transport_from_ct import *
import tensorboard


def test_data_transport_from_ct():
    """测试是否能正常加载单个CT扫描"""
    ct_path = 'dataset/LUNA-Data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'
    trans = CT_Transform()
    # ct_array = trans.transform_one_sample_to_array_for_train(ct_path=ct_path)
    ## 显示一个切片
    # trans.show_one_ct_array(ct_array, slice_pos=50)
    ## 显示一组切片
    # sizez = ct_array.shape[0]
    # for i in range(0, sizez, 5):
    #     trans.show_one_ct_array(ct_array, slice_pos=i)
    ## 上面的CT的切片是随机选的，下面选一个带有结节的
    # trans.show_one_ct_array(ct_array, slice_pos=15)
    ## 下面将要转换一个sample至tensor
    ct_candidate = CT_One_Graphic(ct_path)
    CT_Transform.show_one_ct_tensor(ct_tensor=ct_candidate.ct_tensor)



def test_combination_of_ct_graphics_and_csv():
    """测试是否能正确结合graphics和csv标注信息"""
    ct_all_candidates = CT_All_Candidates()
    ct_all_candidates.Extract_Info_From_CSV()

