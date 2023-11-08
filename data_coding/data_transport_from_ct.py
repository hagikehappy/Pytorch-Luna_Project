import csv

import SimpleITK as sitk
import data_coding.data_cache
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
from data_coding.data_cache import disk_cache


@disk_cache("dataset/Cache/data_for_unet")
def Get_CT_Candidate(index):
    """用缓存封装的获取最终可用于处理图片的方式"""
    pass


class CT_One_Graphic:
    """用于存储单张CT图像本身及各种附带信息的类"""
    def __init__(self, ct_path):
        """初始化时就将所有信息读入并转化为数组"""
        ct_image = sitk.ReadImage(ct_path)
        ## Take an Example At Notes Below
        self.seriesuid, _ = os.path.splitext(os.path.basename(ct_path))
        self.dimsize = ct_image.GetSize()   ## eg: (512, 512, 133)
        self.spacing = ct_image.GetSpacing()    ## eg: (0.78125, 0.78125, 2.5)
        self.offset = ct_image.GetOrigin()  ## eg: (-191.199997, -185.5, -359.0)
        self.ct_tensor = CT_Transform.transform_one_sample_to_tensor_for_train(ct_image=ct_image)
        # print("uid:", self.seriesuid)
        # print("dimsize:", self.dimsize)
        # print("spacing:", self.spacing)
        # print("offset:", self.offset)


class CT_One_Candidate:
    """存储单个的最终用于训练的数据"""

    def __init__(self):
        """初始化"""
        pass


class CT_All_Candidates:
    """管理所有的块"""

    def __init__(self):
        """初始化"""
        self.candidates_list_path = "dataset/LUNA-Data/CSVFILES/candidates.csv"
        self.annotations_list_path = "dataset/LUNA-Data/CSVFILES/annotations.csv"
        self.candidates_list = []
        self.candidates_list_length = None
        self.annotations_list = []
        self.annotations_list_length = None
        self.ct_graphics_paths = []
        self.ct_graphics_length = None
        self.ct_annoted_graphics_list = []
        self.ct_annoted_graphics_length = None
        self.ct_unannoted_graphics_list = []
        self.ct_unannoted_graphics_length = None


    def Extract_Info_From_CSV(self):
        """从表单中提取所有的候选结节信息"""
        ## 读取所有块的信息
        with open(self.candidates_list_path, 'r') as f:
            ## 列表结构：seriesuid coordX coordY coordZ class
            self.candidates_list = list(csv.reader(f))
            del self.candidates_list[0]
            self.candidates_list_length = len(self.candidates_list)
            # print(len(self.candidates_list_length))

        ## 读取所有可疑结节信息
        with open(self.annotations_list_path, 'r') as f:
            ## 列表结构：seriesuid coordX coordY coordZ diameter_mm
            self.annotations_list = list(csv.reader(f))
            del self.annotations_list[0]
            self.annotations_list_length = len(self.annotations_list)
            # print(len(self.annotations_list_length))

        ## 提取所有图像路径信息
        for dirpath, dirnames, filenames in os.walk("dataset/LUNA-Data"):
            for filename in filenames:
                ct_uid, extension = os.path.splitext(filename)
                if extension == ".mhd":
                    file_path = os.path.join(dirpath, filename)
                    self.ct_graphics_paths.append((ct_uid, file_path))
        self.ct_graphics_length = len(self.ct_graphics_paths)

        ## 遍历所有图像，逐个遍历列表解析对应的图像
        index_annoted = 0
        index_unannoted = 0
        for i in range(self.ct_graphics_length):
            ## 取对应的CT图像
            ct_graphic = CT_One_Graphic(self.ct_graphics_paths[i][1])
            ## 遍历对于该图像的所有candidates
            for j in range(self.candidates_list_length):
                # 寻找uid相匹配的所有块对应的东西
                if self.candidates_list[j][0] == self.ct_graphics_paths[i][0]:
                    while True:
                        w_n = (self.candidates_list[j][1] - ct_graphic.offset[0]) / ct_graphic.spacing[0]   ## x
                        h_n = (self.candidates_list[j][2] - ct_graphic.offset[1]) / ct_graphic.spacing[1]   ## y
                        c_n = (self.candidates_list[j][3] - ct_graphic.offset[2]) / ct_graphic.spacing[2]   ## z

                        j += 1
                        if j >= self.candidates_list_length:
                            break
                        ## 利用在csv表格中同一uid都集中在一起
                        if self.candidates_list[j][0] != self.ct_graphics_paths[i][0]:
                            break
                    break


    def Dealing_One_Candidate(self):
        """填充单个候选结节的具体信息"""
        pass

    def Cache_All_CT_Candidates(self):
        """用于将所有最终用于神经网络处理的结节刷入磁盘缓存"""
        pass




class CT_Transform:
    """将CT数据转化为相应的可被处理的张量形式并附带其它信息的函数库"""

    @staticmethod
    def transform_one_sample_to_array_for_train(ct_path=None, ct_image=None):
        """转换单个样本为array"""
        if ct_image is None:
            ct_image = sitk.ReadImage(ct_path)  # (Z(C), Y(H), X(W))
        # (C, H, W)     [一般为(H, W, C)，但这里已经自动处理了]
        ct_array = np.array(sitk.GetArrayFromImage(ct_image), dtype=np.float32)
        print(ct_array.shape)
        ct_array.clip(-1000, 1000, ct_array)
        return ct_array

    @staticmethod
    def show_one_ct_array(ct_array, slice_pos=53):
        """显示一个numpy格式的CT切片图"""
        # print("ct_array:", ct_array.shape)
        # 取一个切片来观察
        ct_one_slice = ct_array[slice_pos, :, :]
        plt.imshow(ct_one_slice, cmap='gray')
        plt.show()

    @staticmethod
    def show_one_ct_tensor(ct_tensor, slice_pos=53):
        """显示一个numpy格式的CT切片图"""
        # 取一个切片来观察，输入默认为(N, C, H, W)
        ct_array = ct_tensor[0, :, :, :].numpy()
        ct_one_slice = ct_array[slice_pos, :, :]
        plt.imshow(ct_one_slice, cmap='gray')
        plt.show()

    @staticmethod
    def transform_one_sample_to_tensor_for_train(ct_path=None, ct_image=None):
        """将一个CT图转化为tensor并收集相关信息"""
        ct_array = CT_Transform.transform_one_sample_to_array_for_train(ct_path=ct_path, ct_image=ct_image)
        ## Pytorch 输入向量需要为 (N, C, H, W)，这里将Z轴作为通道输入
        ## 后续需要拆分 C 通道为适合输入的数量(这里选为10,因为绝大部分结节的直径都比10个切片小)
        ct_tensor = torch.from_numpy(ct_array)    # (C, H, W)
        ct_tensor = ct_tensor.unsqueeze(0)     # (N, C, H, W)
        # print("ct_tensor_ori:", ct_tensor_ori.shape)
        # print("ct_tensor:",ct_tensor.shape)
        return ct_tensor






if __name__ == '__main__':
    pass
