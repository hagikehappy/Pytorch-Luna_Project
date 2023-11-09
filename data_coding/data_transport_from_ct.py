"""本文件存储用于将原始ct图像转为可被模型直接使用的全部相关函数"""


import csv
import SimpleITK as sitk
import data_coding.data_cache
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
from data_coding.data_cache import disk_cache
import config.extern_var as EXTERN_VAR
import pandas as pd
from tools.tool import DynamicCounter


def Get_CT_Candidate(index):
    """用缓存封装的获取最终可用于处理图片的方式"""
    pass

def Save_CT_Candidate(index, path, *args, **kwargs):
    """存储可用CT数据的"""



class CT_One_Graphic:
    """用于存储单张CT图像本身及各种附带信息的类"""
    def __init__(self, ct_path):
        """初始化时就将所有信息读入并转化为数组"""
        ct_image = sitk.ReadImage(ct_path)
        ## Take an Example At Notes Below
        self.seriesuid, _ = os.path.splitext(os.path.basename(ct_path))
        self.dimsize = list(ct_image.GetSize())   ## eg: (512, 512, 133)
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
        self.candidates_list_length = 0
        self.annotations_list = []
        self.annotations_list_length = 0
        self.ct_graphics_paths = []
        self.ct_graphics_length = 0
        self.ct_annoted_slices_length = 0
        self.ct_unannoted_slices_length = 0
        self.ct_annoted_slices_cache_path = "dataset/Cache/data_for_unet/annoted_slices"
        self.ct_unannoted_slices_cache_path = "dataset/Cache/data_for_unet/unannoted_slices"


    def Extract_Info_From_CSV(self):
        """从表单中提取所有的候选结节信息"""
        ## 读取所有块的信息，要求其根据uid排序
        with open(self.candidates_list_path, 'r') as f:
            ## 列表结构：seriesuid coordX coordY coordZ class
            self.candidates_list = (pd.read_csv(f)).values.tolist()
            self.candidates_list.sort(key=lambda x: x[0])
            self.candidates_list_length = len(self.candidates_list)
            # print(self.candidates_list)

        ## 读取所有可疑结节信息，要求其根据uid排序
        with open(self.annotations_list_path, 'r') as f:
            ## 列表结构：seriesuid coordX coordY coordZ diameter_mm
            self.annotations_list = (pd.read_csv(f)).values.tolist()
            self.annotations_list.sort(key=lambda x: x[0])
            self.annotations_list_length = len(self.annotations_list)
            # print(self.annotations_list)

        ## 提取所有图像路径信息
        for dirpath, dirnames, filenames in os.walk("dataset/LUNA-Data"):
            for filename in filenames:
                ct_uid, extension = os.path.splitext(filename)
                if extension == ".mhd":
                    file_path = os.path.join(dirpath, filename)
                    self.ct_graphics_paths.append((ct_uid, file_path))
        self.ct_graphics_paths.sort(key=lambda x: x[0])
        self.ct_graphics_length = len(self.ct_graphics_paths)
        # print(self.ct_graphics_paths)

        ## 遍历所有图像，逐个遍历列表解析对应的图像
        index_annoted = 0
        index_unannoted = 0
        j = 0
        counter = DynamicCounter(self.candidates_list_length, "Extract Progression")
        for i in range(self.ct_graphics_length):
            ## 取对应的CT图像，并进行进一步的处理
            ## 利用二者排序一致的特点
            ct_graphic = CT_One_Graphic(self.ct_graphics_paths[i][1])
            ## 遍历对于该图像的所有candidates
            while j < self.candidates_list_length:
                ## 寻找uid相匹配的所有块对应的东西
                if self.candidates_list[j][0] == self.ct_graphics_paths[i][0]:
                    while True:
                        self.Dealing_One_Candidate(i, j, ct_graphic)
                        counter.increment()
                        j += 1
                        if j >= self.candidates_list_length:
                            break
                        ## 利用在csv表格中同一uid都集中在一起
                        if self.candidates_list[j][0] != self.ct_graphics_paths[i][0]:
                            break
                    break

        with open(self.ct_annoted_slices_cache_path, 'w') as f:
            f.write(f"{self.ct_annoted_slices_length}")
        with open(self.ct_unannoted_slices_cache_path, 'w') as f:
            f.write(f"{self.ct_unannoted_slices_length}")

    def _check_border(self, x_n, min_length, max_border):
        """此函数用于检查是否越界并取为整形"""
        min_length_half = int(min_length / 2)
        if x_n - int(x_n) >= 0.5:
            x_n = int(x_n) + 1
        else:
            x_n = int(x_n)
        ## 边界检查
        delta_x_n = min_length_half + 1 - (max_border - x_n)
        if delta_x_n > 0:
            x_n -= delta_x_n
        else:
            delta_x_n = min_length_half - x_n
            if delta_x_n > 0:
                x_n += delta_x_n
        return x_n

    def _int_border(self, x_n):
        """仅用此函数于四舍五入的整数化"""
        if x_n - int(x_n) >= 0.5:
            x_n = int(x_n) + 1
        else:
            x_n = int(x_n)
        return x_n

    def _set_to_zero_with_range(self, tensor_t, w_range, h_range):
        """用于将范围之外的张量置0，保留区域包含前边界，不包含后边界，输入张量认为是(N, C, H, W)格式的，range为元组类型"""
        ## 创建一个全1的张量作为掩码
        mask = torch.ones_like(tensor_t)
        ## 将指定范围之外的部分置0
        mask[:, :, :, :w_range[0]] = 0
        mask[:, :, :, w_range[1]:] = 0
        mask[:, :, :h_range[0], :] = 0
        mask[:, :, h_range[1]:, :] = 0
        ## 使用掩码将输入张量中指定范围之外的部分置0
        tensor_t *= mask
        return tensor_t

    def Dealing_One_Candidate(self, i, j, ct_graphic):
        """填充单个候选结节的具体信息"""
        ## 转化CSV标注信息中的[Z, Y, X]坐标为[C, H, W]
        w_n = (self.candidates_list[j][1] - ct_graphic.offset[0]) / ct_graphic.spacing[0] \
              - EXTERN_VAR.SLICES_X_CROP  ## x
        ct_graphic.dimsize[0] = EXTERN_VAR.SLICES_CROP_X_LENGTH
        h_n = (self.candidates_list[j][2] - ct_graphic.offset[1]) / ct_graphic.spacing[1] \
              - EXTERN_VAR.SLICES_Y_CROP  ## y
        ct_graphic.dimsize[1] = EXTERN_VAR.SLICES_CROP_Y_LENGTH
        c_n = (self.candidates_list[j][3] - ct_graphic.offset[2]) / ct_graphic.spacing[2]  ## z

        ## 边界检查并调整
        w_n_checked = self._check_border(w_n, EXTERN_VAR.SLICES_X_OUTPUT_LENGTH, EXTERN_VAR.SLICES_CROP_X_LENGTH)
        h_n_checked = self._check_border(h_n, EXTERN_VAR.SLICES_Y_OUTPUT_LENGTH, EXTERN_VAR.SLICES_CROP_Y_LENGTH)
        c_n_checked = self._check_border(c_n, EXTERN_VAR.SLICES_THICKNESS, ct_graphic.dimsize[2])

        ## 取切片，在训练时只将标注部分呈现，其余部分置0（1是语义）（在完全评估时才呈现非标注部分）
        ## 这样可以让UNET更好的关注结节部分   (N, C, H, W)，即(N, Z, Y, X)，大小为64*64
        # print(ct_graphic.ct_tensor.shape)

        ## 这里在制作UNET输入级训练信息，所有输入级训练图全部需要在Z方向裁剪和padding，输入级操作并不区分是否为annoted类型
        ct_slices_t = ct_graphic.ct_tensor[:,
                      c_n_checked - EXTERN_VAR.SLICES_THICKNESS_HALF:
                      c_n_checked + EXTERN_VAR.SLICES_THICKNESS - EXTERN_VAR.SLICES_THICKNESS_HALF,
                      :, :]
        ## TODO：Not Done

        ## 这里在制作输出级的标注信息
        ct_slices_t = ct_slices_t[:, :,
                      h_n_checked - EXTERN_VAR.SLICES_Y_OUTPUT_LENGTH_HALF:
                      h_n_checked + EXTERN_VAR.SLICES_Y_OUTPUT_LENGTH - EXTERN_VAR.SLICES_Y_OUTPUT_LENGTH_HALF,
                      :]
        ct_slices_t = ct_slices_t[:, :, :,
                      w_n_checked - EXTERN_VAR.SLICES_X_OUTPUT_LENGTH_HALF:
                      w_n_checked + EXTERN_VAR.SLICES_X_OUTPUT_LENGTH - EXTERN_VAR.SLICES_X_OUTPUT_LENGTH_HALF]
        ct_slices_t = ct_slices_t.contiguous()
        # print(ct_slices_t.shape)

        ## 区分annoted和unannoted两类数据
        ## 这是unannoted类型，直接保存就行
        if self.candidates_list[j][4] == 0:
            Save_CT_Candidate(self.ct_unannoted_slices_length, self.ct_unannoted_slices_cache_path,
                              ct_slices_t)
            self.ct_unannoted_slices_length += 1
        ## 这是annoted类型
        else:
            ## 寻找该位置对应的diameter，一小部分candidate中标注为1的结节在annotation中并无标注


            ## 这里认为 X, Y 尺度是一样的
            diameter = (self.candidates_list[j][1] - ct_graphic.offset[0]) / ct_graphic.spacing[0]
            diameter = self._int_border(diameter)
            w_n_checked = self._check_border(w_n, diameter, EXTERN_VAR.SLICES_CROP_X_LENGTH)
            h_n_checked = self._check_border(h_n, diameter, EXTERN_VAR.SLICES_CROP_Y_LENGTH)
            ct_result_t = self.Make_Annoted_Infomation(i, w_n, h_n, c_n, diameter, ct_slices_t)
            Save_CT_Candidate(self.ct_annoted_slices_length, self.ct_annoted_slices_cache_path,
                              ct_slices_t)
            self.ct_annoted_slices_length += 1
            for k in range(EXTERN_VAR.SLICES_THICKNESS):
                CT_Transform.show_one_ct_tensor(ct_slices_t, k)
            print()

    def Make_Annoted_Infomation(self, i, w_n, h_n, c_n, diameter, ct_slices_t):
        """制作输入Unet的标注信息"""
        ## TODO: 标注信息制作
        pass
        return ct_slices_t


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
        # print(ct_array.shape)
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
        ct_array = ct_tensor.squeeze(0).numpy()
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
        ## ct_tensor的结果从[-1000,1000]归一化至[0,1]，一满足一般的pytorch灰度图像输入要求
        ct_tensor = (ct_tensor + 1000) / 2000
        transform = transforms.CenterCrop((EXTERN_VAR.SLICES_CROP_Y_LENGTH, EXTERN_VAR.SLICES_CROP_X_LENGTH))
        ct_tensor = transform(ct_tensor)
        # print("ct_tensor_ori:", ct_tensor_ori.shape)
        # print("ct_tensor:",ct_tensor.shape)
        return ct_tensor






if __name__ == '__main__':
    pass
