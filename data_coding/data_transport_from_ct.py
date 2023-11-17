"""本文件存储用于将原始ct图像转为可被模型直接使用的全部相关函数"""

import csv
import glob

import SimpleITK as sitk
import joblib
import time
import random

import data_coding.data_cache_tool
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import config.extern_var as EXTERN_VAR
import pandas as pd
from utils.timer import *
from data_coding.data_cache import *
from config.extern_var import settings


class CT_One_Graphic:
    """用于存储单张CT图像本身及各种附带信息的类"""

    def __init__(self, ct_path):
        """初始化时就将所有信息读入并转化为数组"""
        ct_image = sitk.ReadImage(ct_path)
        ## Take an Example At Notes Below
        self.seriesuid, _ = os.path.splitext(os.path.basename(ct_path))
        self.dimsize = ct_image.GetSize()  ## eg: (512, 512, 133)
        self.spacing = ct_image.GetSpacing()  ## eg: (0.78125, 0.78125, 2.5)
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
        self.candidates_list = []
        self.candidates_list_length = 0
        self.annotations_list = []
        self.annotations_list_length = 0
        self.ct_graphics_paths = []
        self.ct_graphics_length = 0
        self.device = torch.device(settings['device'])

        ## 需每次复位的信息
        self.annotations_list_current_pointer = 0
        self.annotations_list_current_length = 0
        self.ct_caches_length = 0
        self.ct_annotated_length = 0
        self.ct_unannotated_length = 0

        self.cache_path = None
        self.cache_input_path = None
        self.cache_label_path = None
        self.cache_unannotated_path = None
        self.cache_annotated_path = None
        self.mhd_list = None

        self.cache_thickness = None
        self.cache_size_half = None
        self.cache_size = None
        self.cache_thickness_half = None

    def Extract_Info_From_CSV(self):
        """从表单中提取所有的候选结节信息并整合，最终全部缓存"""
        ## 读取所有块的信息，要求其根据uid排序
        with open(settings[Config_Item.candidates_list_path], 'r') as f:
            ## 列表结构：seriesuid coordX coordY coordZ class
            self.candidates_list = (pd.read_csv(f)).values.tolist()
            self.candidates_list.sort(key=lambda x: x[0])
            self.candidates_list_length = len(self.candidates_list)
            self.candidates_list.append([["WATCHDOG", 0.0, 0.0, 0.0, 0]])
            # print(self.candidates_list)

        ## 读取所有可疑结节信息，要求其根据uid排序
        with open(settings[Config_Item.annotations_list_path], 'r') as f:
            ## 列表结构：seriesuid coordX coordY coordZ diameter_mm
            self.annotations_list = (pd.read_csv(f)).values.tolist()
            self.annotations_list.sort(key=lambda x: x[0])
            self.annotations_list_length = len(self.annotations_list)
            self.annotations_list.append([["WATCHDOG", 0.0, 0.0, 0.0, 1.0]])
            # print(self.annotations_list)

    def Make_Cache_for_Type(self, cache_type):
        """制作指定类型的缓存，有train, eval两种情形"""

        self.Get_CT_Graphic_Info(cache_type)

        self.cache_size = settings[Config_Item.Type_train_cache_size]
        self.cache_size_half = settings[Config_Item.Type_train_cache_size_half]
        self.cache_thickness = settings[Config_Item.Type_train_thickness]
        self.cache_thickness_half = settings[Config_Item.Type_train_thickness_half]

        ## 遍历所有图像，逐个遍历列表解析对应的图像
        j = 0
        counter = DynamicCounter(self.candidates_list_length, f"Cache {self.cache_path}", 100)
        for i in range(self.ct_graphics_length):
            ## 取对应的CT图像，并进行进一步的处理
            ## 利用二者排序一致的特点
            ct_graphic = CT_One_Graphic(self.ct_graphics_paths[i][1])
            ## 寻找在annotations_list中对应的位置
            try:
                while self.annotations_list[self.annotations_list_current_pointer][0] < ct_graphic.seriesuid:
                    self.annotations_list_current_pointer += 1
            except IndexError:
                break
            k = self.annotations_list_current_pointer
            ## 已设置过看门狗
            while self.annotations_list[k][0] == ct_graphic.seriesuid:
                k += 1
            self.annotations_list_current_length = k - self.annotations_list_current_pointer
            ## 遍历对于该图像的所有candidates
            while j < self.candidates_list_length:
                ## 寻找uid相匹配的所有块对应的东西
                if self.candidates_list[j][0] == self.ct_graphics_paths[i][0]:
                    while True:
                        self.Dealing_One_Candidate_for_Type(j, ct_graphic)
                        counter.increment()
                        j += 1
                        ## 利用在csv表格中同一uid都集中在一起
                        if self.candidates_list[j][0] != self.ct_graphics_paths[i][0]:
                            break
                    break
                else:
                    j += 1
            self.annotations_list_current_pointer = k
        counter.stop()

        with open(os.path.join(self.cache_annotated_path, "note"), 'w') as f:
            f.write(f"{self.ct_annotated_length}")
        with open(os.path.join(self.cache_unannotated_path, "note"), 'w') as f:
            f.write(f"{self.ct_unannotated_length}")

    def _transform_from_xyz_to_whc(self, j, ct_graphic, jump_unannotated=True):
        """将原始坐标转化为像素点坐标"""

        w_n_raw = self.candidates_list[j][1]
        h_n_raw = self.candidates_list[j][2]
        c_n_raw = self.candidates_list[j][3]
        diameter = 0

        if self.candidates_list[j][4] == 0:
            if jump_unannotated is True:
                return
            else:
                if j % settings[Config_Item.unannodated_data_rate] != 0:
                    return
        else:
            whether_annotated = False
            for k in range(self.annotations_list_current_length):
                if abs(w_n_raw -
                       self.annotations_list[self.annotations_list_current_pointer + k][1]) <= 5.0 and \
                        abs(h_n_raw -
                            self.annotations_list[self.annotations_list_current_pointer + k][2]) <= 5.0 and \
                        abs(c_n_raw -
                            self.annotations_list[self.annotations_list_current_pointer + k][3]) <= 5.0:
                    w_n_raw = self.annotations_list[self.annotations_list_current_pointer + k][1]
                    h_n_raw = self.annotations_list[self.annotations_list_current_pointer + k][2]
                    c_n_raw = self.annotations_list[self.annotations_list_current_pointer + k][3]
                    diameter = self.annotations_list[self.annotations_list_current_pointer + k][4] \
                               / ct_graphic.spacing[0]
                    diameter = self._int_border(diameter)
                    whether_annotated = True
                    break
            ## 如果不存在匹配数据则直接返回
            if whether_annotated is False:
                return

        ## 转化CSV标注信息中的[Z, Y, X]坐标为[C, H, W]
        w_n = (w_n_raw - ct_graphic.offset[0]) / ct_graphic.spacing[0] \
              - settings[Config_Item.raw_data_cropped_size]  ## x
        w_n = self._int_border(w_n)
        h_n = (h_n_raw - ct_graphic.offset[1]) / ct_graphic.spacing[1] \
              - settings[Config_Item.raw_data_cropped_size]  ## y
        h_n = self._int_border(h_n)
        c_n = (c_n_raw - ct_graphic.offset[2]) / ct_graphic.spacing[2]  ## z
        c_n = self._int_border(c_n)

        return w_n, h_n, c_n, diameter

    def _check_all_borders(self, w_n, h_n, c_n, cache_size, cache_thickness, ct_graphic):
        """检查所有的边界"""
        w_n_checked = self._check_border(w_n, cache_size, settings[Config_Item.raw_data_centercrop_size])
        h_n_checked = self._check_border(h_n, cache_size, settings[Config_Item.raw_data_centercrop_size])
        c_n_checked = self._check_border(c_n, cache_thickness, ct_graphic.dimsize[2])
        return w_n_checked, h_n_checked, c_n_checked

    def _get_output(self, w_n_checked, h_n_checked, c_n_checked, ct_graphic):
        """获取基本tensor输入"""
        ct_slices_raw = ct_graphic.ct_tensor[
                        c_n_checked - self.cache_thickness_half:
                        c_n_checked + self.cache_thickness - self.cache_thickness_half, :, :]
        ct_slices_raw = ct_slices_raw.contiguous()

        ## 这里在制作输入级的原始切割
        ct_slices_output = ct_slices_raw[
                           :, :,
                           w_n_checked - self.cache_size_half: w_n_checked + self.cache_size - self.cache_size_half]
        ct_slices_output = ct_slices_output[
                           :,
                           h_n_checked - self.cache_size_half: h_n_checked + self.cache_size - self.cache_size_half, :]
        ct_slices_output = ct_slices_output.contiguous()

        return ct_slices_output

    def Dealing_One_Candidate_for_Type(self, j, ct_graphic):
        """对于Type的单个Candidate处理"""
        try:
            w_n, h_n, c_n, diameter = self._transform_from_xyz_to_whc(j, ct_graphic, jump_unannotated=False)
        except TypeError:
            return

        ## 边界检查并调整
        w_n_checked, h_n_checked, c_n_checked = self._check_all_borders(
            w_n, h_n, c_n, self.cache_size, self.cache_thickness, ct_graphic)

        ## 获取基本output
        ct_slices_output = self._get_output(w_n_checked, h_n_checked, c_n_checked, ct_graphic)
        ct_slices_output = self._Normalize(ct_slices_output, (-1, 1), (0, 1))

        # CT_Transform.show_one_ct_tensor(ct_slices_output, self.cache_thickness_half, (0, 1))
        if self.candidates_list[j][4] == 0:
            ## 缓存input数据
            ct_slices_unannotated = self._set_to_threshold(ct_slices_output, self.device)
            Save_CT_Candidate(self.ct_unannotated_length, self.cache_unannotated_path, ct_slices_unannotated)
            self.ct_unannotated_length += 1
            # CT_Transform.show_one_ct_tensor(ct_slices_unannotated, self.cache_thickness_half, (0, 1))
            # print()
        else:
            if diameter > self.cache_size:
                diameter = self.cache_size
            w_n -= w_n_checked - self.cache_size_half
            h_n -= h_n_checked - self.cache_size_half
            w_n_checked = self._check_border(w_n, diameter, self.cache_size)
            h_n_checked = self._check_border(h_n, diameter, self.cache_size)

            ## 获得滤去背景的标签并缓存
            ct_slices_annotated = self.Make_Annoted_Infomation(w_n_checked, h_n_checked, diameter, ct_slices_output,
                                                               threshold=settings[Config_Item.Type_train_threshold])
            Save_CT_Candidate(self.ct_annotated_length, self.cache_annotated_path, ct_slices_annotated)
            self.ct_annotated_length += 1
            # CT_Transform.show_one_ct_tensor(ct_slices_annotated, self.cache_thickness_half, (0, 1))
            # print()
        # print()

    def Get_CT_Graphic_Info(self, cache_type):
        """根据cache_type的对于CT图像的预处理"""
        ## 鲁棒性复位部分信息
        self.annotations_list_current_pointer = 0
        self.annotations_list_current_length = 0
        self.ct_caches_length = 0
        self.ct_annotated_length = 0
        self.ct_unannotated_length = 0

        self.cache_path = from_cache_type_to_parameter(cache_type)[0]
        if cache_type == dataset_cache_type.train_UNet:
            self.cache_input_path = from_cache_type_to_parameter(dataset_cache_type.train_UNet_input)[0]
            self.cache_label_path = from_cache_type_to_parameter(dataset_cache_type.train_UNet_label)[0]
            subset_begin = 0
            subset_end = subset_begin + settings[Config_Item.num_for_train]
        elif cache_type == dataset_cache_type.eval_UNet:
            self.cache_input_path = from_cache_type_to_parameter(dataset_cache_type.eval_UNet_input)[0]
            self.cache_label_path = from_cache_type_to_parameter(dataset_cache_type.eval_UNet_label)[0]
            subset_begin = settings[Config_Item.num_for_train]
            subset_end = subset_begin + settings[Config_Item.num_for_eval]
        elif cache_type == dataset_cache_type.train_type:
            self.cache_annotated_path = from_cache_type_to_parameter(dataset_cache_type.train_type_annotated)[0]
            self.cache_unannotated_path = from_cache_type_to_parameter(dataset_cache_type.train_type_unannotated)[0]
            subset_begin = 0
            subset_end = subset_begin + settings[Config_Item.num_for_train]
        elif cache_type == dataset_cache_type.eval_type:
            self.cache_annotated_path = from_cache_type_to_parameter(dataset_cache_type.eval_type_annotated)[0]
            self.cache_unannotated_path = from_cache_type_to_parameter(dataset_cache_type.eval_type_unannotated)[0]
            subset_begin = settings[Config_Item.num_for_train]
            subset_end = subset_begin + settings[Config_Item.num_for_eval]
        else:
            raise abort.CacheAbort("Wrong Cache Type!!!")

        self.mhd_list = []
        for subset in range(subset_begin, subset_end):
            self.mhd_list += glob.glob(
                os.path.join(settings[Config_Item.dataset_path], "subset" + str(subset), "*.mhd"))
        self.ct_graphics_paths = [(os.path.splitext(os.path.basename(path))[0], path) for path in self.mhd_list]
        self.ct_graphics_paths.sort(key=lambda x: x[0])
        self.ct_graphics_length = len(self.ct_graphics_paths)
        self.ct_graphics_paths.append(("WATCHDOG", "WATCHDOG"))

    def Make_Cache_for_UNet(self, cache_type):
        """制作指定类型的缓存，有train, eval两种情形"""

        self.Get_CT_Graphic_Info(cache_type)

        self.cache_size = settings[Config_Item.UNet_train_input_cache_size]
        self.cache_size_half = settings[Config_Item.UNet_train_input_cache_size_half]
        self.cache_thickness = settings[Config_Item.UNet_train_thickness]
        self.cache_thickness_half = settings[Config_Item.UNet_train_thickness_half]

        ## 遍历所有图像，逐个遍历列表解析对应的图像
        j = 0
        counter = DynamicCounter(self.candidates_list_length, f"Cache {self.cache_path}", 100)
        for i in range(self.ct_graphics_length):
            ## 取对应的CT图像，并进行进一步的处理
            ## 利用二者排序一致的特点
            ct_graphic = CT_One_Graphic(self.ct_graphics_paths[i][1])
            ## 寻找在annotations_list中对应的位置
            if ct_graphic.seriesuid < self.annotations_list[self.annotations_list_current_pointer][0]:
                continue
            try:
                while self.annotations_list[self.annotations_list_current_pointer][0] < ct_graphic.seriesuid:
                    self.annotations_list_current_pointer += 1
                if ct_graphic.seriesuid < self.annotations_list[self.annotations_list_current_pointer][0]:
                    continue
            except IndexError:
                break
            k = self.annotations_list_current_pointer
            ## 已设置过看门狗
            while self.annotations_list[k][0] == ct_graphic.seriesuid:
                k += 1
            self.annotations_list_current_length = k - self.annotations_list_current_pointer
            ## 遍历对于该图像的所有candidates
            while j < self.candidates_list_length:
                ## 寻找uid相匹配的所有块对应的东西
                if self.candidates_list[j][0] == self.ct_graphics_paths[i][0]:
                    while True:
                        self.Dealing_One_Candidate_for_UNet(j, ct_graphic)
                        counter.increment()
                        j += 1
                        ## 利用在csv表格中同一uid都集中在一起
                        if self.candidates_list[j][0] != self.ct_graphics_paths[i][0]:
                            break
                    break
                else:
                    j += 1
            self.annotations_list_current_pointer = k
        counter.stop()

        with open(os.path.join(self.cache_input_path, "note"), 'w') as f:
            f.write(f"{self.ct_caches_length}")
        with open(os.path.join(self.cache_label_path, "note"), 'w') as f:
            f.write(f"{self.ct_caches_length}")

    def _check_border(self, x_n, min_length, max_border):
        """此函数用于检查是否越界并取为整形，要求输入的x_n为整形"""
        min_length_half = int(min_length / 2)
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

    def _set_to_threshold(self, tensor_t, device):
        """
        该代码直接改变原张量，用于将范围之外的张量置-1，保留区域包含前边界，不包含后边界，
        输入张量认为是(N, C, H, W)格式的，range为元组类型
        """
        ## 创建一个全1的张量作为掩码
        device = torch.device(device)
        mask = torch.ones_like(tensor_t).to(device)
        ## 将掩码塑形
        mask *= settings[Config_Item.UNet_low_threshold_rate]
        ## 使用掩码将输入张量中指定范围之外的部分进行软阈值处理
        return tensor_t * mask

    def _set_to_threshold_with_range(self, tensor_t, w_range, h_range, k_layer, device, threshold):
        """
        该代码改变原张量，用于将范围之外的张量置0，保留区域包含前边界，不包含后边界，
        输入张量认为是(N, C, H, W)格式的，range为元组类型
        """
        ## 创建一个全1的张量作为掩码
        device = torch.device(device)
        mask = torch.ones_like(tensor_t).to(device)
        ## 将指定范围之外的部分置0
        mask[k_layer, :, :w_range[0]] = threshold
        mask[k_layer, :, w_range[1]:] = threshold
        mask[k_layer, :h_range[0], :] = threshold
        mask[k_layer, h_range[1]:, :] = threshold
        ## 使用掩码将输入张量中指定范围之外的部分置0
        tensor_t *= mask
        return tensor_t

    def _set_to_zero_with_range(self, tensor_t, w_range, h_range, k_layer, device):
        """
        该代码改变原张量，用于将范围之外的张量置0，保留区域包含前边界，不包含后边界，
        输入张量认为是(N, C, H, W)格式的，range为元组类型
        """
        ## 创建一个全1的张量作为掩码
        device = torch.device(device)
        mask = torch.ones_like(tensor_t).to(device)
        ## 将指定范围之外的部分置0
        mask[k_layer, :, :w_range[0]] = 0
        mask[k_layer, :, w_range[1]:] = 0
        mask[k_layer, :h_range[0], :] = 0
        mask[k_layer, h_range[1]:, :] = 0
        ## 使用掩码将输入张量中指定范围之外的部分置0
        tensor_t *= mask
        return tensor_t

    def Dealing_One_Candidate_for_UNet(self, j, ct_graphic):
        """填充单个候选结节的具体信息"""
        try:
            w_n, h_n, c_n, diameter = self._transform_from_xyz_to_whc(j, ct_graphic, jump_unannotated=True)
        except TypeError:
            return

        ## 边界检查并调整
        w_n_checked = self._check_border(w_n, settings[Config_Item.UNet_train_input_cache_size],
                                         settings[Config_Item.raw_data_centercrop_size])
        h_n_checked = self._check_border(h_n, settings[Config_Item.UNet_train_input_cache_size],
                                         settings[Config_Item.raw_data_centercrop_size])
        c_n_checked = self._check_border(c_n, settings[Config_Item.UNet_train_thickness], ct_graphic.dimsize[2])

        ## 取切片，在训练时只将标注部分呈现，其余部分置0（1是语义）（在完全评估时才呈现非标注部分）
        ## 这样可以让UNET更好的关注结节部分   (N, C, H, W)，即(N, Z, Y, X)，大小为64*64
        # print(ct_graphic.ct_tensor.shape)

        ## 这里在制作UNET输入级训练信息，所有输入级训练图全部需要在Z方向裁剪和padding，输入级操作并不区分是否为annotated类型
        ## 注意，测试信息也有与训练数据预处理方式完全一致，仅因分割数据集而分开
        ## 但是评估信息并没有掩码处理，而是整个图像直接输入；对于高度假阳性的过滤放在二级模型上

        ## 缓存input数据
        ct_slices_output = self._get_output(w_n_checked, h_n_checked, c_n_checked, ct_graphic)
        Save_CT_Candidate(self.ct_caches_length, self.cache_input_path, ct_slices_output)

        # CT_Transform.show_one_ct_tensor(ct_slices_output, settings[Config_Item.UNet_train_thickness_half], (0, 1))

        ## 处理label
        ## 这里认为 X, Y 尺度是一样的，将坐标转换为output内的坐标
        ct_slices_output = self._Normalize(ct_slices_output, (-1, 1), (0, 1))
        if diameter > settings[Config_Item.UNet_train_input_cache_size]:
            diameter = settings[Config_Item.UNet_train_input_cache_size]
        w_n -= w_n_checked - settings[Config_Item.UNet_train_input_cache_size_half]
        h_n -= h_n_checked - settings[Config_Item.UNet_train_input_cache_size_half]
        w_n_checked = self._check_border(w_n, diameter, settings[Config_Item.UNet_train_input_cache_size])
        h_n_checked = self._check_border(h_n, diameter, settings[Config_Item.UNet_train_input_cache_size])

        ## 获得滤去背景的标签并缓存
        ct_slices_label = self.Make_Annoted_Infomation(w_n_checked, h_n_checked, diameter, ct_slices_output,
                                                       threshold=0.0)
        Save_CT_Candidate(self.ct_caches_length, self.cache_label_path, ct_slices_label)

        # CT_Transform.show_one_ct_tensor(ct_slices_label, settings[Config_Item.UNet_train_thickness_half], (0, 1))
        # print()

        self.ct_caches_length += 1

    def Make_Annoted_Infomation(self, w_n, h_n, diameter, ct_slices_output, threshold=0.0):
        """制作输入Unet的标注信息"""

        ## 下面是使用从中心开始延伸搜索的方法，这种方法可以避免将肺壁记入最终的标注中
        ct_slices_label = torch.ones_like(ct_slices_output).to(self.device)
        search_begin = 0
        for k in range(self.cache_thickness):
            label_radius_tmp = [2, 2]
            label_radius_ori = [2, 2]
            label_radius = [2, 2]  # [w, h]
            for m in [0, 1]:
                ## [0]位置总是目标位置
                label_radius_tmp[0], label_radius_tmp[1] = 1 + search_begin, 0
                try:
                    while ct_slices_output[k, h_n + label_radius_tmp[1 - m], w_n + label_radius_tmp[m]] \
                            >= settings[Config_Item.UNet_low_threshold_rate] and \
                            ct_slices_output[k, h_n - label_radius_tmp[1 - m], w_n - label_radius_tmp[m]] \
                            >= settings[Config_Item.UNet_low_threshold_rate]:
                        label_radius_tmp[0] += 1
                except IndexError:
                    label_radius_tmp[0] -= 1
                ## 记录未进行进一步扩展时的原始半径
                label_radius_ori[m] = label_radius_tmp[0]
                for n in range(settings[Config_Item.UNet_Radius_Expansion] + 1):
                    try:
                        label_radius_tmp[0] += 1
                        ct_slices_output[k, h_n + label_radius_tmp[1 - m], w_n + label_radius_tmp[m]]
                    except IndexError:
                        include_max = 0
                        break
                    else:
                        include_max = 1
                label_radius_tmp[0] -= 1
                ## 保存对应探测结果
                label_radius[m] = label_radius_tmp[0]

            ## 从output中裁剪出第k层的label掩码板
            ct_slices_label = self._set_to_threshold_with_range(ct_slices_output,
                                                           (w_n - label_radius[0], w_n + label_radius[0] + include_max),
                                                           (h_n - label_radius[1], h_n + label_radius[1] + include_max),
                                                           k, "cuda", threshold)

            ## 进行进一步的掩码板细化，去除边角料
            radius_border = (label_radius_ori[0] + label_radius_ori[1]) / 2
            ## 横向检查联通
            for j in range(h_n - label_radius[1], h_n + label_radius[1] + include_max):
                try:
                    i = int((radius_border ** 2 - (j - h_n) ** 2) ** 0.5)
                except TypeError:
                    i = 0
                i_small = w_n - i
                i_big = w_n + i
                ## 处理低界的那一部分
                all_zero_flag = 0
                for m in range(i_small, w_n - label_radius[0] - 1, -1):
                    try:
                        if all_zero_flag == 0:
                            if (ct_slices_label[k, j, m] < settings[Config_Item.UNet_low_threshold_rate] and
                                ct_slices_label[k, j, m - 1] < settings[Config_Item.UNet_low_threshold_rate]) or \
                                    (ct_slices_label[k, j, m] < ct_slices_label[k, j, m - 1] <
                                     ct_slices_label[k, j, m - 2]):
                                all_zero_flag = 1
                                ct_slices_label[k, j, m] *= threshold
                        else:
                            ct_slices_label[k, j, m] *= threshold
                    except IndexError:
                        break
                ## 处理高界那一部分
                all_zero_flag = 0
                for m in range(i_big, w_n + label_radius[0] + include_max, 1):
                    try:
                        if all_zero_flag == 0:
                            if (ct_slices_label[k, j, m] < settings[Config_Item.UNet_low_threshold_rate] and
                                ct_slices_label[k, j, m + 1] < settings[Config_Item.UNet_low_threshold_rate]) or \
                                    (ct_slices_label[k, j, m] < ct_slices_label[k, j, m + 1] <
                                     ct_slices_label[k, j, m + 2]):
                                all_zero_flag = 1
                                ct_slices_label[k, j, m] *= threshold
                        else:
                            ct_slices_label[k, j, m] *= threshold
                    except IndexError:
                        break

            ## 纵向检查联通
            for i in range(w_n - label_radius[0], w_n + label_radius[0] + include_max):
                try:
                    j = int((radius_border ** 2 - (i - w_n) ** 2) ** 0.5)
                except TypeError:
                    j = 0
                j_small = h_n - j
                j_big = h_n + j
                ## 处理低界的那一部分
                all_zero_flag = 0
                for m in range(j_small, h_n - label_radius[1] - 1, -1):
                    try:
                        if all_zero_flag == 0:
                            if (ct_slices_label[k, m, i] < settings[Config_Item.UNet_low_threshold_rate] and
                                ct_slices_label[k, m - 1, i] < settings[Config_Item.UNet_low_threshold_rate]) or \
                                    (ct_slices_label[k, m, i] < ct_slices_label[k, m - 1, i] <
                                     ct_slices_label[k, m - 2, i]):
                                all_zero_flag = 1
                                ct_slices_label[k, m, i] *= threshold
                        else:
                            ct_slices_label[k, m, i] *= threshold
                    except IndexError:
                        break
                ## 处理高界那一部分
                all_zero_flag = 0
                for m in range(j_big, h_n + label_radius[1] + include_max, 1):
                    try:
                        if all_zero_flag == 0:
                            if (ct_slices_label[k, m, i] < settings[Config_Item.UNet_low_threshold_rate] and
                                ct_slices_label[k, m + 1, i] < settings[Config_Item.UNet_low_threshold_rate]) or \
                                    (ct_slices_label[k, m, i] < ct_slices_label[k, m + 1, i] <
                                     ct_slices_label[k, m + 2, i]):
                                all_zero_flag = 1
                                ct_slices_label[k, m, i] *= threshold
                        else:
                            ct_slices_label[k, m, i] *= threshold
                    except IndexError:
                        break

        return ct_slices_label

    @staticmethod
    def _Normalize(val, ori, dest):
        """将数据缩放至指定范围，ori, dest均表示为(min, max)"""
        return (val - ori[0]) * (dest[1] - dest[0]) / (ori[1] - ori[0]) + dest[0]

    def Cache_All_CT_Candidates(self):
        """用于将所有最终用于神经网络处理的结节刷入磁盘缓存，此处用Extract_Info_From_CSV实现了"""
        self.Extract_Info_From_CSV()
        self.Make_Cache_for_UNet(dataset_cache_type.train_UNet)
        self.Make_Cache_for_UNet(dataset_cache_type.eval_UNet)
        self.Make_Cache_for_Type(dataset_cache_type.train_type)
        self.Make_Cache_for_Type(dataset_cache_type.eval_type)
        pass

    @staticmethod
    def Test_UNet_Cache(index=None):
        """测试cache的可用性，指定某种cache后从中抽取并显示"""
        if index is None:
            random.seed(int(time.time()))
            index = random.randint(0, settings[Config_Item.train_UNet_dataset_num] - 1)
        print(f"Now We are showing UNet Input and Label, index: {index}")
        input_t = Get_CT_Cache(index, dataset_cache_type.train_UNet_input)
        label_t = Get_CT_Cache(index, dataset_cache_type.train_UNet_label)
        CT_Transform.show_one_ct_tensor(input_t, settings[Config_Item.UNet_train_thickness_half], (-1, 1))
        CT_Transform.show_one_ct_tensor(label_t, settings[Config_Item.UNet_train_thickness_half], (0, 1))
        print(input_t.shape, label_t.shape)

    @staticmethod
    def Test_Type_Cache(index=None):
        """测试cache的可用性，指定某种cache后从中抽取并显示"""
        if index is None:
            index = []
            random.seed(int(time.time()))
            index.append(random.randint(0, settings[Config_Item.train_type_annotated_dataset_num] - 1))
            index.append(random.randint(0, settings[Config_Item.train_type_unannotated_dataset_num] - 1))
        print(f"Now We are showing Type Annotated and Unannotated, index: {index}")
        annotated_t = Get_CT_Cache(index[0], dataset_cache_type.train_type_annotated)
        unannotated_t = Get_CT_Cache(index[1], dataset_cache_type.train_type_unannotated)
        for k in range(settings[Config_Item.Type_train_thickness_half]):
            CT_Transform.show_one_ct_tensor(annotated_t, k, (0, 1))
        # CT_Transform.show_one_ct_tensor(unannotated_t, settings[Config_Item.Type_train_thickness_half], (0, 1))
        print(annotated_t.shape, unannotated_t.shape)


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
    def show_one_ct_tensor(ct_tensor, slice_pos=53, ct_scale=(-1, 1)):
        """显示一个numpy格式的CT切片图"""
        # 取一个切片来观察，输入默认为(N, C, H, W)
        if ct_tensor.dim() == 4:
            ct_tensor = ct_tensor.squeeze(0)
        ct_array = ct_tensor.to(torch.device("cpu")).detach().numpy()
        ct_one_slice = ct_array[slice_pos, :, :]
        plt.imshow(ct_one_slice, cmap='gray', vmin=ct_scale[0], vmax=ct_scale[1])
        plt.show()

    @staticmethod
    def transform_one_sample_to_tensor_for_train(ct_path=None, ct_image=None):
        """将一个CT图转化为tensor并收集相关信息"""
        ct_array = CT_Transform.transform_one_sample_to_array_for_train(ct_path=ct_path, ct_image=ct_image)
        ## Pytorch 输入向量需要为 (C, H, W)，这里将Z轴作为通道输入
        ## 后续需要拆分 C 通道为适合输入的数量(这里选为10,因为绝大部分结节的直径都比10个切片小)
        ct_tensor = torch.from_numpy(ct_array)  # (C, H, W)
        ## ct_tensor的结果从[-1000,1000]归一化至[0, 1]
        # ct_tensor = (ct_tensor + 1000) / 2000
        ## 归一化至[-1, 1]的情况，这样后面的激活函数ReLU的处理才是有效的
        ct_tensor = ct_tensor.to(torch.device(settings['device']))
        ct_tensor = ct_tensor / 1000
        transform = transforms.CenterCrop(settings[Config_Item.raw_data_centercrop_size])
        ct_tensor = transform(ct_tensor)
        # print("ct_tensor_ori:", ct_tensor_ori.shape)
        # print("ct_tensor:",ct_tensor.shape)
        return ct_tensor


if __name__ == '__main__':
    pass
