"""数据存取类具体函数"""


import os
import joblib
import functools
import config.extern_var as EXTERN_VAR
from tools.tool import *


def Get_CT_Candidate(index, path):
    """用缓存封装的获取最终可用于处理图片的方式"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(path, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=EXTERN_VAR.CACHE_MAXSIZE_ANNOTED, typed=True)
def Get_CT_Annoted_INPUT(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(EXTERN_VAR.CACHE_PATH_ANNOTED_INPUT, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=EXTERN_VAR.CACHE_MAXSIZE_ANNOTED, typed=True)
def Get_CT_Annoted_LABEL(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(EXTERN_VAR.CACHE_PATH_ANNOTED_LABEL, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=EXTERN_VAR.CACHE_MAXSIZE_ANNOTED, typed=True)
def Get_CT_Annoted_RAW(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(EXTERN_VAR.CACHE_PATH_ANNOTED_RAW, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED, typed=True)
def Get_CT_Unannoted_OUTPUT(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(EXTERN_VAR.CACHE_PATH_UNANNOTED_OUTPUT, index_str)
    return joblib.load(cache_file)

@functools.lru_cache(maxsize=EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED, typed=True)
def Get_CT_Unannoted_RAW(index):
    """从内存缓存获取"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(EXTERN_VAR.CACHE_PATH_UNANNOTED_RAW, index_str)
    return joblib.load(cache_file)

def Flush_All_CT_Data_To_Mem():
    """将所有缓存刷入内存中"""
    print("\nFlush_All_CT_Data_To_Mem:")
    counter = DynamicCounter(EXTERN_VAR.CACHE_MAXSIZE_ANNOTED, "CT_Annoted_INPUT", 100)
    for i in range(EXTERN_VAR.CACHE_MAXSIZE_ANNOTED):
        Get_CT_Annoted_INPUT(i)
        counter.increment()
    counter = DynamicCounter(EXTERN_VAR.CACHE_MAXSIZE_ANNOTED, "CT_Annoted_LABEL", 100)
    for i in range(EXTERN_VAR.CACHE_MAXSIZE_ANNOTED):
        Get_CT_Annoted_LABEL(i)
        counter.increment()
    counter = DynamicCounter(EXTERN_VAR.CACHE_MAXSIZE_ANNOTED, "CT_Annoted_RAW", 100)
    for i in range(EXTERN_VAR.CACHE_MAXSIZE_ANNOTED):
        Get_CT_Annoted_RAW(i)
        counter.increment()
    counter = DynamicCounter(EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED, "CT_Unannoted_OUTPUT", 100)
    for i in range(EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED):
        Get_CT_Unannoted_OUTPUT(i)
        counter.increment()
    # counter = DynamicCounter(EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED, "CT_Unannoted_RAW", 100)
    # for i in range(EXTERN_VAR.CACHE_MAXSIZE_UNANNOTED):
    #     Get_CT_Unannoted_RAW(i)
    #     counter.increment()
    print()


def Save_CT_Candidate(index, path, data, *arg, **kwargs):
    """存储可用CT数据的"""
    index_str = "{:06}".format(index)
    cache_file = os.path.join(path, index_str)
    joblib.dump(data, cache_file)
