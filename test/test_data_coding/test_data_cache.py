from data_coding.data_cache import *


def test_data_cache():
    """测试cache功能是否正常"""
    for i in range(3):
        expensive_function(2, 3)
    num_code = 0
    for i in range(3):
        num_code += 5
        expensive_function(2, 3, cache_num_code=num_code)


@disk_cache("dataset/Cache/test")
def expensive_function(a, b):
    """假设这是一个计算量大且耗时的函数"""
    print("Performing expensive computation...")
    return a + b



