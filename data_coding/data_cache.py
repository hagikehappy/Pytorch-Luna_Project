import os
import functools
import joblib


# @functools.lru_cache(maxsize=1, typed=True)
"""函数的内存缓存装饰器，其中maxsize代表缓存量的大小；typed对不同类型的函数参数进行单独缓存。"""


def disk_cache(directory):
    """函数的磁盘缓存装饰器"""
    def decorator(func):
        func_name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 构建缓存文件的路径
            if 'cache_num_code' in kwargs:
                num_code = kwargs.pop('cache_num_code')
            else:
                num_code = 0
            cache_file = os.path.join(directory,
                                      f"{func_name + '_' + str(num_code) + '_' + str(args) + str(kwargs)}.pkl")
            # 检查缓存文件是否存在
            if os.path.exists(cache_file):
                # 如果缓存文件存在，从文件中加载结果
                # print(f"Loading {func_name} from disk cache...")
                result = joblib.load(cache_file)
            else:
                # 如果缓存文件不存在，调用原始函数并保存结果到缓存文件
                # print(f"Computing {func_name} and saving to disk cache...")
                result = func(*args, **kwargs)
                joblib.dump(result, cache_file)
            return result
        return wrapper
    return decorator

