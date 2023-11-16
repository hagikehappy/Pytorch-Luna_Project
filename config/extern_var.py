"""全局变量区"""

from config.settings import Settings

# ## 经验上一般厚度不超过5
# SLICES_THICKNESS = 3
# SLICES_THICKNESS_HALF = int(SLICES_THICKNESS / 2)
# SLICES_INIT_X_LENGTH = 512
# SLICES_INIT_Y_LENGTH = 512
#
# ## 中心化裁剪，按此尺寸可最大裁剪且完全保留原有信息
# SLICES_X_CROP = 34
# SLICES_Y_CROP = 34
# SLICES_CROP_X_LENGTH = SLICES_INIT_X_LENGTH - SLICES_X_CROP * 2     ## 444
# SLICES_CROP_Y_LENGTH = SLICES_INIT_Y_LENGTH - SLICES_Y_CROP * 2     ## 444
#
# ## UNET输出的大小，按该尺寸可覆盖最大结节情况
# SLICES_X_OUTPUT_LENGTH = 64
# SLICES_X_OUTPUT_LENGTH_HALF = int(SLICES_X_OUTPUT_LENGTH / 2)
# SLICES_Y_OUTPUT_LENGTH = 64
# SLICES_Y_OUTPUT_LENGTH_HALF = int(SLICES_Y_OUTPUT_LENGTH / 2)
#
# ## 制作UNET标注信息时向外扩展的diameter倍数
# UNET_DIAMETER_EXPANSION = 0
# ## 该THRESHOLD针对(-1, 1)范围
# UNET_LOW_THRESHOLD = -0.3
# UNET_LOW_THRESHOLD_RATE = 0.15
# # UNET_HIGH_THRESHOLD = 0.4
#
# ## 保存非标注数据占原始数据的数量比
# CACHE_UNANNOTED_DATA_RATE = 100

## 内存中缓存数量的最大限制
settings = Settings(update=True)
# CACHE_MAXSIZE_ANNOTED = settings.config_dict['annotated_data_num']
# CACHE_MAXSIZE_UNANNOTED = settings.config_dict['unannotated_data_num']
# 
# ## 内存中缓存的位置
# CACHE_PATH_DATA_FOR_UNET = "dataset/Cache/data_for_unet/"
# CACHE_PATH_ANNOTED = CACHE_PATH_DATA_FOR_UNET + "annotated_slices/"
# CACHE_PATH_ANNOTED_INPUT = CACHE_PATH_ANNOTED + "input/"
# CACHE_PATH_ANNOTED_LABEL = CACHE_PATH_ANNOTED + "label/"
# CACHE_PATH_ANNOTED_RAW = CACHE_PATH_ANNOTED + "raw/"
# CACHE_PATH_UNANNOTED = CACHE_PATH_DATA_FOR_UNET + "unannotated_slices/"
# CACHE_PATH_UNANNOTED_OUTPUT = CACHE_PATH_UNANNOTED + "output/"
# CACHE_PATH_UNANNOTED_RAW = CACHE_PATH_UNANNOTED + "raw/"
