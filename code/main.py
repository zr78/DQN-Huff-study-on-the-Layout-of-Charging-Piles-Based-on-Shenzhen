from DQNtrain_model import train
import os
import torch

import matplotlib.pyplot as plt
# 设置 matplotlib 正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体，可根据系统情况修改
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# # CUDA相关优化
# if torch.cuda.is_available():
#     # 设置GPU内存分配策略
#     torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用率，避免OOM
#     # 设置cudnn
#     torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    # 设置数据和配置文件路径
    data_path = "data"    # 您的数据文件夹
    output_path = "output"  # 可以定义一个默认输出路径
    config_path = "config.json"  # 您的配置文件
    
    trained_agent = train(data_path, output_path, config_path)

