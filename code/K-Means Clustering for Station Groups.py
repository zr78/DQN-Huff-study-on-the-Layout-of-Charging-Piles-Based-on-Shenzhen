'''
聚类分析图说明
1. 聚类目标：
该图展示了基于 充电桩数量（Charging Piles） 对不同 DQN训练轮数（Episode） 的站点进行的 K-Means 聚类 结果。每个子图表示一个训练轮数（不同的训练阶段），聚类的颜色代表不同的聚类组。

2. 数据特征：
经度（Longitude）和纬度（Latitude）：这些地理坐标用于确定站点的地理位置。

充电桩数量（Charging Piles）：用于衡量每个站点的服务能力。不同的充电桩数量反映了站点的规模和需求。

3. 聚类分析：
K-Means 聚类算法：使用 K-Means 算法将站点划分为 5 个聚类组，根据站点的地理位置和充电桩数量进行聚类。

颜色表示：每个聚类组通过不同的颜色表示，颜色的变化代表不同的聚类组。

4. DQN训练轮数：
每个子图对应不同的 DQN训练轮数（Episode），这些轮数反映了训练过程中站点分布和充电桩数量随时间的演变情况。时间点包括 0, 60, 120, 180, 240, 299。这些不同的训练阶段能够帮助分析在训练过程中，站点的地理分布是否发生变化。

5. 分析目的：
聚类变化：观察每个训练轮次下的聚类结果，分析聚类是否随着训练轮数的增加发生变化。特别是查看某些站点是否在不同的轮次中被分配到不同的聚类组。

站点群体演化：分析站点随着 DQN 训练的演变，是否能够逐渐识别出不同类型的站点群体，并且这些群体的地理分布和充电桩数量是否有显著变化。

6. 可视化目的：
该图通过可视化不同 DQN 训练轮数下的聚类结果，帮助我们理解 站点的地理分布 和 充电桩数量 是如何在训练过程中变化的。

此图有助于评估 DQN 训练模型在训练过程中对站点分布的识别和学习能力，进而评估训练是否有效、聚类是否在训练后有所变化。
'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# CSV 文件路径列表
files = [
    'output/station_summary_ep0.csv',
    'output/station_summary_ep60.csv',
    'output/station_summary_ep120.csv',
    'output/station_summary_ep180.csv',
    'output/station_summary_ep240.csv',
    'output/station_summary_ep299.csv'
]

# 读取和合并数据
data_combined = pd.concat([pd.read_csv(file) for file in files], 
                          keys=[0, 60, 120, 180, 240, 299], names=["Episode", "Index"])

# 选择聚类相关的列
charging_data = data_combined[['longitude', 'latitude', 'charging_piles']]

# 初始化绘图，调整图像大小
fig, axs = plt.subplots(2, 3, figsize=(21, 10))  # 增加图像的宽度和高度，确保清晰显示
time_points = [0, 60, 120, 180, 240, 299]

# 聚类并绘图
for i, t in enumerate(time_points):
    # 提取每个时间点的数据
    episode_data = charging_data.loc[t]
    
    # 进行 K-Means 聚类
    kmeans = KMeans(n_clusters=5, random_state=42)
    episode_data['Cluster'] = kmeans.fit_predict(episode_data[['longitude', 'latitude', 'charging_piles']])
    
    # 绘制图像
    ax = axs[i // 3, i % 3]
    scatter = ax.scatter(episode_data['longitude'], episode_data['latitude'], c=episode_data['Cluster'], cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
    ax.set_title(f"Charging Piles Clustering - Episode {t}", fontsize=14)  # 设置标题字体大小
    ax.set_xlabel("Longitude", fontsize=12)  # 设置坐标轴标签字体大小
    ax.set_ylabel("Latitude", fontsize=12)  # 设置坐标轴标签字体大小
    fig.colorbar(scatter, ax=ax, label="Cluster", shrink=0.8)  # 调整 colorbar 大小，使其不会占用过多空间

# 调整子图布局，增加子图间的间距，避免重叠
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# 调整布局并显示图像
plt.tight_layout()
plt.show()
