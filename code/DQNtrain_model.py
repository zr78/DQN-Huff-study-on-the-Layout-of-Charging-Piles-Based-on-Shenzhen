'''
author:zhanguri
date:2023-11-24
version:0.2
bug:
    1.不知道什么原因停止了训练
'''
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import math
import json
import glob
import time
import math 
import pickle  
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合没有GUI的环境
import matplotlib.pyplot as plt


from Huff import DynamicHuffEV, UrbanEVDataLoader
from expert_knowledge import ExpertKnowledge

# 设置 matplotlib 正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体，可根据系统情况修改
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

#############################################
# 1. 数据加载与处理
#############################################
class DataLoader:
    @staticmethod
    def load_charging_station_data(data_path="data"):
        """
        加载充电站相关数据
        
        预期的数据文件:
        - station_inf.csv: 包含充电站位置、初始容量等信息
        """
        try:
            # 首先加载吸引力数据，这样我们只保留有吸引力数据的站点
            attractions_path = os.path.join(data_path, "station_attractions.csv")
            if os.path.exists(attractions_path):
                attractions_df = pd.read_csv(attractions_path)
                print(f"加载站点吸引力数据: {len(attractions_df)}条记录")
                
                # 检查是否包含必要的列
                if 'station_id' not in attractions_df.columns or 'attraction' not in attractions_df.columns:
                    print("警告: 吸引力数据缺少必要的列(station_id或attraction)，将使用默认值")
                    valid_station_ids = None
                else:
                    # 获取有效的站点ID列表
                    valid_station_ids = attractions_df['station_id'].unique().tolist()
                    print(f"找到{len(valid_station_ids)}个有吸引力数据的站点ID")
            else:
                print(f"警告: 未找到站点吸引力数据文件 {attractions_path}")
                valid_station_ids = None
            
            # 加载充电站数据
            stations_path = os.path.join(data_path, "station_inf.csv")
            if os.path.exists(stations_path):
                stations_df = pd.read_csv(stations_path)
                original_count = len(stations_df)
                print(f"原始充电站数据: {original_count}个站点")
                
                # 如果有有效的站点ID列表，只保留这些站点
                if valid_station_ids is not None and 'station_id' in stations_df.columns:
                    stations_df = stations_df[stations_df['station_id'].isin(valid_station_ids)]
                    filtered_count = len(stations_df)
                    print(f"过滤后的充电站数据: {filtered_count}个站点，移除了{original_count-filtered_count}个无匹配吸引力数据的站点")
                
                # 从原始数据提取经纬度坐标作为x和y
                if 'longitude' in stations_df.columns and 'latitude' in stations_df.columns:
                    stations_df['x'] = stations_df['longitude']
                    stations_df['y'] = stations_df['latitude']
                
                # 确保有充电桩容量信息
                if 'capacity' not in stations_df.columns and 'maximum_power' in stations_df.columns:
                    # 根据最大功率估算容量
                    stations_df['capacity'] = (stations_df['maximum_power'] / 10).astype(int)
                    stations_df['capacity'] = stations_df['capacity'].apply(lambda x: max(x, 5))  # 确保至少有5个充电桩
                
                # 提取站点位置和初始容量
                station_positions = stations_df[['x', 'y']].values
                initial_capacities = stations_df['capacity'].values if 'capacity' in stations_df.columns else None
                
                # 初始充电桩数量设为容量的一半
                if 'capacity' in stations_df.columns:
                    stations_df['piles'] = (stations_df['capacity'] * 0.5).astype(int)

                # 提取其他可能的站点属性
                station_attrs = {}
                for col in stations_df.columns:
                    if col not in ['x', 'y', 'capacity'] and not pd.isna(stations_df[col]).all():
                        station_attrs[col] = stations_df[col].values
                
                # 加载站点吸引力数据并与筛选后的站点匹配
                if valid_station_ids is not None:
                    # 将站点吸引力数据合并到站点数据中
                    if 'station_id' in stations_df.columns and 'station_id' in attractions_df.columns:
                        # 使用左连接合并，保证所有站点都有吸引力值
                        stations_df = pd.merge(stations_df, attractions_df[['station_id', 'attraction']], 
                                              on='station_id', how='left')
                        
                        # 确保所有站点都有吸引力值
                        if stations_df['attraction'].isnull().any():
                            # 对缺失值填充平均值
                            mean_attr = attractions_df['attraction'].mean()
                            stations_df['attraction'].fillna(mean_attr, inplace=True)
                    
                    # 将吸引力添加到属性字典
                    station_attrs['attraction'] = stations_df['attraction'].values
                    print(f"站点吸引力范围: {stations_df['attraction'].min():.2f} - {stations_df['attraction'].max():.2f}")
                else:
                    print("警告: 无法获取有效的站点ID列表，将使用默认吸引力值")
                    stations_df['attraction'] = 1.0
                    station_attrs['attraction'] = np.ones(len(stations_df))
                
                return {
                    'positions': station_positions,
                    'capacities': initial_capacities,
                    'attributes': station_attrs,
                    'count': len(stations_df),
                    'dataframe': stations_df  # 保留原始DataFrame以便进一步处理
                }
            else:
                print(f"警告: 未找到充电站数据文件 {stations_path}, 将使用随机生成数据")
                return None
        
        except Exception as e:
            print(f"加载充电站数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def load_traffic_data(data_path="data"):
        """加载交通网络数据"""
        try:
            # 尝试加载POI数据作为路网节点
            poi_path = os.path.join(data_path, "poi.csv")
            traffic_path = os.path.join(data_path, "深圳市交通运输局街道实时数据.csv")
            
            # 加载POI数据作为路网节点
            if os.path.exists(poi_path):
                poi_df = pd.read_csv(poi_path)
                print(f"成功加载POI数据: {poi_df.shape}")
                
                # 处理POI数据作为路网节点
                if 'longitude' in poi_df.columns and 'latitude' in poi_df.columns:
                    # 选择部分POI点作为路网节点
                    sample_size = min(50, len(poi_df))  # 取50个样本点
                    sampled_poi = poi_df.sample(sample_size)
                    road_nodes = sampled_poi[['longitude', 'latitude']].values
                    
                    # 创建简化的边结构
                    edges = []
                    for i in range(len(road_nodes)):
                        for j in range(i+1, len(road_nodes)):
                            if j < i + 5:  # 只连接相邻几个节点，避免全连接
                                # 计算两点间距离
                                dist = np.sqrt(np.sum((road_nodes[i] - road_nodes[j])**2))
                                edges.append([i, j, dist])
                    
                    print(f"从POI数据生成路网: {len(road_nodes)}个节点, {len(edges)}条边")
                    return {
                        'nodes': road_nodes,
                        'edges': np.array(edges)
                    }
            
            # 尝试加载深圳交通数据
            if os.path.exists(traffic_path):
                traffic_df = pd.read_csv(traffic_path)
                print(f"成功加载交通数据: {traffic_df.shape}")
                
                # 处理交通数据
                if 'locsn' in traffic_df.columns and 'locsw' in traffic_df.columns:
                    # 提取位置信息
                    locs = []
                    for _, row in traffic_df.iterrows():
                        try:
                            # 假设格式为 "纬度,经度"
                            lat, lon = map(float, row['locsn'].split(','))
                            locs.append([lon, lat])
                        except:
                            continue
                    
                    if locs:
                        road_nodes = np.array(locs)
                        print(f"从交通数据提取了{len(road_nodes)}个路网节点")
                        
                        # 创建简化的边结构
                        edges = []
                        for i in range(len(road_nodes)):
                            for j in range(i+1, min(i+5, len(road_nodes))):
                                dist = np.sqrt(np.sum((road_nodes[i] - road_nodes[j])**2))
                                edges.append([i, j, dist])
                        
                        return {
                            'nodes': road_nodes,
                            'edges': np.array(edges)
                        }
            
            print("警告: 未找到有效的交通数据，将使用随机生成数据")
            return None
                
        except Exception as e:
            print(f"加载交通数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def load_demand_data(data_path="data"):
        """加载需求数据
        
        从POI数据或站点数据生成模拟需求
        """
        stations_df = pd.read_csv(f"{data_path}/station_inf.csv")
        num_stations = len(stations_df)
        demand_by_time = {}
        for hour in range(24):
            # 根据 POI 或站点特征生成需求，而不是全 0
            station_demands = np.random.uniform(1, 10, size=num_stations)  # 示例：随机值
            demand_by_time[hour] = station_demands
        print(f"Generated demand data for {num_stations} stations")
        return demand_by_time
        
    @staticmethod
    def load_config(config_path="config.json"):
        """加载配置参数"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"成功加载配置文件: {config_path}")
                return config
            else:
                print(f"警告: 未找到配置文件 {config_path}，将使用默认参数")
                return {}
        except Exception as e:
            print(f"加载配置文件时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}

#############################################
# 2. Dueling DQN网络
#############################################
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DuelingDQN, self).__init__()
        # 共享特征层 - 使用LayerNorm避免BatchNorm在单样本时的问题
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # 价值流和优势流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

#############################################
# 3. Huff模型实现
#############################################
def huff_model(station_locations, attractiveness, target_points, gamma=2.0):
    """
    计算Huff模型下每个目标点选择各充电站的概率
    
    参数:
    - station_locations: 充电站位置 (n_stations, 2)
    - attractiveness: 充电站吸引力 (n_stations,)
    - target_points: 目标点位置 (n_points, 2)
    - gamma: 距离衰减指数
    
    返回:
    - probabilities: 每个目标点选择各充电站的概率 (n_points, n_stations)
    """
    n_stations = len(station_locations)
    n_points = len(target_points)
    
    # 初始化概率矩阵
    probabilities = np.zeros((n_points, n_stations))
    
    for i, point in enumerate(target_points):
        # 计算目标点到各充电站的距离
        distances = np.array([np.linalg.norm(point - loc) for loc in station_locations])
        distances = np.maximum(distances, 1e-5)  # 避免除零错误
        
        # 计算吸引力/距离^gamma
        utility = attractiveness / (distances ** gamma)
        
        # 计算选择概率
        total_utility = np.sum(utility)
        if total_utility > 0:
            probabilities[i] = utility / total_utility
        else:
            # 如果总吸引力为0，则均等概率
            probabilities[i] = 1.0 / n_stations
            
    return probabilities

#############################################
# 4. 充电站环境类
#############################################
class DynamicChargingEnv:
    def __init__(self, num_stations=None, area_size=1000, data_path="data"):
        self.area_size = area_size
        self.time_step = 0
        self.data_path = data_path  # 添加这一行以确保data_path被保存为类属性
        
        # Load data using UrbanEVDataLoader from Huff.py
        self.data_loader = UrbanEVDataLoader(data_path)
        self.data_loader.load_datasets().preprocess().add_temporal_features().create_spatiotemporal_features()
        self.huff_model = DynamicHuffEV(self.data_loader)

        # 加载充电站数据
        try:
            # 加载充电站数据 - 从data_path目录读取
            station_data = pd.read_csv(os.path.join(data_path, "station_inf.csv"))
            
            if len(station_data) > 0:
                print(f"已加载{len(station_data)}个充电站位置数据")

                if 'longitude' in station_data.columns and 'latitude' in station_data.columns:
                    station_data['x'] = station_data['longitude']
                    station_data['y'] = station_data['latitude']

                self.station_pos = station_data[['x', 'y']].values
                self.num_stations = len(self.station_pos)
                self.station_capacity = station_data['capacity'].values if 'capacity' in station_data.columns else np.random.randint(5, 10, size=self.num_stations)
                self.init_piles = station_data['piles'].values if 'piles' in station_data.columns else np.random.randint(0, 5, size=self.num_stations)
                self.charging_piles = self.init_piles.copy()
                
                # 存储站点ID，用于后续匹配
                self.station_ids = station_data['station_id'].values if 'station_id' in station_data.columns else np.arange(1001, 1001 + self.num_stations)

                # 加载POI数据（用于生成路网）
                try:
                    # 尝试多个可能的文件名
                    poi_file_options = [
                        os.path.join(data_path, "poi_data.csv"),
                        os.path.join(data_path, "poi.csv"),
                        "poi_data.csv",
                        "poi.csv",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "poi.csv")
                    ]
                    
                    poi_data = None
                    used_path = None
                    
                    for file_path in poi_file_options:
                        if os.path.exists(file_path):
                            poi_data = pd.read_csv(file_path)
                            used_path = file_path
                            break
                    
                    if poi_data is not None:
                        print(f"成功从 {used_path} 加载POI数据: {poi_data.shape}")
                        
                        # 确保POI数据包含必要的列
                        required_cols = ['x', 'y', 'type']
                        missing_cols = [col for col in required_cols if col not in poi_data.columns]
                        
                        if missing_cols:
                            print(f"POI数据缺少必要的列: {missing_cols}")
                            
                            # 尝试映射现有列到所需列
                            column_mapping = {}
                            
                            # 映射坐标列 - 特别关注longitude/latitude
                            if 'x' in missing_cols:
                                if 'longitude' in poi_data.columns:
                                    column_mapping['x'] = 'longitude'
                                    print("已将'longitude'列映射为'x'")
                                    missing_cols.remove('x')
                                else:
                                    for col in ['lon', 'lng', 'x_coord']:
                                        if col in poi_data.columns:
                                            column_mapping['x'] = col
                                            missing_cols.remove('x')
                                            break
                            
                            if 'y' in missing_cols:
                                if 'latitude' in poi_data.columns:
                                    column_mapping['y'] = 'latitude'
                                    print("已将'latitude'列映射为'y'")
                                    missing_cols.remove('y')
                                else:
                                    for col in ['lat', 'y_coord']:
                                        if col in poi_data.columns:
                                            column_mapping['y'] = col
                                            missing_cols.remove('y')
                                            break
                            
                            # 特别处理primary_types
                            if 'type' in missing_cols:
                                if 'primary_types' in poi_data.columns:
                                    # 将文本类型转换为数值类型
                                    type_mapping = {}
                                    unique_types = poi_data['primary_types'].unique()
                                    for i, t in enumerate(unique_types):
                                        type_mapping[t] = i + 1
                                    
                                    poi_data['type'] = poi_data['primary_types'].map(type_mapping)
                                    print("已将'primary_types'列映射为'type'")
                                    missing_cols.remove('type')
                                else:
                                    # 尝试其他可能的类型列
                                    for col in ['category', 'poi_type', 'function', 'class']:
                                        if col in poi_data.columns:
                                            column_mapping['type'] = col
                                            missing_cols.remove('type')
                                            break
                            
                            # 应用列映射
                            for target, source in column_mapping.items():
                                poi_data[target] = poi_data[source]
                            
                            # 如果仍有缺失列，创建默认值
                            if 'x' in missing_cols:
                                poi_data['x'] = np.random.uniform(0, self.area_size, len(poi_data))
                                print("未找到X坐标列，创建随机X坐标")
                                missing_cols.remove('x')
                                
                            if 'y' in missing_cols:
                                poi_data['y'] = np.random.uniform(0, self.area_size, len(poi_data))
                                print("未找到Y坐标列，创建随机Y坐标")
                                missing_cols.remove('y')
                                
                            if 'type' in missing_cols:
                                poi_data['type'] = 1  # 默认类型
                                print("已为POI数据添加默认类型")
                                missing_cols.remove('type')
                            
                            # 使用处理后的POI数据
                            self.poi_data = poi_data[['x', 'y', 'type']].values
                            
                            # 使用POI数据生成简化路网
                            num_nodes = min(50, len(self.poi_data))
                            indices = np.random.choice(len(self.poi_data), num_nodes, replace=False)
                            self.road_nodes = self.poi_data[indices, :2]  # 只取x,y坐标
                            
                            # 生成简单的路网连接
                            self.road_edges = []
                            for i in range(num_nodes):
                                # 为每个节点连接到最近的几个节点
                                distances = [np.linalg.norm(self.road_nodes[i] - self.road_nodes[j]) for j in range(num_nodes) if i != j]
                                nearest_indices = np.argsort(distances)[:4]  # 连接到最近的4个节点
                                for j in nearest_indices:
                                    if j != i:  # 避免自连接
                                        self.road_edges.append((i, j))
                            
                            print(f"从POI数据生成路网: {len(self.road_nodes)}个节点, {len(self.road_edges)}条边")
                        else:
                            # 如果已经有所需的列，直接使用
                            self.poi_data = poi_data[['x', 'y', 'type']].values
                            
                            # 使用POI数据生成路网
                            num_nodes = min(50, len(self.poi_data))
                            indices = np.random.choice(len(self.poi_data), num_nodes, replace=False)
                            self.road_nodes = self.poi_data[indices, :2]
                            
                            # 生成路网连接
                            self.road_edges = []
                            for i in range(num_nodes):
                                distances = [np.linalg.norm(self.road_nodes[i] - self.road_nodes[j]) for j in range(num_nodes) if i != j]
                                nearest_indices = np.argsort(distances)[:4]
                                for j in nearest_indices:
                                    if j != i:
                                        self.road_edges.append((i, j))
                            
                            print(f"从POI数据生成路网: {len(self.road_nodes)}个节点, {len(self.road_edges)}条边")
                    else:
                        print("未找到POI数据文件，将使用随机生成的路网")
                        self.road_nodes = np.random.rand(50, 2) * self.area_size
                        self.road_edges = []
                        for i in range(len(self.road_nodes)):
                            for j in range(i+1, min(i+5, len(self.road_nodes))):
                                self.road_edges.append((i, j))
                except Exception as e:
                    print(f"加载POI数据异常: {e}，将使用随机生成的路网")
                    self.road_nodes = np.random.rand(50, 2) * self.area_size
                    self.road_edges = []
                    for i in range(len(self.road_nodes)):
                        for j in range(i+1, min(i+5, len(self.road_nodes))):
                            self.road_edges.append((i, j))


                # 加载需求数据
                self.demand_data = DataLoader.load_demand_data(data_path)
                if self.demand_data:
                    print(f"成功加载需求数据")
                else:
                    print("警告: 未加载到需求数据，将使用随机生成数据")
            else:
                # 如果充电站数据为空，进入默认数据生成流程
                print("充电站数据为空，将使用随机生成数据")
                self.num_stations = num_stations if num_stations is not None else 10
                self._generate_world()
        except Exception as e:
            print(f"加载数据失败: {e}，将使用随机生成数据")
            # 数据加载失败时的默认值
            self.num_stations = num_stations if num_stations is not None else 10
            self._generate_world()
        
        # 设置时间戳
        if not hasattr(self, 'timestamps'):
            if hasattr(self.data_loader, 'timestamps') and self.data_loader.timestamps is not None:
                self.timestamps = self.data_loader.timestamps
            else:
                print("警告: 未加载到时间戳数据，将使用默认时间戳")
                self.timestamps = pd.date_range(start='2023-01-01', periods=240, freq='6min').to_pydatetime().tolist()
    

        # 加载站点吸引力数据
        attractions_file_options = [
            os.path.join(data_path, "station_attractions.csv"),
            "station_attractions.csv",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "station_attractions.csv")
        ]

        attractions_loaded = False
        for file_path in attractions_file_options:
            if os.path.exists(file_path):
                print(f"尝试从 {file_path} 加载站点吸引力数据...")
                if self.load_attraction_data(file_path):
                    attractions_loaded = True
                    break

        if not attractions_loaded:
            print("未找到站点吸引力数据文件，将使用Huff模型生成吸引力")
            self.generate_attractions_from_huff()

        # 加载选择概率数据
        probs_file_options = [
            os.path.join(data_path, "selection_probabilities.csv"),
            "selection_probabilities.csv",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "selection_probabilities.csv")
        ]

        probs_loaded = False
        for file_path in probs_file_options:
            if os.path.exists(file_path):
                print(f"尝试从 {file_path} 加载选择概率数据...")
                if self.load_selection_probabilities(file_path):
                    probs_loaded = True
                    break

        if not probs_loaded:
            print("未找到选择概率数据文件，将使用默认模型生成")
            # 这里可以添加默认的选择概率生成方法，或者直接跳过


        
    def reset(self):
        """
        重置环境状态，可选择使用分析报告数据优化初始部署
        
        Returns:
            numpy.ndarray: 初始化后的状态向量
        """
        # 重置当前时间步
        self.current_step = 0
        
        # 使用分析报告中的数据来初始化充电桩部署
        if hasattr(self, 'use_analysis_data') and self.use_analysis_data and random.random() < 0.3:  # 30%概率使用分析建议的初始状态
            try:
                # 获取站点ID列表
                zone_ids = self.data_loader.zones['zone_id'].values[:self.num_stations]
                
                for i, zone_id in enumerate(zone_ids):
                    layout_data = self.station_layout_analysis[self.station_layout_analysis['station_id'] == zone_id]
                    if not layout_data.empty and 'recommended_piles' in layout_data.columns:
                        recommended = int(layout_data['recommended_piles'].values[0])
                        # 确保在容量范围内
                        self.charging_piles[i] = min(recommended, self.station_capacity[i])
                
                # 调整以确保总数符合要求
                while np.sum(self.charging_piles) > self.total_piles:
                    idx = np.argmax(self.charging_piles)
                    if self.charging_piles[idx] > 0:
                        self.charging_piles[idx] -= 1
                
                while np.sum(self.charging_piles) < self.total_piles:
                    idx = np.argmin(self.charging_piles / np.maximum(self.station_capacity, 1))
                    if self.charging_piles[idx] < self.station_capacity[idx]:
                        self.charging_piles[idx] += 1
                
                print("已使用分析报告数据进行初始化")
            except Exception as e:
                print(f"使用分析数据初始化失败: {e}")
                # 回退到随机初始化
                self._random_initialize_piles()
        else:
            # 原有的随机初始化
            self._random_initialize_piles()
        
        # 重置其他状态变量
        if hasattr(self, 'huff_model'):
            self._calculate_huff_model()
        
        # 返回初始状态
        return self._get_state()
    
    def load_attraction_data(self, attractions_path):
        """
        加载站点吸引力数据
        
        参数:
        attractions_path (str): 吸引力数据文件路径
        """
        try:
            # 尝试加载CSV文件
            attraction_data = pd.read_csv(attractions_path)
            
            # 检查列名是否匹配
            if 'zone_id' in attraction_data.columns and 'attraction' in attraction_data.columns:
                print(f"成功加载站点吸引力数据: {attraction_data.shape[0]}个站点")
                
                # 创建站点ID到吸引力的映射
                attraction_dict = dict(zip(attraction_data['zone_id'], attraction_data['attraction']))
                
                # 为每个站点分配吸引力
                self.attractions = np.zeros(self.num_stations)
                self.station_attractions = np.zeros(self.num_stations)  # 同时创建两个属性名以保持兼容性
                
                for i, station_id in enumerate(self.station_ids):
                    if station_id in attraction_dict:
                        self.attractions[i] = attraction_dict[station_id]
                        self.station_attractions[i] = attraction_dict[station_id]
                    else:
                        print(f"警告: 站点ID {station_id} 没有吸引力数据")
                        # 使用平均值或合理的默认值
                        self.attractions[i] = np.mean(list(attraction_dict.values()))
                        self.station_attractions[i] = self.attractions[i]
                
                print(f"吸引力数据统计: 最小值={np.min(self.attractions):.2f}, 最大值={np.max(self.attractions):.2f}, 平均值={np.mean(self.attractions):.2f}")
                return True
            else:
                # 如果列名不匹配，尝试检测正确的列名
                possible_id_cols = [col for col in attraction_data.columns if 'id' in col.lower() or 'zone' in col.lower() or 'station' in col.lower()]
                possible_attr_cols = [col for col in attraction_data.columns if 'attr' in col.lower() or 'score' in col.lower() or 'value' in col.lower()]
                
                if possible_id_cols and possible_attr_cols:
                    id_col = possible_id_cols[0]
                    attr_col = possible_attr_cols[0]
                    print(f"使用检测到的列名: {id_col} -> zone_id, {attr_col} -> attraction")
                    
                    # 创建站点ID到吸引力的映射
                    attraction_dict = dict(zip(attraction_data[id_col], attraction_data[attr_col]))
                    
                    # 为每个站点分配吸引力
                    self.attractions = np.zeros(self.num_stations)
                    self.station_attractions = np.zeros(self.num_stations)
                    
                    for i, station_id in enumerate(self.station_ids):
                        if station_id in attraction_dict:
                            self.attractions[i] = attraction_dict[station_id]
                            self.station_attractions[i] = attraction_dict[station_id]
                        else:
                            self.attractions[i] = np.mean(list(attraction_dict.values()))
                            self.station_attractions[i] = self.attractions[i]
                    
                    return True
                else:
                    print(f"吸引力数据文件格式不匹配。预期列: zone_id, attraction，实际列: {attraction_data.columns.tolist()}")
                    return False
        except Exception as e:
            print(f"加载吸引力数据时出错: {e}")
            return False

    def load_selection_probabilities(self, probs_path):
        """
        加载选择概率数据
        
        参数:
        probs_path (str): 选择概率数据文件路径
        """
        try:
            # 尝试加载CSV文件
            # 注意：选择概率数据似乎是带行索引的宽格式表格
            selection_probs = pd.read_csv(probs_path, index_col=0)
            
            print(f"成功加载选择概率数据: {selection_probs.shape}，前5个站点IDs: {list(selection_probs.columns[:5])}")
            
            # 将选择概率矩阵设置为类属性
            self.selection_probabilities = selection_probs
            
            # 还需要将数据转换为P_ij和C_ij格式以供使用
            # P_ij是选择概率，C_ij可以从选择概率推导阻抗
            station_ids = self.station_ids
            n_stations = len(station_ids)
            
            # 初始化P_ij和C_ij数组
            self.P_ij = []
            self.C_ij = []
            
            # 对每对站点计算选择概率和阻抗
            for i in range(n_stations):
                origin_id = station_ids[i]
                if str(origin_id) in selection_probs.index:
                    for j in range(n_stations):
                        dest_id = station_ids[j]
                        if str(dest_id) in selection_probs.columns:
                            try:
                                # 获取从origin到dest的选择概率
                                p_ij = selection_probs.loc[str(origin_id), str(dest_id)]
                                
                                # 从选择概率推导阻抗 (简单模型：C_ij = 1/P_ij)
                                c_ij = 1.0 / (p_ij + 1e-5)  # 避免除零错误
                                
                                self.P_ij.append(p_ij)
                                self.C_ij.append(c_ij)
                            except:
                                # 如果无法获取概率，使用默认值
                                self.P_ij.append(0.01)
                                self.C_ij.append(100.0)
                        else:
                            # 如果目的地ID不在列中
                            self.P_ij.append(0.01)
                            self.C_ij.append(100.0)
                else:
                    # 如果起源ID不在索引中
                    for j in range(n_stations):
                        self.P_ij.append(0.01)
                        self.C_ij.append(100.0)
            
            print(f"成功生成选择概率(P_ij)和阻抗(C_ij)数据: {len(self.P_ij)}个元素")
            return True
        except Exception as e:
            print(f"加载选择概率数据时出错: {e}")
            # 创建一些默认值
            self.P_ij = [0.01] * (self.num_stations * self.num_stations)
            self.C_ij = [100.0] * (self.num_stations * self.num_stations)
            return False

        
    def load_attraction_data(self, attractions_path):
        """
        加载站点吸引力数据
        
        参数:
        attractions_path (str): 吸引力数据文件路径
        """
        try:
            attraction_data = pd.read_csv(attractions_path)
            
            # 确保数据包含必要的列
            if 'station_id' in attraction_data.columns and 'attraction' in attraction_data.columns:
                # 创建站点ID到吸引力的映射
                attraction_dict = dict(zip(attraction_data['station_id'], attraction_data['attraction']))
                
                # 为每个站点分配吸引力
                self.attractions = np.zeros(self.num_stations)
                for i, station_id in enumerate(self.station_ids):
                    if station_id in attraction_dict:
                        self.attractions[i] = attraction_dict[station_id]
                    else:
                        # 如果站点没有吸引力数据，设置默认值
                        self.attractions[i] = 1.0
                
                print(f"成功加载{len(attraction_data)}个站点的吸引力数据")
                return True
            else:
                print("吸引力数据文件缺少必要的列：station_id, attraction")
                return False
        except Exception as e:
            print(f"加载吸引力数据时出错: {e}")
            return False

    def load_attraction_data(self, attractions_path):
        """
        加载站点吸引力数据
        
        参数:
        attractions_path (str): 吸引力数据文件路径
        """
        try:
            attraction_data = pd.read_csv(attractions_path)
            
            # 检查列名是否为zone_id和attraction
            if 'zone_id' in attraction_data.columns and 'attraction' in attraction_data.columns:
                # 创建站点ID到吸引力的映射
                attraction_dict = dict(zip(attraction_data['zone_id'], attraction_data['attraction']))
                
                # 为每个站点分配吸引力
                self.attractions = np.zeros(self.num_stations)
                self.station_attractions = np.zeros(self.num_stations)  # 同时创建station_attractions属性
                
                for i, station_id in enumerate(self.station_ids):
                    if station_id in attraction_dict:
                        self.attractions[i] = attraction_dict[station_id]
                        self.station_attractions[i] = attraction_dict[station_id]  # 同步两个属性
                    else:
                        # 如果站点没有吸引力数据，设置默认值
                        self.attractions[i] = 1.0
                        self.station_attractions[i] = 1.0  # 同步两个属性
                
                print(f"成功加载{len(attraction_data)}个站点的吸引力数据")
                return True
            # 也检查station_id列名的情况
            elif 'station_id' in attraction_data.columns and 'attraction' in attraction_data.columns:
                # 原有的station_id处理逻辑
                attraction_dict = dict(zip(attraction_data['station_id'], attraction_data['attraction']))
                
                self.attractions = np.zeros(self.num_stations)
                self.station_attractions = np.zeros(self.num_stations)
                
                for i, station_id in enumerate(self.station_ids):
                    if station_id in attraction_dict:
                        self.attractions[i] = attraction_dict[station_id]
                        self.station_attractions[i] = attraction_dict[station_id]
                    else:
                        self.attractions[i] = 1.0
                        self.station_attractions[i] = 1.0
                
                print(f"成功加载{len(attraction_data)}个站点的吸引力数据")
                return True
            else:
                print(f"吸引力数据文件缺少必要的列。当前列: {attraction_data.columns.tolist()}, 需要: zone_id/station_id, attraction")
                return False
        except Exception as e:
            print(f"加载吸引力数据时出错: {e}")
            return False

    
    def _map_position_to_zone(self, position):
        """
        将位置映射到最近的区域ID
        
        Args:
            position (numpy.ndarray): 坐标位置 [x, y]
            
        Returns:
            int or None: 最近区域的ID，如果无法映射则返回None
        """
        if not hasattr(self, 'data_loader') or not hasattr(self.data_loader, 'zones'):
            return None
            
        # 获取区域坐标
        if 'x' in self.data_loader.zones.columns and 'y' in self.data_loader.zones.columns:
            zone_positions = self.data_loader.zones[['x', 'y']].values
            
            # 计算欧氏距离
            distances = np.sqrt(np.sum((zone_positions - position)**2, axis=1))
            
            # 找到最近的区域
            nearest_idx = np.argmin(distances)
            
            # 返回区域ID
            return self.data_loader.zones.iloc[nearest_idx]['zone_id']
        
        return None
    
    def _calculate_accessibility(self, locations):
        """
        计算站点的可达性得分
        
        Args:
            locations (numpy.ndarray): 站点位置数组 shape=(n_stations, 2)
            
        Returns:
            numpy.ndarray: 每个站点的可达性得分 shape=(n_stations,)
        """
        # 如果没有路网节点，则使用随机值
        if not hasattr(self, 'road_nodes') or len(self.road_nodes) == 0:
            return np.random.uniform(0.5, 1.0, size=len(locations))
        
        # 初始化可达性得分
        accessibility = np.zeros(len(locations))
        
        # 计算每个站点到所有路网节点的平均距离倒数作为可达性指标
        for i, station in enumerate(locations):
            # 计算站点到所有路网节点的距离
            distances = np.sqrt(np.sum((self.road_nodes - station)**2, axis=1))
            
            # 避免除零错误
            distances = np.maximum(distances, 1e-5)
            
            # 距离倒数的平均值作为可达性得分（距离越近，得分越高）
            accessibility[i] = np.mean(1.0 / distances)
        
        # 归一化处理，使所有得分在0-1之间
        if np.max(accessibility) > 0:
            accessibility = accessibility / np.max(accessibility)
        
        # 添加小随机扰动，避免所有站点可达性完全相同
        accessibility += np.random.uniform(-0.05, 0.05, size=len(accessibility))
        accessibility = np.clip(accessibility, 0.0, 1.0)  # 确保值在0-1范围内
        
        return accessibility
    
    def _dynamic_demand(self):
        """计算当前时间步的需求"""
        # 如果有加载的需求数据，使用预设需求
        if hasattr(self, 'demand_data') and self.demand_data:
            # 根据当前时间步获取对应小时的需求
            current_hour = (self.time_step // 10) % 24  # 假设每10步对应1小时
            if current_hour in self.demand_data:
                return self.demand_data[current_hour]
        
        # 没有预设需求数据时，生成动态需求
        demand = np.zeros(self.num_stations)
        for i, station in enumerate(self.station_pos):
            # 计算站点到当前需求中心的距离
            distance = np.linalg.norm(station - self.demand_center)
            # 距离越近，需求越高
            base_demand = max(0, 10 - 0.01 * distance)
            # 添加随机波动
            demand[i] = base_demand * (0.8 + 0.4 * np.random.random())
            
            # 考虑站点吸引力
            if hasattr(self, 'station_attractions'):
                demand[i] *= (0.5 + 0.5 * self.station_attractions[i])
        
        # 根据时间变化需求
        hour = (self.time_step // 10) % 24
        time_factor = 1.0
        if 7 <= hour <= 9:  # 早高峰
            time_factor = 1.5
        elif 17 <= hour <= 19:  # 晚高峰
            time_factor = 1.8
        elif 23 <= hour or hour <= 5:  # 深夜
            time_factor = 0.3
            
        # 应用时间因子和随机波动
        demand = demand * time_factor * (0.9 + 0.2 * np.random.random())
        
        return demand
    
    def _generate_world(self):
        """随机生成充电站位置、路网等数据"""
        # 随机生成充电站位置
        self.station_pos = np.random.rand(self.num_stations, 2) * self.area_size
        
        # 生成动态需求中心（随时间移动）
        self.demand_center = np.random.rand(2) * self.area_size
        
        # 生成基础交通网络
        self.road_nodes = np.random.rand(5, 2) * self.area_size  # 简化路网
        
        # 初始化站点容量和吸引力
        self.station_capacity = np.random.randint(5, 10, size=self.num_stations)
        self.charging_piles = np.random.randint(0, 5, size=self.num_stations)
        self.init_piles = self.charging_piles.copy()
        self.station_attractions = np.ones(self.num_stations)  # 默认吸引力为1

    def _get_state(self):
        """状态向量：充电桩数量 + 可达性 + 需求预测 + 吸引力 + 分析报告数据"""
        # 计算所需的站点数量，使最终状态向量维度与模型期望一致
        # 10092 / 6 = 1682，假设我们使用6个主要特征而非8个
        max_stations = 1682  # 这应该与您的训练集中的站点数一致
        
        # 确保所有特征向量长度一致
        accessibility = self._calculate_accessibility(self.station_pos)
        if len(accessibility) > max_stations:
            accessibility = accessibility[:max_stations]
        else:
            # 填充到目标长度
            accessibility = np.pad(accessibility, (0, max_stations - len(accessibility)), 'constant')
        
        demand = self._dynamic_demand()
        if len(demand) > max_stations:
            demand = demand[:max_stations]
        else:
            demand = np.pad(demand, (0, max_stations - len(demand)), 'constant')
        
        # 确保需求非零，避免除零错误
        max_demand = max(np.max(demand), 1e-5) 
        demand_normalized = demand / max_demand

        # 添加吸引力作为状态的一部分
        if hasattr(self, 'station_attractions'):
            attraction = self.station_attractions.copy()
        elif hasattr(self, 'attractions'):
            attraction = self.attractions.copy()
        else:
            print("警告: 未找到吸引力数据，使用默认值")
            attraction = np.ones(max_stations)
        
        # 确保吸引力向量长度正确
        if len(attraction) > max_stations:
            attraction = attraction[:max_stations]
        else:
            attraction = np.pad(attraction, (0, max_stations - len(attraction)), 'constant')
        
        attraction_normalized = attraction / (np.max(attraction) + 1e-5)
        
        # 简化P_ij和C_ij数据处理，减少维度
        P_ij_normalized = np.zeros(max_stations)
        C_ij_normalized = np.zeros(max_stations)
        
        if hasattr(self, 'P_ij') and len(self.P_ij) > 0:
            # 聚合每个站点的P_ij和C_ij
            for i in range(min(max_stations, self.num_stations)):
                station_p_values = []
                station_c_values = []
                for j in range(min(max_stations, self.num_stations)):
                    idx = i * self.num_stations + j
                    if idx < len(self.P_ij):
                        station_p_values.append(self.P_ij[idx])
                        station_c_values.append(self.C_ij[idx])
                
                # 每个站点使用平均值
                if station_p_values:
                    P_ij_normalized[i] = np.mean(station_p_values)
                if station_c_values:
                    C_ij_normalized[i] = np.mean(station_c_values)
        
        # 归一化P_ij和C_ij
        P_ij_max = np.max(P_ij_normalized)
        if P_ij_max > 0:
            P_ij_normalized = P_ij_normalized / P_ij_max
        
        C_ij_max = np.max(C_ij_normalized)
        if C_ij_max > 0:
            C_ij_normalized = C_ij_normalized / C_ij_max
        
        # 简化分析报告数据处理，仅使用必要的特征
        # 这里我们直接跳过报告数据，可以显著减少状态向量维度
        
        # 处理充电桩数量
        charging_piles = self.charging_piles.copy()
        if len(charging_piles) > max_stations:
            charging_piles = charging_piles[:max_stations]
        else:
            charging_piles = np.pad(charging_piles, (0, max_stations - len(charging_piles)), 'constant')
        
        # 确保station_capacity长度正确
        station_capacity = self.station_capacity.copy()
        if len(station_capacity) > max_stations:
            station_capacity = station_capacity[:max_stations]
        else:
            station_capacity = np.pad(station_capacity, (0, max_stations - len(station_capacity)), 'constant')
        
        # 归一化充电桩数量
        normalized_piles = charging_piles / (np.max(station_capacity) + 1e-5)
        
        # 构建最终状态向量 - 仅使用6个主要特征以减少维度
        state = np.concatenate([
            normalized_piles,        # 归一化充电桩数量
            accessibility,           # 可达性得分
            demand_normalized,       # 归一化需求
            attraction_normalized,   # 归一化吸引力
            P_ij_normalized,         # 归一化选择概率
            C_ij_normalized          # 归一化阻抗
        ])
        
        # 打印状态向量维度以辅助调试
        print(f"状态向量维度: {state.shape}") 
        return state
    
    # Huff模型计算
    def _calculate_huff_model(self):
        """计算基于Huff模型的奖励组件 - 改进稳定版"""
        try:
            # 如果没有Huff模型或相关属性，返回默认值
            if not hasattr(self, 'huff_model') or self.huff_model is None:
                return 0.2  # 默认值，与错误信息中显示的一致
            
            # 确保所有输入数据都是 numpy 数组
            charging_piles = np.array(self.charging_piles, dtype=float)
            
            # 确保attractions是数值数组而非列表
            if hasattr(self, 'attractions') and self.attractions is not None:
                attractions = np.array(self.attractions, dtype=float)
            elif hasattr(self, 'station_attractions') and self.station_attractions is not None:
                attractions = np.array(self.station_attractions, dtype=float)
            else:
                attractions = np.ones(len(charging_piles), dtype=float)
            
            # 确保长度一致
            if len(attractions) != len(charging_piles):
                print(f"警告：吸引力数组长度({len(attractions)})与充电桩数组长度({len(charging_piles)})不匹配，将进行截断")
                # 截断至相同长度
                min_len = min(len(attractions), len(charging_piles))
                attractions = attractions[:min_len]
                charging_piles = charging_piles[:min_len]
            
            # 获取站点位置并确保是numpy数组
            if hasattr(self, 'station_pos') and self.station_pos is not None:
                station_positions = np.array(self.station_pos, dtype=float)
                # 确保站点位置与其他数组长度一致
                if len(station_positions) != len(charging_piles):
                    print(f"警告：站点位置数组长度({len(station_positions)})与充电桩数组长度({len(charging_piles)})不匹配")
                    min_len = min(len(station_positions), len(charging_piles))
                    station_positions = station_positions[:min_len]
                    charging_piles = charging_piles[:min_len]
                    attractions = attractions[:min_len]
            else:
                print("警告：找不到站点位置数据，将使用随机位置")
                station_positions = np.random.rand(len(charging_piles), 2)  # 创建随机2D位置
            
            # 计算可达性 - 确保_calculate_accessibility正确实现
            try:
                accessibility = self._calculate_accessibility(station_positions)
                # 确保accessibility是数值数组
                accessibility = np.array(accessibility, dtype=float)
                
                # 检查accessibility长度
                if len(accessibility) != len(charging_piles):
                    print(f"警告：可达性数组长度({len(accessibility)})与充电桩数组长度({len(charging_piles)})不匹配")
                    min_len = min(len(accessibility), len(charging_piles))
                    accessibility = accessibility[:min_len]
                    charging_piles = charging_piles[:min_len]
                    attractions = attractions[:min_len]
            except Exception as acc_error:
                print(f"计算可达性时出错: {acc_error}，将使用默认值")
                accessibility = np.ones(len(charging_piles), dtype=float)
        
            # ==== 改进部分1: 更稳健的归一化方法 ====
            def robust_normalize(x, alpha=0.1):
                """稳健的sigmoid归一化，alpha控制曲线陡峭度"""
                # 处理全零数组特例
                if np.sum(np.abs(x)) < 1e-10:
                    return np.zeros_like(x)
                    
                # Z-score标准化，减少极端值影响
                if np.std(x) > 0:
                    z = (x - np.mean(x)) / (np.std(x) + 1e-10)
                else:
                    return np.ones_like(x) * 0.5  # 所有值相同时返回中间值
                    
                # 应用sigmoid函数进行平滑归一化
                sigmoid = 1.0 / (1.0 + np.exp(-alpha * z))
                return sigmoid
            
            # ==== 改进部分2: 优化权重配置 ====
            weights = {
                'capacity': 0.5,     # 提高容量权重 (0.4 -> 0.5)
                'attraction': 0.25,  # 降低吸引力权重 (0.3 -> 0.25)
                'accessibility': 0.25 # 降低可达性权重 (0.3 -> 0.25)
            }
            
            # 应用改进的归一化方法
            capacity_norm = robust_normalize(charging_piles, alpha=0.2)
            attraction_norm = robust_normalize(attractions, alpha=0.2)
            accessibility_norm = robust_normalize(accessibility, alpha=0.2)
            
            # ==== 改进部分3: 计算平滑得分 ====
            # 使用加权几何平均增强稳定性
            epsilon = 1e-5  # 防止零值
            
            # 计算各组件得分并保持在安全范围内
            capacity_component = np.clip(capacity_norm, 0.01, 0.99)
            attraction_component = np.clip(attraction_norm, 0.01, 0.99)
            accessibility_component = np.clip(accessibility_norm, 0.01, 0.99)
            
            # 几何平均计算
            weighted_scores = (
                (capacity_component) ** weights['capacity'] *
                (attraction_component) ** weights['attraction'] *
                (accessibility_component) ** weights['accessibility']
            )
            
            # 平滑处理，减少极端值影响
            log_scores = np.log1p(weighted_scores)
            huff_reward = np.mean(log_scores) / np.log(2)  # 归一化到合理范围
            
            # ==== 改进部分4: 时间平滑机制 ====
            # 如果有上一次的得分记录，进行时间平滑
            if hasattr(self, '_previous_huff_score'):
                smoothing_factor = 0.7  # 平滑因子，控制新旧得分的权重
                huff_reward = smoothing_factor * huff_reward + (1 - smoothing_factor) * self._previous_huff_score
            
            # 存储当前得分用于下次平滑
            self._previous_huff_score = huff_reward
            
            # 确保最终奖励在合理范围内
            huff_reward = np.clip(huff_reward, 0.1, 0.6)
            
            # 边界检查
            if not np.isfinite(huff_reward):  # 检查是否为NaN或无穷大
                print(f"警告：Huff奖励计算结果非有限值({huff_reward})，将使用默认值")
                huff_reward = 0.2
            
            # 调试信息
            print(f"Huff模型计算完成，得分: {huff_reward:.4f}")
            
            return huff_reward
            
        except Exception as e:
            print(f"Huff计算错误: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            return 0.2  # 发生错误时返回默认值

    

    def _calculate_reward(self):
        """
        计算奖励，基于Huff模型输出和现有指标，并结合分析报告数据
        同时整合专家知识
        
        Returns:
            float: 总奖励
        """
        demand = self._dynamic_demand()
        demand_ratio = np.array(self.charging_piles, dtype=float) / (np.array(demand, dtype=float) + 1e-5)

        # 需求匹配奖励
        demand_reward = np.mean(np.clip(demand_ratio, 0, 1))  # 鼓励供应满足需求，最大值为1.0

        # 成本惩罚（降低强度）
        cost_penalty = -0.01 * np.sum(self.charging_piles) / self.num_stations
        
        # 可达性奖励 - 确保数据类型一致
        accessibility = np.array(self._calculate_accessibility(self.station_pos), dtype=float)
        charging_piles_array = np.array(self.charging_piles, dtype=float)
        
        # 确保长度一致
        min_len = min(len(accessibility), len(charging_piles_array))
        accessibility = accessibility[:min_len]
        charging_piles_array = charging_piles_array[:min_len]
        
        # 为access_reward添加归一化处理
        max_possible_access = np.sum(np.ones_like(accessibility)) / self.num_stations  # 理论最大值
        access_reward = (np.sum(accessibility * charging_piles_array) / self.num_stations) / max_possible_access
                
        # 均衡惩罚
        utilization = np.array(self.charging_piles, dtype=float) / (np.array(self.station_capacity, dtype=float) + 1e-5)
        balance_reward = -0.05 * np.std(utilization)

        # 吸引力贡献率 - 根据站点吸引力加权的充电桩分配效率
        # 确保station_attractions存在并且是numpy数组
        if hasattr(self, 'station_attractions') and self.station_attractions is not None:
            attraction_array = np.array(self.station_attractions, dtype=float)
        elif hasattr(self, 'attractions') and self.attractions is not None:
            attraction_array = np.array(self.attractions, dtype=float)
        else:
            attraction_array = np.ones(self.num_stations, dtype=float)
            
        # 确保长度一致
        min_stations = min(len(attraction_array), self.num_stations, len(self.charging_piles))
        attraction_array = attraction_array[:min_stations]
        charging_piles_array = np.array(self.charging_piles[:min_stations], dtype=float)
            
        attraction_weight = attraction_array / (np.sum(attraction_array) + 1e-5)
        pile_distribution = charging_piles_array / (np.sum(charging_piles_array) + 1e-5)
        attraction_reward = 0.5 * (1 - np.sum(np.abs(attraction_weight - pile_distribution)))

        # Huff模型奖励 - 使用正确的方法名称
        huff_reward = self._calculate_huff_model()

        # 分析报告数据奖励
        analysis_reward = 0
        if hasattr(self, 'use_analysis_data') and self.use_analysis_data:
            # 获取站点ID列表
            if hasattr(self.data_loader, 'zones') and 'zone_id' in self.data_loader.zones.columns:
                zone_ids = self.data_loader.zones['zone_id'].values[:self.num_stations]
                
                # 选择概率匹配奖励
                prob_match_reward = 0
                total_valid_stations = 0
                
                # 检查selection_probabilities是否存在且有必要的列
                if hasattr(self, 'selection_probabilities'):
                    available_cols = self.selection_probabilities.columns.tolist()
                    station_id_col = None
                    
                    # 尝试找到站点ID列
                    for possible_col in ['station_id', 'id', 'site_id', 'charging_station_id']:
                        if possible_col in available_cols:
                            station_id_col = possible_col
                            break
                    
                    if station_id_col is not None:
                        for i, zone_id in enumerate(zone_ids):
                            station_data = self.selection_probabilities[self.selection_probabilities[station_id_col] == zone_id]
                            if not station_data.empty:
                                # 尝试找到概率列
                                prob_col = None
                                for col in ['avg_selection_probability', 'selection_probability', 'probability', 'prob']:
                                    if col in station_data.columns:
                                        prob_col = col
                                        break
                                
                                if prob_col is not None:
                                    recommended_prob = station_data[prob_col].values[0]
                                    current_prob = pile_distribution[i]
                                    prob_match = 1 - min(abs(current_prob - recommended_prob) / max(recommended_prob, 0.01), 1)
                                    prob_match_reward += prob_match
                                    total_valid_stations += 1
                
                if total_valid_stations > 0:
                    prob_match_reward /= total_valid_stations
                
                # 空间布局匹配奖励
                layout_match_reward = 0
                total_valid_layouts = 0
                
                # 检查station_layout_analysis是否存在且有必要的列
                if hasattr(self, 'station_layout_analysis'):
                    available_cols = self.station_layout_analysis.columns.tolist()
                    station_id_col = None
                    
                    # 尝试找到站点ID列
                    for possible_col in ['station_id', 'id', 'site_id', 'charging_station_id']:
                        if possible_col in available_cols:
                            station_id_col = possible_col
                            break
                    
                    if station_id_col is not None:
                        for i, zone_id in enumerate(zone_ids):
                            if i >= len(self.charging_piles):
                                continue  # 避免索引超出范围
                                
                            layout_data = self.station_layout_analysis[self.station_layout_analysis[station_id_col] == zone_id]
                            if not layout_data.empty:
                                # 查找得分列
                                score_col = None
                                for col in ['recommended_piles', 'layout_score', 'accessibility_score', 'score']:
                                    if col in layout_data.columns:
                                        score_col = col
                                        break
                                
                                if score_col == 'recommended_piles':
                                    # 推荐充电桩数量
                                    recommended = layout_data[score_col].values[0]
                                    current = self.charging_piles[i]
                                    layout_match = 1 - min(abs(current - recommended) / max(recommended, 1), 1)
                                elif score_col is not None:
                                    # 布局得分
                                    layout_score = layout_data[score_col].values[0]
                                    layout_match = layout_score * (self.charging_piles[i] / max(self.station_capacity[i], 1))
                                else:
                                    continue
                                
                                layout_match_reward += layout_match
                                total_valid_layouts += 1
                
                if total_valid_layouts > 0:
                    layout_match_reward /= total_valid_layouts
        
        # 定义权重
        weights = {
            'demand': 0.3,
            'cost': 0.1,
            'access': 0.2,
            'balance': 0.05,
            'attraction': 0.15,
            'huff': 0.2
        }

        # 计算加权总奖励
        original_reward = (
            weights['demand'] * demand_reward +
            weights['cost'] * cost_penalty +
            weights['access'] * access_reward +
            weights['balance'] * balance_reward +
            weights['attraction'] * attraction_reward +
            weights['huff'] * huff_reward
        )
        
        # 缩放总奖励
        original_reward = np.clip(original_reward, -5.0, 5.0)  # 限制在合理范围

        # 如果self.last_reward存在，平滑当前奖励
        if hasattr(self, 'last_reward'):
            original_reward = 0.8 * original_reward + 0.2 * self.last_reward
        self.last_reward = original_reward

        # =============== 专家知识奖励部分 ===============
        # 计算专家知识奖励
        expert_reward = 0.0

        # 1. 需求匹配奖励 - 惩罚充电桩与需求的不匹配
        supply_demand_ratio = charging_piles_array / (np.array(demand, dtype=float) + 1e-5)

        # 修改: 加入归一化处理，避免极端值
        # 惩罚比例过高或过低的站点 - 现在对平均值进行惩罚，而非简单求和
        oversupply_stations = np.sum(supply_demand_ratio > 1.5)
        oversupply_penalty = -0.2 * min(1.0, oversupply_stations / (self.num_stations + 1e-5))

        undersupply_stations = np.sum(supply_demand_ratio < 0.5)
        undersupply_penalty = -0.3 * min(1.0, undersupply_stations / (self.num_stations + 1e-5))

        # 鼓励适度的供需比 (0.8-1.2范围内最优)
        optimal_supply = np.sum((supply_demand_ratio >= 0.8) & (supply_demand_ratio <= 1.2))
        optimal_ratio_reward = 0.5 * optimal_supply / self.num_stations

        # 2. 考虑站点的地理位置重要性
        if hasattr(self, 'station_importance'):
            importance_array = np.array(self.station_importance, dtype=float)
            # 鼓励在重要站点投放更多充电桩
            importance_match = np.sum(importance_array * charging_piles_array) / np.sum(charging_piles_array)
            importance_reward = 0.4 * importance_match
        else:
            importance_reward = 0

        # 3. 空间分布合理性 - 惩罚充电桩分布过于集中
        if hasattr(self, 'station_pos'):
            # 计算充电桩分布的基尼系数
            sorted_piles = np.sort(charging_piles_array)
            cumsum_piles = np.cumsum(sorted_piles)
            gini = (np.sum((2 * np.arange(1, len(charging_piles_array) + 1) - len(charging_piles_array) - 1) * sorted_piles) / 
                    (len(charging_piles_array) * np.sum(sorted_piles) + 1e-10))
            
            # 惩罚过高的不平等性 (适度的不平等是可以接受的，但过高不行)
            distribution_reward = -0.3 * max(0, gini - 0.4)
        else:
            distribution_reward = 0

        # 4. 确保最低服务水平 - 每个站点至少有一定数量的充电桩
        min_piles_required = 1  # 每个站点最低要求
        stations_below_min = np.sum(charging_piles_array < min_piles_required)
        min_service_penalty = -0.2 * min(1.0, stations_below_min / self.num_stations)

        # 5. 历史表现相关 - 如果有历史效果好的布局，鼓励与其相似
        if hasattr(self, 'historical_best_layout') and self.historical_best_layout is not None:
            try:
                historical = np.array(self.historical_best_layout, dtype=float)
                if historical.size > 0:
                    # 确保长度一致
                    min_len = min(len(historical), len(charging_piles_array))
                    historical = historical[:min_len]
                    charging_array_comp = charging_piles_array[:min_len]
                    
                    similarity = 1.0 - np.sum(np.abs(charging_array_comp - historical)) / (2 * np.sum(historical) + 1e-10)
                    historical_reward = 0.2 * similarity
                else:
                    historical_reward = 0
            except (TypeError, ValueError):
                # 转换失败则忽略此奖励
                historical_reward = 0
                print("警告: historical_best_layout格式不正确")
        else:
            historical_reward = 0

        # 组合所有专家知识奖励
        expert_reward = (oversupply_penalty + undersupply_penalty + optimal_ratio_reward + 
                        importance_reward + distribution_reward + min_service_penalty + 
                        historical_reward)

        # 限制专家奖励的范围，避免极端值
        expert_reward = np.clip(expert_reward, -1.0, 1.0)

        # =============== 合并原始奖励和专家奖励 ===============
        # 根据训练进度调整专家知识的权重 - 从0.8逐渐减少到0.2
        if hasattr(self, 'episode_count'):
            # 从0.8逐渐减少到0.2，且最多训练200个回合
            expert_weight = max(0.2, 0.8 - 0.6 * min(1.0, self.episode_count / 200))
        else:
            expert_weight = 0.5
            
        # 确保原始奖励也被限制在合理范围
        original_reward = np.clip(original_reward, -2.0, 2.0)

        # 合并奖励
        final_reward = (1 - expert_weight) * original_reward + expert_weight * expert_reward

        
        # =============== 合并原始奖励和专家奖励 ===============
        # 根据训练进度调整专家知识的权重
        if hasattr(self, 'episode_count'):
            # 从0.8逐渐减少到0.2
            expert_weight = max(0.2, 0.8 - 0.6 * min(1.0, self.episode_count / 200))
        else:
            expert_weight = 0.5
            
        final_reward = (1 - expert_weight) * original_reward + expert_weight * expert_reward
        
        # 调试输出
        print(f"奖励明细: 需求={demand_reward:.2f}, 成本={cost_penalty:.2f}, "
            f"可达性={access_reward:.2f}, 均衡={balance_reward:.2f}, "
            f"吸引力={attraction_reward:.2f}, Huff={huff_reward:.2f}, "
            f"分析报告={analysis_reward:.2f}, 总计原始奖励={original_reward:.2f}")
        
        if hasattr(self, 'episode_count') and self.episode_count % 10 == 0:
            print(f"专家奖励: 过度供应={oversupply_penalty:.2f}, 供应不足={undersupply_penalty:.2f}, "
                f"最优比例={optimal_ratio_reward:.2f}, 重要性={importance_reward:.2f}, "
                f"分布={distribution_reward:.2f}, 最低服务={min_service_penalty:.2f}, "
                f"历史相似={historical_reward:.2f}")
            print(f"奖励组成: 原始={original_reward:.2f}, 专家={expert_reward:.2f}, "
                f"专家权重={expert_weight:.2f}, 最终={final_reward:.2f}")
        
        return final_reward

    def step(self, action):
        """执行动作
        action: 整数 (0 ~ 2*num_stations-1)
            - station_id = action // 2
            - operation = action % 2  (0:增加, 1:减少)
        """
        station = action // 2
        operation = action % 2
        
        # 确保站点索引在有效范围
        station = min(station, self.num_stations - 1)
        
        # 执行充电桩调整
        delta = 1 if operation == 0 else -1
        self.charging_piles[station] = np.clip(
            self.charging_piles[station] + delta, 
            0, 
            self.station_capacity[station]
        )
        
        # 更新区域数据中的充电桩数量
        for i, zone_id in enumerate(self.data_loader.zones['zone_id']):
            if i < self.num_stations:
                self.data_loader.zones.loc[i, 'charging_capacity'] = self.charging_piles[i]
        
        # 更新时间步
        current_time = self.timestamps[self.time_step % len(self.timestamps)]

        # 使用Huff模型计算P_ij和C_ij
        P_ij_list = []
        C_ij_list = []
        zone_ids = self.data_loader.zones['zone_id'].values[:self.num_stations]
        user_positions = self.road_nodes[:10] if len(self.road_nodes) > 0 else self.station_pos[:10]

        for user_pos in user_positions:
            origin_id = self._map_position_to_zone(user_pos)
            if origin_id is not None:
                probs = self.huff_model.predict_probability(origin_id, current_time)
                for dest_id in zone_ids:
                    if dest_id in probs.index:
                        p = probs[dest_id]
                        c = self.huff_model.calculate_impedance(origin_id, dest_id, current_time)
                        P_ij_list.append(p)  # 添加概率
                        C_ij_list.append(c)  # 添加阻抗
        
        # 存储为环境属性
        self.P_ij = np.array(P_ij_list) if P_ij_list else np.zeros(self.num_stations)
        self.C_ij = np.array(C_ij_list) if C_ij_list else np.ones(self.num_stations) * 10  # 默认阻抗
        
        # 更新状态和奖励
        next_state = self._get_state()
        reward = self._calculate_reward()
        
        # 增加时间步
        self.time_step += 1
        done = self.time_step >= 240  # 24小时，每步6分钟
        
        return next_state, reward, done
    
    def _random_initialize_piles(self):
        """随机初始化充电桩分布"""
        # 设置总桩数为容量总和的一半
        self.total_piles = int(np.sum(self.station_capacity) * 0.5)
        
        # 初始化全零分配
        self.charging_piles = np.zeros(self.num_stations, dtype=np.int32)
        
        # 随机分配充电桩
        remaining_piles = self.total_piles
        while remaining_piles > 0:
            # 随机选择一个站点
            station_idx = np.random.randint(0, self.num_stations)
            
            # 确保不超过容量
            if self.charging_piles[station_idx] < self.station_capacity[station_idx]:
                self.charging_piles[station_idx] += 1
                remaining_piles -= 1
        
        return self.charging_piles


    def validate_against_analysis(self):
        """
        验证DQN学习结果与分析报告建议的差异
        
        Returns:
            pandas.DataFrame: 包含验证结果的数据框，或者错误消息字符串
        """
        if not hasattr(self, 'use_analysis_data') or not self.use_analysis_data:
            return "未加载分析数据，无法验证"
        
        result = {
            "station_id": [],
            "dqn_deployment": [],
            "analysis_recommended": [],
            "match_ratio": []
        }
        
        # 获取站点ID列表
        zone_ids = self.data_loader.zones['zone_id'].values[:self.num_stations]
        
        total_match = 0
        valid_stations = 0
        
        for i, zone_id in enumerate(zone_ids):
            dqn_piles = self.charging_piles[i]
            
            layout_data = self.station_layout_analysis[self.station_layout_analysis['station_id'] == zone_id]
            if not layout_data.empty and 'recommended_piles' in layout_data.columns:
                recommended = int(layout_data['recommended_piles'].values[0])
                match_ratio = 1 - min(abs(recommended - dqn_piles) / max(recommended, 1), 1)
                total_match += match_ratio
                valid_stations += 1
                
                result["station_id"].append(zone_id)
                result["dqn_deployment"].append(dqn_piles)
                result["analysis_recommended"].append(recommended)
                result["match_ratio"].append(match_ratio)
        
        average_match = total_match / valid_stations if valid_stations > 0 else 0
        
        # 添加总体匹配度统计
        result_df = pd.DataFrame(result)
        result_df.loc["平均值"] = ["", result_df["dqn_deployment"].mean(), 
                                result_df["analysis_recommended"].mean(), 
                                average_match]
        
        print(f"DQN部署与分析报告建议的平均匹配度: {average_match:.2f}")
        
        return result_df
    
    def save_environment_state(self, filepath):
        """保存环境当前状态，用于后续分析"""
        state_dict = {
            'charging_piles': self.charging_piles.copy(),
            'station_pos': self.station_pos.copy() if hasattr(self, 'station_pos') else None,
            'station_attractions': self.station_attractions.copy() if hasattr(self, 'station_attractions') else None,
            'demand': self._dynamic_demand(),
            'accessibility': self._calculate_accessibility(self.station_pos) if hasattr(self, 'station_pos') else None,
            'reward_components': {
                'demand': None,  # 这些值在下次_calculate_reward时计算
                'cost': None,
                'accessibility': None,
                'balance': None,
                'attraction': None,
                'huff': None
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
        
        print(f"环境状态已保存至 {filepath}")




#############################################
# 5. 优先经验回放缓冲区
##############################print(f"demand_data keys: {list(self.demand_data.keys())}")###############
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        
    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []
            
        # 确保批次大小不超过缓冲区大小
        batch_size = min(batch_size, len(self.buffer))
        
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = max(prio, 1e-4)  # 避免零优先级
            
    def __len__(self):
        return len(self.buffer)

#############################################
# 6. DQN智能体
#############################################
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 自适应学习率优化器
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', patience=10, factor=0.5, verbose=True
        )
        
        self.gamma = gamma
        self.memory = PrioritizedReplayBuffer(50000)
        self.batch_size = 128
        self.tau = 0.01  # 软更新参数

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.max_grad_norm = 10.0  # 梯度裁剪阈值
        
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.policy_net.advantage_stream[-1].out_features - 1)
        
        state = torch.FloatTensor(state).to(self.device)
        self.policy_net.eval()  # 设置为评估模式
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()  # 恢复训练模式
        return q_values.argmax().item()
    
    def update(self, batch_size=None):
        """
        更新神经网络
        
        Args:
            batch_size: 可选参数，指定批量大小，默认使用预设值
            
        Returns:
            float: 损失值或None（如果没有更新）
        """
        # 使用动态批量大小或默认值
        if batch_size is None:
            batch_size = self.batch_size
            
        # 检查内存是否足够进行批量采样
        if len(self.memory) < batch_size:
            return None
        
        try:
            # 优先经验回放采样
            transitions, indices, weights = self.memory.sample(batch_size)
            if not transitions:  # 检查是否为空
                return None
                
            # 将转换列表解包成批量
            batch = list(zip(*transitions))
            
            # 转换为张量并移动到正确的设备
            states = torch.FloatTensor(np.array(batch[0])).to(self.device)
            actions = torch.LongTensor(np.array(batch[1])).to(self.device).unsqueeze(1)
            rewards = torch.FloatTensor(np.array(batch[2])).to(self.device)
            next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
            dones = torch.BoolTensor(np.array(batch[4])).to(self.device)
            
            # 检查张量形状
            self._check_tensor_shapes(states, actions, rewards, next_states, dones)
            
            # 计算当前Q值
            current_q = self.policy_net(states).gather(1, actions)
            
            # 计算目标Q值（Double DQN）
            with torch.no_grad():
                # 选择下一个最佳动作使用策略网络，但用目标网络评估值
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                
                # 计算目标值：对于终止状态，未来奖励为0
                target_q = rewards.unsqueeze(1) + self.gamma * next_q * (~dones.unsqueeze(1))
            
            # 计算TD误差，用于更新优先级
            td_errors = (target_q - current_q).abs()
            
            # 更新优先级缓冲区
            # 确保包含少量噪声以避免零概率
            new_priorities = td_errors.detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, new_priorities)
            
            # 应用重要性采样权重并计算加权损失
            weights_tensor = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
            elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (weights_tensor * elementwise_loss).mean()
            
            # 优化步骤
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            
            # 检查梯度是否健康
            if torch.isfinite(grad_norm):
                self.optimizer.step()
            else:
                print(f"警告: 梯度爆炸 ({grad_norm})，跳过此次更新")
                return None
            
            # 软更新目标网络
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
            
            # 返回损失值用于监控
            return loss.item()
            
        except Exception as e:
            print(f"更新时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _check_tensor_shapes(self, states, actions, rewards, next_states, dones):
        """检查输入张量的形状是否正确"""
        batch_size = states.shape[0]
        
        if actions.shape[0] != batch_size or actions.shape[1] != 1:
            print(f"警告: 动作张量形状异常: {actions.shape}, 预期: ({batch_size}, 1)")
            
        if rewards.shape[0] != batch_size:
            print(f"警告: 奖励张量形状异常: {rewards.shape}, 预期: ({batch_size},)")
            
        if next_states.shape[0] != batch_size:
            print(f"警告: 下一状态张量形状异常: {next_states.shape}, 预期: ({batch_size}, {states.shape[1]})")
        
        if dones.shape[0] != batch_size:
            print(f"警告: 完成标志张量形状异常: {dones.shape}, 预期: ({batch_size},)")

    def save_checkpoint(self, path, episode, optimizer_state=True):
        """保存检查点，包含模型权重和训练状态"""
        checkpoint = {
            'episode': episode,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'best_reward': getattr(self, 'best_reward', float('-inf')),
            'epsilon': getattr(self, 'current_epsilon', 0.1)
        }
        
        # 可选保存优化器状态
        if optimizer_state and self.optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            
        # 如果使用学习率调度器，也保存它的状态
        if hasattr(self, 'scheduler') and self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        print(f"模型检查点已保存到 {path}")
    
    def load_checkpoint(self, path):
        """从检查点恢复模型和训练状态"""
        try:
            print(f"尝试加载检查点: {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # 使用与save_checkpoint匹配的键名
            if 'policy_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                print("已加载策略网络权重")
            else:
                print("警告: 未找到策略网络权重")
                
            if 'target_state_dict' in checkpoint:
                self.target_net.load_state_dict(checkpoint['target_state_dict'])
                print("已加载目标网络权重")
            else:
                print("警告: 未找到目标网络权重")
                
            # 加载优化器状态
            if 'optimizer' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("已加载优化器状态")
                
            # 加载学习率调度器状态
            if 'scheduler' in checkpoint and hasattr(self, 'scheduler') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                print("已加载学习率调度器状态")
                
            # 恢复训练状态
            episode = checkpoint.get('episode', 0)
            self.best_reward = checkpoint.get('best_reward', -float('inf'))
            self.current_epsilon = checkpoint.get('epsilon', 0.01)
                
            print(f"成功加载检查点，恢复到第 {episode} 回合")
            return episode
        except Exception as e:
            print(f"加载检查点时出错: {str(e)}")
            # 打印检查点内容以进行调试
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                print(f"检查点键: {list(checkpoint.keys())}")
            except Exception as debug_e:
                print(f"无法读取检查点内容: {debug_e}")
            return None

#############################################
# 7. 可视化模块
#############################################
class Visualization:

    @staticmethod
    def plot_charging_stations(env, episode, save_path="output"):
        """可视化充电站分布和充电桩数量"""
        plt.figure(figsize=(12, 10))
        
        # 绘制充电站位置和容量
        capacities = env.charging_piles
        max_capacity = max(capacities) if len(capacities) > 0 else 10
        norm = plt.Normalize(0, max_capacity)
        cmap = plt.cm.viridis
        
        # 绘制道路网络
        if hasattr(env, 'road_nodes') and len(env.road_nodes) > 1:
            x_road = env.road_nodes[:, 0]
            y_road = env.road_nodes[:, 1]
            plt.scatter(x_road, y_road, c='gray', s=10, alpha=0.5, label='道路节点')
            
            # 可选：绘制简化道路连接
            for i in range(min(len(x_road)-1, 20)):  # 限制连接数量，避免过度绘制
                plt.plot([x_road[i], x_road[i+1]], [y_road[i], y_road[i+1]], 'gray', alpha=0.3, linewidth=1)
        
        # 绘制充电站
        scatter = plt.scatter(
            env.station_pos[:, 0], 
            env.station_pos[:, 1], 
            c=capacities, 
            cmap=cmap, 
            norm=norm, 
            s=100, 
            alpha=0.7,
            edgecolors='black'
        )
        
        # 绘制吸引力值（如果存在）
        if hasattr(env, 'station_attractions'):
            # 仅绘制部分站点的吸引力标签，避免拥挤
            n_labels = min(20, len(env.station_pos))
            indices = np.random.choice(len(env.station_pos), n_labels, replace=False)
            for i in indices:
                plt.annotate(
                    f"A:{env.station_attractions[i]:.2f}", 
                    (env.station_pos[i, 0], env.station_pos[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
        
        # 绘制需求中心（如果存在）
        if hasattr(env, 'demand_center'):
            plt.plot(env.demand_center[0], env.demand_center[1], 'r*', markersize=20, label='需求中心')
        
        # 添加颜色条和标签
        cbar = plt.colorbar(scatter)
        cbar.set_label('充电桩数量')
        
        plt.title(f'充电站分布与充电桩配置 (回合 {episode})')
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图像
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'charging_stations_ep{episode}.png'), dpi=200)
        plt.close()
    
    @staticmethod
    def plot_heatmap(env, episode, save_path="output"):
        """绘制Huff模型下的市场覆盖热力图"""
        plt.figure(figsize=(14, 10))
        
        # 生成网格点进行插值
        grid_size = 100
        lon_min, lon_max = env.station_pos[:, 0].min(), env.station_pos[:, 0].max()
        lat_min, lat_max = env.station_pos[:, 1].min(), env.station_pos[:, 1].max()
        
        lon_range = np.linspace(lon_min, lon_max, grid_size)
        lat_range = np.linspace(lat_min, lat_max, grid_size)
        grid_points = []
        for lon in lon_range:
            for lat in lat_range:
                grid_points.append([lon, lat])
        grid_points = np.array(grid_points)
        
        # 计算Huff模型的市场覆盖概率
        attractiveness = env.charging_piles + 1
        huff_probs = huff_model(env.station_pos, attractiveness, grid_points)
        
        # 找出每个网格点的最大概率站点
        max_prob_stations = np.argmax(huff_probs, axis=1)
        
        # 重塑为网格
        coverage_grid = max_prob_stations.reshape(grid_size, grid_size)
        
        # 绘制热力图
        plt.pcolormesh(lon_range, lat_range, coverage_grid.T, cmap='tab20', alpha=0.7)
        plt.colorbar(label='最优选择的充电站ID')
        
        # 绘制充电站位置
        plt.scatter(env.station_pos[:, 0], env.station_pos[:, 1], 
                   s=env.charging_piles * 30 + 50, c='red', edgecolors='black', 
                   label='充电站位置')
        
        # 添加站点标签
        for i, (lon, lat) in enumerate(env.station_pos):
            plt.text(lon, lat, f"{i+1}\n({env.charging_piles[i]})", fontsize=10, ha='center', va='center')
        
        plt.title(f"Episode {episode} - 充电站Huff模型市场覆盖热力图")
        plt.xlabel("经度")
        plt.ylabel("纬度")
        plt.grid(False)
        
        # 保存图像
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/coverage_heatmap_ep{episode}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_training_curve(rewards, save_path="output"):
        """绘制训练奖励曲线"""
        plt.figure(figsize=(12, 6))
        # 使用窗口平滑处理曲线
        window_size = min(10, len(rewards)//10) if len(rewards) > 20 else 1
        smoothed_rewards = []
        for i in range(len(rewards)):
            if i < window_size:
                smoothed_rewards.append(np.mean(rewards[:i+1]))
            else:
                smoothed_rewards.append(np.mean(rewards[i-window_size+1:i+1]))
        
        plt.plot(rewards, alpha=0.3, color='blue', label='原始奖励')
        plt.plot(smoothed_rewards, linewidth=2, color='red', label='平滑奖励')
        plt.xlabel("训练回合")
        plt.ylabel("总奖励")
        plt.title("DQN优化充电桩布局训练进度")
        plt.legend()
        plt.grid(True)
        
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/training_curve.png", dpi=300)
        plt.close()

    @staticmethod
    def create_station_summary(env, episode, save_path="output"):
        """创建充电站布局摘要文件"""
        summary = {
            'station_id': list(range(1001, env.num_stations + 1)),
            'longitude': env.station_pos[:, 0],
            'latitude': env.station_pos[:, 1],
            'charging_piles': env.charging_piles,
            'capacity': env.station_capacity
        }
        
        # 创建DataFrame并保存
        df = pd.DataFrame(summary)
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(f"{save_path}/station_summary_ep{episode}.csv", index=False)
        
        # 打印摘要信息
        print(f"\n=== Episode {episode} 充电站布局摘要 ===")
        print(df.to_string())
        print(f"总充电桩数量: {env.charging_piles.sum()}")
        print(f"平均利用率: {(env.charging_piles / env.station_capacity).mean():.2f}")
        print("=====================================\n")
        
#############################################
# 8. 优先级经验回放函数
#############################################
class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 确定优先级的程度
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
    def push(self, experience):
        """添加经验到缓冲区，并分配最高优先级"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        """基于优先级采样"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        """更新经验的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)


def plot_training_curve_local(rewards, moving_avg=None, save_path="output"):
    """本地版本的训练曲线绘制函数"""
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, 'b-', alpha=0.5, label='单回合奖励')
    
    if moving_avg:
        plt.plot(moving_avg, 'r-', label='移动平均奖励')
    
    plt.xlabel('Episode')
    plt.ylabel('累计奖励')
    plt.title('训练奖励曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_rewards.png'))
    plt.close()

def plot_eval_curve_local(eval_rewards, save_path="output"):
    """本地版本的评估曲线绘制函数"""
    plt.figure(figsize=(10, 6))
    plt.plot(eval_rewards, 'r-', label='评估奖励')
    plt.xlabel('评估次数')
    plt.ylabel('平均奖励')
    plt.title('评估奖励曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'eval_rewards.png'))
    plt.close()

#############################################
#9. 训练函数
#############################################
def train(data_path, output_path="output", config_path="config.json", num_episodes=None, 
          batch_size=None, gamma=None, epsilon_start=1.0, epsilon_end=0.01, 
          epsilon_decay=0.995, target_update=None, learning_rate=None, 
          load_checkpoint=False, checkpoint_path=None):
    """使用用户实际数据进行训练，集成专家知识引导"""
    print("加载数据并开始训练充电桩布局优化模型...")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 安全加载配置文件
    try:
        config = DataLoader.load_config(config_path)
        print(f"成功加载配置文件: {config_path}")
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        print("将使用默认配置参数...")
        config = {}
    
    # 优先使用显式传入的参数，其次使用配置文件中的参数，最后使用默认值
    num_episodes = num_episodes or config.get('num_episodes', 300)
    batch_size = batch_size or config.get('batch_size', 64)
    gamma = gamma or config.get('discount_factor', 0.99)
    target_update = target_update or config.get('update_target', 10)
    learning_rate = learning_rate or config.get('learning_rate', 0.001)
    
    # 打印使用的参数
    print(f"训练参数: episodes={num_episodes}, batch_size={batch_size}, "
          f"learning_rate={learning_rate}, gamma={gamma}, target_update={target_update}")
    
    # 初始化环境
    env = DynamicChargingEnv(data_path=data_path)
    
    # 创建专家知识实例并关联到环境
    from expert_knowledge import ExpertKnowledge
    expert = ExpertKnowledge()
    print("正在加载专家知识...")
    
    # 为环境设置专家知识引用和回合计数
    env.expert = expert
    env.episode_count = 0
    env.historical_best_layout = None  # 初始化历史最佳布局属性

    # 状态和动作维度设置 - 现在包含吸引力特征
    state_dim = env.num_stations * 6  # 增加了吸引力维度
    action_dim = env.num_stations * 2
    
    # 改进的探索参数设置
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = num_episodes * 0.75  # 使衰减更慢，探索期更长
    
    # 创建智能体 - 使用更小的学习率
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=config.get('learning_rate', 0.0005),  # 降低学习率
        gamma=config.get('discount_factor', 0.99)
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        agent.optimizer, 
        step_size=50,  # 每50个episode降低学习率
        gamma=0.9      # 降低到90%
    )
    
    # 检查是否从检查点恢复训练
    start_episode = 0
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint):
        if input("发现检查点，是否继续训练? (y/n): ").lower().startswith('y'):
            loaded_episode = agent.load_checkpoint(latest_checkpoint)
            if loaded_episode:
                start_episode = loaded_episode + 1
    
    best_reward = getattr(agent, 'best_reward', -float('inf'))
    reward_history = []
    eval_rewards = []  # 新增：评估奖励历史
    moving_avg_reward = []  # 新增：移动平均奖励
    
    # 设置检查点保存频率
    checkpoint_freq = config.get('checkpoint_frequency', 10)  # 每10个回合保存一次
    eval_freq = config.get('eval_frequency', 10)  # 评估频率
    
    # 目标网络软更新参数
    target_update_freq = 5  # 每5个episode更新一次目标网络
    soft_update_tau = 0.01  # 软更新系数
    
    # 记录训练开始时间
    start_time = time.time()
    
    try:
        for episode in tqdm(range(start_episode, num_episodes), desc="训练进度"):
            # 更新环境的回合计数
            env.episode_count = episode
            
            # 使用专家初始化（前50%的训练回合）
            use_expert_init = episode < num_episodes * 0.5
            if use_expert_init:
                # 先调用reset初始化环境
                state = env.reset()
                # 然后用专家布局替换默认布局
                expert_layout = expert.expert_initialization(env)
                env.charging_piles = expert_layout.copy()
                # 重新计算状态 (如果有专门的方法)
                if hasattr(env, '_get_state'):
                    state = env._get_state()
            else:
                state = env.reset()
                
            total_reward = 0
            episode_losses = []
            episode_steps = 0
            
            # 调整epsilon随训练进度递减 - 使用指数衰减
            progress = max(0, min(1, episode / epsilon_decay))
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-5 * progress)
            agent.current_epsilon = epsilon  # 保存当前epsilon以便检查点恢复
                
            done = False
            while not done:
                # 选择动作 - 整合专家知识
                use_expert, expert_action = expert.select_expert_action(state, env, epsilon)
                if use_expert and expert_action is not None:
                    action = expert_action
                else:
                    action = agent.select_action(state, epsilon=epsilon)
                    
                next_state, reward, done = env.step(action)
                
                # 存储经验
                agent.memory.push((state, action, reward, next_state, done))
                
                # 更新状态
                state = next_state
                total_reward += reward
                episode_steps += 1
                
                # 更改更新频率：更频繁地更新，但batch_size更小
                if len(agent.memory) > agent.batch_size:
                    # 根据训练阶段动态调整batch大小
                    dynamic_batch_size = min(
                        agent.batch_size, 
                        max(32, int(agent.batch_size * (episode / num_episodes)))
                    )
                    loss = agent.update(batch_size=dynamic_batch_size)
                    if loss is not None:
                        episode_losses.append(loss)
            
            # 记录奖励
            reward_history.append(total_reward)
            
            # 计算移动平均奖励
            window_size = min(10, len(reward_history))
            avg_reward = sum(reward_history[-window_size:]) / window_size
            moving_avg_reward.append(avg_reward)
            
            # 目标网络软更新
            if episode % target_update_freq == 0:
                # 使用软更新策略
                for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - soft_update_tau) + 
                        policy_param.data * soft_update_tau
                    )
                print(f"Episode {episode}: 执行目标网络软更新 (tau={soft_update_tau})")
            
            # 记录平均损失和专家动作使用情况
            if episode_losses:
                avg_loss = sum(episode_losses) / len(episode_losses)
                if episode % 10 == 0:
                    expert_ratio = expert.expert_actions_taken / max(1, expert.total_actions)
                    elapsed_time = time.time() - start_time
                    print(f"Episode {episode}, 奖励: {total_reward:.2f}, 移动平均: {avg_reward:.2f}, "
                          f"平均损失: {avg_loss:.4f}, 步数: {episode_steps}, "
                          f"专家动作使用率: {expert_ratio:.2%}, Epsilon: {epsilon:.3f}, 耗时: {elapsed_time:.1f}秒")
                    expert.expert_actions_taken = 0
                    expert.total_actions = 0
            
            # 学习率调度
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            if episode % 50 == 0:
                print(f"当前学习率: {current_lr:.6f}")
            
            # 定期评估（不使用探索）
            if episode % eval_freq == 0 or episode == num_episodes - 1:
                eval_reward = evaluate_agent(env, agent, num_eval_episodes=3)
                eval_rewards.append(eval_reward)
                print(f"Episode {episode}: 评估奖励: {eval_reward:.2f}")
                
                # 评估后执行专家知识提取
                expert.track_environment_state(env)
                
                # 如果发现了好的布局，更新历史最佳布局
                if expert.best_layout is not None:
                    env.historical_best_layout = expert.best_layout.copy()
                
            # 保存最佳模型 - 基于评估结果而非训练结果
            if len(eval_rewards) > 0 and eval_rewards[-1] > best_reward:
                best_reward = eval_rewards[-1]
                agent.best_reward = best_reward  # 保存最佳奖励以便检查点恢复
                torch.save(agent.policy_net.state_dict(), f"{output_path}/best_model.pth")
                print(f"Episode {episode}: 新的最佳模型已保存, 评估奖励: {best_reward:.2f}")
                
                # 同时也保存完整检查点
                agent.save_checkpoint(f"{checkpoint_dir}/best_checkpoint.pth", episode)
                
                # 保存当前环境状态
                env.save_environment_state(f"{output_path}/best_env_state.pkl")
                
                # 保存专家知识最佳布局
                expert.save_best_layout(output_path)
            
            # 定期保存检查点
            if episode % checkpoint_freq == 0 or episode == num_episodes - 1:
                agent.save_checkpoint(f"{checkpoint_dir}/checkpoint_ep{episode}.pth", episode)
                # 同时更新最新检查点
                agent.save_checkpoint(latest_checkpoint, episode)
                
            # 可视化当前分布和训练曲线
            if episode % 20 == 0 or episode == num_episodes - 1:
                Visualization.plot_charging_stations(env, episode, save_path=output_path)
                plot_training_curve_local(reward_history, moving_avg=moving_avg_reward, save_path=output_path)
                Visualization.create_station_summary(env, episode, save_path=output_path)
        
    except KeyboardInterrupt:
        print("\n训练被手动中断")
        # 在中断时保存检查点
        agent.save_checkpoint(f"{checkpoint_dir}/interrupt_checkpoint.pth", episode)
        print("已保存中断时的检查点，可以稍后恢复")
        
    print("训练完成！")
    # 最终评估
    final_eval_reward = evaluate_agent(env, agent, num_eval_episodes=5)
    print(f"最终评估奖励: {final_eval_reward:.2f}")
    
    # 保存最终的专家知识结果
    expert.save_best_layout(output_path)
    expert.plot_expert_metrics(output_path)
    
    # 最终可视化
    plot_training_curve_local(reward_history, moving_avg=moving_avg_reward, save_path=output_path)
    plot_eval_curve_local(eval_rewards, save_path=output_path)
    Visualization.create_station_summary(env, num_episodes-1, save_path=output_path)
        
    return agent

# 修改：更新评估函数以跟踪专家知识
def evaluate_agent(env, agent, num_eval_episodes=3):
    """
    在不使用探索策略的情况下评估智能体性能
    
    Args:
        env: 环境实例
        agent: 智能体实例
        num_eval_episodes: 评估回合数
    
    Returns:
        float: 平均评估奖励
    """
    eval_rewards = []
    
    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 使用贪婪策略（epsilon=0）
            action = agent.select_action(state, epsilon=0)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            
        eval_rewards.append(episode_reward)
    
    avg_reward = sum(eval_rewards) / len(eval_rewards)
    
    # 记录评估时的环境状态
    if hasattr(env, 'expert') and env.expert is not None:
        env.expert.track_environment_state(env)
    
    return avg_reward

