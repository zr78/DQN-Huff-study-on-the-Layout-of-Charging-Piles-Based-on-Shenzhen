'''
author:zhanguri
date:2023-11-24
version:0.1
bug:
    1.奖励一直在减小且为负数
    2.Huff可能与DQN结合不好
    3.huff_model函数仅仅是为了画图(可以删去，这个项奕轩可以画)，没有实际作用
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
            # 加载充电站数据
            stations_path = os.path.join(data_path, "station_inf.csv")
            if os.path.exists(stations_path):
                stations_df = pd.read_csv(stations_path)
                print(f"成功加载充电站数据: {len(stations_df)}个站点")
                
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
        try:
            # 尝试根据POI数据生成需求
            poi_path = os.path.join(data_path, "poi.csv")
            station_path = os.path.join(data_path, "station_inf.csv")
            
            if os.path.exists(station_path):
                stations_df = pd.read_csv(station_path)
                station_locs = None
                if 'longitude' in stations_df.columns and 'latitude' in stations_df.columns:
                    station_locs = stations_df[['longitude', 'latitude']].values
                
                # 如果有POI数据，用POI数据的密度生成需求
                if os.path.exists(poi_path) and station_locs is not None:
                    poi_df = pd.read_csv(poi_path)
                    poi_locs = None
                    if 'longitude' in poi_df.columns and 'latitude' in poi_df.columns:
                        poi_locs = poi_df[['longitude', 'latitude']].values
                    
                    if poi_locs is not None:
                        # 生成24小时的模拟需求数据
                        demand_by_time = {}
                        num_stations = len(stations_df)
                        
                        # 生成需求密度
                        station_demands = np.zeros(num_stations)
                        for i in range(num_stations):
                            # 计算每个站点周围POI点的数量，作为需求参考
                            station_point = station_locs[i]
                            poi_distances = np.sqrt(np.sum((poi_locs - station_point)**2, axis=1))
                            nearby_pois = np.sum(poi_distances < 0.01)  # 0.01度约1公里
                            station_demands[i] = max(1, nearby_pois / 10)  # 简化为每10个POI产生1单位需求
                        
                        # 根据时间段变化需求
                        for hour in range(24):
                            # 时间因子: 早晚高峰需求高
                            time_factor = 1.0
                            if 7 <= hour <= 9:  # 早高峰
                                time_factor = 1.5
                            elif 17 <= hour <= 19:  # 晚高峰
                                time_factor = 1.8
                            elif 23 <= hour or hour <= 5:  # 深夜
                                time_factor = 0.3
                            
                            # 应用时间因子调整需求
                            demand_by_time[hour] = station_demands * time_factor
                        
                        print(f"根据POI数据生成了24小时的需求数据")
                        return demand_by_time
                        
                # 如果没有POI数据，根据站点自身特征生成需求
                elif station_locs is not None:
                    num_stations = len(stations_df)
                    demand_by_time = {}
                    
                    # 根据站点位置简单生成基础需求
                    base_demands = np.random.uniform(1, 10, size=num_stations)
                    
                    # 生成24小时需求
                    for hour in range(24):
                        # 时间因子
                        time_factor = 1.0
                        if 7 <= hour <= 9:
                            time_factor = 1.5
                        elif 17 <= hour <= 19:
                            time_factor = 1.8
                        elif 23 <= hour or hour <= 5:
                            time_factor = 0.3
                            
                        demand_by_time[hour] = base_demands * time_factor
                    
                    print(f"生成了24小时的模拟需求数据")
                    return demand_by_time
            
            print(f"警告: 未找到充分的数据生成需求模型，将使用随机生成数据")
            return None
                
        except Exception as e:
            print(f"生成需求数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        
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
        """
        num_stations: 充电站数量，如果未提供则尝试从数据加载或使用默认值
        area_size: 模拟区域尺寸(单位：米)
        data_path: 数据文件路径
        """
        self.area_size = area_size
        self.time_step = 0
        
        # 尝试加载真实数据
        try:
            station_data = pd.read_csv(f"{data_path}/stations_inf.csv")
            if len(station_data) > 0:
                print(f"已加载{len(station_data)}个充电站位置数据")
                if 'x' in station_data.columns and 'y' in station_data.columns:
                    self.station_pos = station_data[['x', 'y']].values
                    self.num_stations = len(self.station_pos)  # 从数据中推导 num_stations
                    if 'capacity' in station_data.columns:
                        self.station_capacity = station_data['capacity'].values
                    else:
                        self.station_capacity = np.random.randint(5, 10, size=self.num_stations)
                    
                    if 'piles' in station_data.columns:
                        self.init_piles = station_data['piles'].values
                    else:
                        self.init_piles = np.random.randint(0, 5, size=self.num_stations)
                        
                    # 加载路网数据
                    try:
                        traffic_data = pd.read_csv(f"{data_path}/traffic.csv")
                        if 'x' in traffic_data.columns and 'y' in traffic_data.columns:
                            self.road_nodes = traffic_data[['x', 'y']].values
                            print(f"已加载{len(self.road_nodes)}个路网节点")
                        else:
                            self.road_nodes = np.random.rand(5, 2) * self.area_size
                    except:
                        self.road_nodes = np.random.rand(5, 2) * self.area_size
                        
                    # 加载需求数据
                    try:
                        demand_data = pd.read_csv(f"{data_path}/demand.csv")
                        if len(demand_data) > 0:
                            self.demand_data = demand_data
                            print(f"已加载需求数据，形状: {demand_data.shape}")
                        else:
                            self.demand_data = None
                    except:
                        self.demand_data = None
                        
                    return  # 成功加载数据后返回
                    
        except Exception as e:
            print(f"加载数据失败: {e}，将使用随机生成数据")
            
        # 如果没有加载到真实数据，则需要 num_stations
        if num_stations is None:
            num_stations = 10  # 默认值
        self.num_stations = num_stations
        self._generate_world()
        
    def _generate_world(self):
        """随机生成充电站位置、路网等数据"""
        # 随机生成充电站位置
        self.station_pos = np.random.rand(self.num_stations, 2) * self.area_size
        
        # 生成动态需求中心（随时间移动）
        self.demand_center = np.random.rand(2) * self.area_size
        
        # 生成基础交通网络
        self.road_nodes = np.random.rand(5, 2) * self.area_size  # 简化路网
        
        # 初始化站点容量
        self.station_capacity = np.random.randint(5, 10, size=self.num_stations)
        self.init_piles = np.random.randint(0, 5, size=self.num_stations)
        self.demand_data = None
        
    def _calculate_accessibility(self, stations):
        """计算可达性得分（使用路网距离）"""
        distances = []
        for pos in stations:
            # 简化计算：取最近路网节点的直线距离
            min_dist = np.min([np.linalg.norm(pos - node) for node in self.road_nodes])
            distances.append(min_dist)
        return 1 / (1 + np.array(distances))
    
    def _dynamic_demand(self):
        """随时间变化的需求模式"""
        if self.demand_data is not None and 'time' in self.demand_data.columns:
            # 使用真实需求数据
            time_idx = self.time_step % len(self.demand_data)
            row = self.demand_data.iloc[time_idx]
            if 'demand' in row:
                return row['demand'].values
            else:
                # 如果找不到需求列，使用除时间外的所有列
                demand_cols = [col for col in row.index if col != 'time']
                return row[demand_cols].values
        
    # 需求中心周期性移动
        if hasattr(self, 'demand_center'):
            self.demand_center += np.random.normal(0, 10, size=2)  # 减小移动步长
            self.demand_center = np.clip(self.demand_center, 0, self.area_size)
            demand = np.zeros(self.num_stations)
            for i in range(self.num_stations):
                dist = np.linalg.norm(self.station_pos[i] - self.demand_center)
                demand[i] = 10 * math.exp(-dist**2/(2*(200**2)))  # 降低峰值到 10
            demand = np.clip(demand, 0, 20)  # 添加上限
            return demand
        return np.random.randint(1, 10, size=self.num_stations)
    
    def reset(self):
        """重置环境状态"""
        self.time_step = 0
        
        # 初始化充电桩数量
        self.charging_piles = self.init_piles.copy()
        
        # 如果没有预定义的需求中心，则随机生成
        if not hasattr(self, 'demand_center'):
            self.demand_center = np.random.rand(2) * self.area_size
            
        return self._get_state()
    
    def _get_state(self):
        """状态向量：充电桩数量 + 可达性 + 需求预测"""
        accessibility = self._calculate_accessibility(self.station_pos)
        demand = self._dynamic_demand()
        
        # 确保需求非零，避免除零错误
        max_demand = max(np.max(demand), 1e-5) 
        demand_normalized = demand / max_demand
        print(f"Debug: demand={demand}, max_demand={max_demand}, normalized={demand_normalized}")

        return np.concatenate([
            self.charging_piles / np.max(self.station_capacity),  # 归一化充电桩数
            accessibility,                                      # 可达性得分
            demand_normalized                               # 归一化需求
        ])
    
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
        
        # 获取新状态
        next_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 更新时间步
        self.time_step += 1
        done = self.time_step >= 240  # 模拟24小时（每6分钟一步）
        
        return next_state, reward, done
    
    def _calculate_reward(self):
        demand = self._dynamic_demand()
        demand_ratio = self.charging_piles / (demand + 1e-5)
        demand_reward = -0.5 * np.mean(np.abs(1 - demand_ratio))  # 减小惩罚
        cost_penalty = -0.05 * self.charging_piles.sum() / self.num_stations  # 减小成本惩罚
        accessibility = self._calculate_accessibility(self.station_pos)
        access_reward = 0.5 * (accessibility * self.charging_piles).sum() / self.num_stations  # 增加正向奖励
        utilization = self.charging_piles / (self.station_capacity + 1e-5)
        balance_reward = -0.1 * np.std(utilization)  # 减小均衡惩罚
        total_reward = demand_reward + cost_penalty + access_reward + balance_reward
        print(f"Reward breakdown: demand={demand_reward:.2f}, cost={cost_penalty:.2f}, access={access_reward:.2f}, balance={balance_reward:.2f}, total={total_reward:.2f}")
        return total_reward

#############################################
# 5. 优先经验回放缓冲区
#############################################
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
        
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.policy_net.advantage_stream[-1].out_features - 1)
        
        state = torch.FloatTensor(state).to(self.device)
        self.policy_net.eval()  # 设置为评估模式
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()  # 恢复训练模式
        return q_values.argmax().item()
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 优先经验回放采样
        transitions, indices, weights = self.memory.sample(self.batch_size)
        if not transitions:  # 检查是否为空
            return
            
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(np.array(batch[1])).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(batch[2])).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.BoolTensor(np.array(batch[4])).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (~dones.unsqueeze(1))
        
        # 计算优先级
        td_errors = (target_q - current_q).abs()
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # 计算损失
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        loss = (weights_tensor * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        
        # 优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
        self.optimizer.step()
        
        # 软更新目标网络
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau*policy_param.data + (1-self.tau)*target_param.data)
        
        return loss.item()
    
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
    
    def load_checkpoint(self, path, load_optimizer=True):
        """从检查点加载模型和训练状态"""
        if not os.path.exists(path):
            print(f"找不到检查点文件: {path}")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # 加载模型权重
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            
            # 恢复训练状态
            self.best_reward = checkpoint.get('best_reward', float('-inf'))
            self.current_epsilon = checkpoint.get('epsilon', 0.1)
            
            # 可选地加载优化器状态
            if load_optimizer and 'optimizer' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                
            # 如果有学习率调度器，也加载它
            if hasattr(self, 'scheduler') and self.scheduler and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                
            episode = checkpoint.get('episode', 0)
            print(f"成功从检查点加载模型，继续训练自第 {episode} 个回合")
            return episode
            
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

#############################################
# 7. 可视化模块
#############################################
class Visualization:
    @staticmethod
    def plot_charging_stations(env, episode, save_path="output"):
        """绘制充电站分布和充电桩数量"""
        plt.figure(figsize=(12, 10))
        
        # 准备数据
        lons = env.station_pos[:, 0]
        lats = env.station_pos[:, 1]
        sizes = env.charging_piles * 30 + 50  # 根据充电桩数量设置点大小
        
        # 绘制散点图，大小表示充电桩数量
        scatter = plt.scatter(lons, lats, s=sizes, c=env.charging_piles, 
                  cmap='viridis', alpha=0.8, edgecolors='black')
        
        # 添加色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('充电桩数量')
        
        # 绘制需求中心
        plt.plot(env.demand_center[0], env.demand_center[1], 'r*', markersize=20, label='需求中心')
        
        # 添加站点标签
        for i, (lon, lat) in enumerate(zip(lons, lats)):
            plt.text(lon, lat, f"{i+1}", fontsize=10, ha='center', va='center')
        
        plt.title(f"Episode {episode} - 深圳市充电站分布与充电桩数量")
        plt.xlabel("经度")
        plt.ylabel("纬度")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图像
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/station_distribution_ep{episode}.png", dpi=300, bbox_inches='tight')
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
            'station_id': list(range(1, env.num_stations + 1)),
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
# 8. 训练函数
#############################################
def train(data_path="data", config_path="config.json"):
    """使用用户实际数据进行训练"""
    print("加载数据并开始训练充电桩布局优化模型...")
    
    # 加载配置
    config = DataLoader.load_config(config_path)
    num_episodes = config.get('num_episodes', 300)
    
    # 创建输出目录
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化环境
    env = DynamicChargingEnv(data_path=data_path)
    
    # 状态和动作维度设置
    state_dim = env.num_stations * 3
    action_dim = env.num_stations * 2
    
    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim,
        lr=config.get('learning_rate', 0.001),
        gamma=config.get('discount_factor', 0.99)
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
    
    # 设置检查点保存频率
    checkpoint_freq = config.get('checkpoint_frequency', 10)  # 每10个回合保存一次
    
    try:
        for episode in tqdm(range(start_episode, num_episodes), desc="训练进度"):
            state = env.reset()
            total_reward = 0
            episode_losses = []
            
            # 调整epsilon随训练进度递减
            epsilon = max(0.1, 0.9 * (0.995 ** episode))
            agent.current_epsilon = epsilon  # 保存当前epsilon以便检查点恢复
                
            done = False
            while not done:
                action = agent.select_action(state, epsilon=epsilon)
                next_state, reward, done = env.step(action)
                
                # 存储经验
                agent.memory.push((state, action, reward, next_state, done))
                
                # 更新状态
                state = next_state
                total_reward += reward
                
                # 每4步更新一次网络
                if len(agent.memory) > agent.batch_size and env.time_step % 4 == 0:
                    loss = agent.update()
                    if loss is not None:
                        episode_losses.append(loss)
                
                # 记录奖励
                reward_history.append(total_reward)
                
                # 记录平均损失
                if episode_losses:
                    avg_loss = sum(episode_losses) / len(episode_losses)
                    if episode % 10 == 0:
                        print(f"Episode {episode}, 奖励: {total_reward:.2f}, 平均损失: {avg_loss:.4f}")
                
                # 保存最佳模型
                if total_reward > best_reward:
                    best_reward = total_reward
                    agent.best_reward = best_reward  # 保存最佳奖励以便检查点恢复
                    torch.save(agent.policy_net.state_dict(), f"{output_path}/best_model.pth")
                    print(f"Episode {episode}: 新的最佳模型已保存, 奖励: {total_reward:.2f}")
                    
                    # 同时也保存完整检查点
                    agent.save_checkpoint(f"{checkpoint_dir}/best_checkpoint.pth", episode)
                
                # 定期保存检查点
                if episode % checkpoint_freq == 0 or episode == num_episodes - 1:
                    agent.save_checkpoint(f"{checkpoint_dir}/checkpoint_ep{episode}.pth", episode)
                    # 同时更新最新检查点
                    agent.save_checkpoint(latest_checkpoint, episode)
                    
                # 可视化当前分布
                if episode % 50 == 0 or episode == num_episodes - 1:
                    Visualization.plot_charging_stations(env, episode, save_path=output_path)
                    Visualization.plot_training_curve(reward_history, save_path=output_path)
                    Visualization.create_station_summary(env, episode, save_path=output_path)
        
    except KeyboardInterrupt:
        print("\n训练被手动中断")
        # 在中断时保存检查点
        agent.save_checkpoint(f"{checkpoint_dir}/interrupt_checkpoint.pth", episode)
        print("已保存中断时的检查点，可以稍后恢复")
        
    print("训练完成！")
    Visualization.plot_training_curve(reward_history, save_path=output_path)
    Visualization.create_station_summary(env, num_episodes-1, save_path=output_path)
        
    return agent


