import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from scipy.spatial.distance import squareform, pdist
import geopandas as gpd
from haversine import haversine
from skopt import gp_minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import os
import warnings
import datetime
import joblib
import logging

# 设置 matplotlib 正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体，可根据系统情况修改
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings("ignore")

# 一、数据加载与预处理
class UrbanEVDataLoader:
    """负责加载和预处理UrbanEV项目中的数据"""
#C:\Users\zhang\Desktop\DQN\data   
    def __init__(self, data_path='C:/Users/zhang/Desktop/DQN/data/'):
        """初始化数据加载器"""
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        
    def load_datasets(self):
        """加载所有需要的数据集"""
        logger.info("加载数据集...")
        
        # 加载基础信息 (站点信息)
        self.inf = pd.read_csv(os.path.join(self.data_path, 'station_inf.csv'))
        
        # 加载邻接矩阵
        self.adj = pd.read_csv(os.path.join(self.data_path, 'adj.csv'))
        
        # 加载距离矩阵
        self.distance_df = pd.read_csv(os.path.join(self.data_path, 'distance.csv'))
        
        # 加载时间序列数据(宽格式)
        self.occupancy_wide = pd.read_csv(os.path.join(self.data_path, 'occupancy.csv'))
        self.occupancy_wide['time'] = pd.to_datetime(self.occupancy_wide['time'])
        
        self.volume_wide = pd.read_csv(os.path.join(self.data_path, 'volume.csv'))
        self.volume_wide['time'] = pd.to_datetime(self.volume_wide['time'])
        
        self.e_price_wide = pd.read_csv(os.path.join(self.data_path, 'e_price.csv'))
        self.e_price_wide['time'] = pd.to_datetime(self.e_price_wide['time'])
        
        self.s_price_wide = pd.read_csv(os.path.join(self.data_path, 's_price.csv'))
        self.s_price_wide['time'] = pd.to_datetime(self.s_price_wide['time'])
        
        # 加载天气数据
        self.weather = pd.read_csv(os.path.join(self.data_path, 'weather_airport.csv'))
        self.weather['time'] = pd.to_datetime(self.weather['time'])
        
        # 加载POI数据
        self.poi = pd.read_csv(os.path.join(self.data_path, 'poi.csv'))
        
        logger.info("数据集加载完成")
        return self
    
    def _wide_to_long(self, df_wide, value_name):
        """将宽格式数据转换为长格式"""
        # 确保所有列名都是字符串
        df_wide.columns = df_wide.columns.astype(str)
        
        # 选择站点ID列（除了time列以外的所有列）
        id_columns = [col for col in df_wide.columns if col != 'time']
        
        # 使用melt函数将宽格式转为长格式
        df_long = pd.melt(
            df_wide, 
            id_vars=['time'], 
            value_vars=id_columns,
            var_name='station_id', 
            value_name=value_name
        )
        
        # 强制将station_id列转换为整数类型
        try:
            df_long['station_id'] = pd.to_numeric(df_long['station_id'], errors='coerce')
            # 删除可能产生的NaN值
            df_long = df_long.dropna(subset=['station_id'])
            df_long['station_id'] = df_long['station_id'].astype(int)
        except:
            logger.warning("无法将station_id转换为整数，保持原始格式")
        
        return df_long
    
    def preprocess(self):
        """预处理所有加载的数据"""
        logger.info("预处理数据...")
        
        # 输出数据形状信息
        logger.info(f"基础信息表(inf)行数: {len(self.inf)}")
        logger.info(f"占用率表(宽格式)行数: {len(self.occupancy_wide)}, 列数: {len(self.occupancy_wide.columns)}")
        
        # 显示几个区域ID作为示例
        sample_ids = list(self.inf['station_id'].unique())[:5]
        logger.info(f"示例站点ID: {sample_ids}")
        
        # 创建站点列表（从inf.csv提取）
        self.station_ids = self.inf['station_id'].unique()
        
        # 将宽格式数据转换为长格式
        self.occupancy = self._wide_to_long(self.occupancy_wide, 'occupancy')
        self.volume = self._wide_to_long(self.volume_wide, 'volume')
        self.e_price = self._wide_to_long(self.e_price_wide, 'price')
        self.s_price = self._wide_to_long(self.s_price_wide, 'service_price')
        
        # 标准化天气数据
        if 'T' in self.weather.columns:  # 温度
            self.weather['temperature'] = self.weather['T']
            self.weather[['temperature']] = self.scaler.fit_transform(self.weather[['temperature']])
        
        if 'U' in self.weather.columns:  # 湿度
            self.weather['humidity'] = self.weather['U']
            self.weather[['humidity']] = self.scaler.fit_transform(self.weather[['humidity']])
        
        if 'P' in self.weather.columns:  # 气压 - 可能影响充电
            self.weather[['P']] = self.scaler.fit_transform(self.weather[['P']])
        
        # 计算区域特征
        self.zones = self.inf.copy()
        
        # 由于没有直接的capacity字段，我们使用charge_count作为容量估计
        if 'charge_count' in self.zones.columns:
            self.zones['capacity'] = self.zones['charge_count']
            self.zones['charging_capacity'] = self.zones['capacity'] * 0.85  # 假设85%的充电桩可用
        else:
            # 若没有charge_count, 设置默认容量
            self.zones['capacity'] = 10
            self.zones['charging_capacity'] = 8.5
        
        # 将station_id重命名为zone_id以便与代码其他部分兼容
        self.zones.rename(columns={'station_id': 'zone_id'}, inplace=True)
        
        # 构造距离矩阵（确保为方阵）
        station_ids = [str(id) for id in self.station_ids]
        self.distance = np.zeros((len(station_ids), len(station_ids)))
        
        # 提取距离矩阵中的值
        for i, src_id in enumerate(station_ids):
            if src_id in self.distance_df.columns:
                for j, dst_id in enumerate(station_ids):
                    if dst_id in self.distance_df.columns:
                        self.distance[i, j] = self.distance_df[dst_id].iloc[i]
        
        # 为占用率数据添加等待时间（基于占用率估计）
        self.occupancy['wait_time'] = np.exp(self.occupancy['occupancy'] / 25) - 1
        self.occupancy['wait_time'] = self.occupancy['wait_time'].clip(0, 30)  # 最多等待30分钟
        
        # 创建时空索引
        self.timestamps = pd.date_range(
            start=self.occupancy['time'].min(),
            end=self.occupancy['time'].max(),
            freq='H'
        )
        
        logger.info("数据预处理完成")
        return self
    
    def add_temporal_features(self):
        """为数据集添加时间特征"""
        logger.info("添加时间特征...")
        
        # 为所有时间序列数据添加时间特征
        for dataset_name in ['occupancy', 'volume', 'e_price', 's_price']:
            if hasattr(self, dataset_name):
                dataset = getattr(self, dataset_name)
                dataset['hour'] = dataset['time'].dt.hour
                dataset['day_of_week'] = dataset['time'].dt.dayofweek
                dataset['weekend'] = dataset['day_of_week'].isin([5, 6]).astype(int)
                dataset['month'] = dataset['time'].dt.month
                
                # 添加高峰时段指标
                dataset['morning_rush'] = dataset['hour'].between(7, 9).astype(int)
                dataset['evening_rush'] = dataset['hour'].between(16, 19).astype(int)
                dataset['rush_hour'] = ((dataset['morning_rush'] == 1) | (dataset['evening_rush'] == 1)).astype(int)
                
                setattr(self, dataset_name, dataset)
        
        logger.info("时间特征添加完成")
        return self
    
    def create_spatiotemporal_features(self):
        """创建时空特征"""
        logger.info("创建时空特征...")
        
        # 计算每小时每区域的平均占用率
        hour_zone_occupancy = self.occupancy.groupby(['station_id', 'hour'])['occupancy'].mean().reset_index()
        hour_zone_occupancy.rename(columns={'occupancy': 'avg_hourly_occupancy'}, inplace=True)
        self.hour_zone_occupancy = hour_zone_occupancy
        
        # 计算星期几模式
        dow_zone_occupancy = self.occupancy.groupby(['station_id', 'day_of_week'])['occupancy'].mean().reset_index()
        dow_zone_occupancy.rename(columns={'occupancy': 'avg_dow_occupancy'}, inplace=True)
        self.dow_zone_occupancy = dow_zone_occupancy
        
        # 计算平均充电量特征
        hour_zone_volume = self.volume.groupby(['station_id', 'hour'])['volume'].mean().reset_index()
        hour_zone_volume.rename(columns={'volume': 'avg_hourly_volume'}, inplace=True)
        self.hour_zone_volume = hour_zone_volume
        
        # 计算电价变化特征
        hour_zone_price = self.e_price.groupby(['station_id', 'hour'])['price'].mean().reset_index()
        hour_zone_price.rename(columns={'price': 'avg_hourly_price'}, inplace=True)
        self.hour_zone_price = hour_zone_price
            
        logger.info("时空特征创建完成")
        return self
    
    def get_zone_id_mapping(self):
        """创建一个从数字索引到zone_id的映射"""
        zone_ids = self.zones['zone_id'].values
        return {i: zone_id for i, zone_id in enumerate(zone_ids)}
    
    def get_zone_index_mapping(self):
        """创建一个从zone_id到数字索引的映射"""
        zone_ids = self.zones['zone_id'].values
        return {zone_id: i for i, zone_id in enumerate(zone_ids)}

# 二、动态Huff模型
class DynamicHuffEV:
    """电动汽车充电站的动态Huff模型"""
    
    def __init__(self, data_loader):
        """初始化模型"""
        self.dl = data_loader
        # 默认参数设置
        self.params = {
            'alpha': 0.6,  # 容量吸引力权重
            'beta': 0.35,  # 价格敏感度
            'gamma': 0.4,  # 空间阻抗因子
            'lambda': 1.5, # 距离衰减参数
            'delta': 0.2,  # 天气影响因子
            'eta': 0.15,   # POI吸引力权重
            'theta': 0.25  # 时间依赖因子
        }
        # 创建zone_id映射
        self.zone_id_map = self.dl.get_zone_id_mapping()
        self.zone_index_map = self.dl.get_zone_index_mapping()

    def calculate_station_attractiveness(self, station_positions, charging_piles, road_nodes):
        """
        使用Huff模型计算站点吸引力
        
        Args:
            station_positions: 站点位置坐标
            charging_piles: 各站点充电桩数量
            road_nodes: 道路网络节点
            
        Returns:
            numpy array: 各站点的吸引力值
        """
        n_stations = len(station_positions)
        if n_stations < 2:
            return np.ones(n_stations)
        
        # 计算站点间距离矩阵
        distances = np.zeros((n_stations, n_stations))
        for i in range(n_stations):
            for j in range(n_stations):
                if i != j:
                    distances[i, j] = np.linalg.norm(station_positions[i] - station_positions[j])
                else:
                    distances[i, j] = 0.1  # 避免除零错误
        
        # 初始化吸引力为充电桩数量(+1避免0值)
        attractions = charging_piles + 1
        
        # 计算可达性 (到道路网络的平均距离)
        accessibility = np.zeros(n_stations)
        for i in range(n_stations):
            # 计算到最近3个道路节点的平均距离
            node_distances = [np.linalg.norm(station_positions[i] - node) for node in road_nodes]
            node_distances.sort()
            accessibility[i] = np.mean(node_distances[:3]) if len(node_distances) >= 3 else np.mean(node_distances)
        
        # 可达性转换为吸引力因子 (越近越有吸引力)
        max_access = np.max(accessibility)
        accessibility_factor = 1 - (accessibility / max_access)
        
        # 吸引力结合充电桩数量和可达性
        attractions = attractions * (0.7 + 0.3 * accessibility_factor)
        
        # 归一化到0.5-1.5范围
        min_attr = np.min(attractions)
        max_attr = np.max(attractions)
        normalized_attractions = 0.5 + (attractions - min_attr) / (max_attr - min_attr)
        
        return normalized_attractions

    def calculate_attraction(self, zone_id, timestamp, include_weather=True, include_poi=True):
        """计算特定时间下充电站区域的吸引力"""
        # 将numpy.datetime64转换为pandas.Timestamp
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp)
            
        # 获取区域信息
        zone_info = self.dl.zones[self.dl.zones['zone_id'] == zone_id]
        if len(zone_info) == 0:
            logger.warning(f"未找到区域ID {zone_id} 的信息")
            return 0.001  # 返回默认最小值
        
        zone_info = zone_info.iloc[0]
        
        # 获取实时电价
        e_price_data = self.dl.e_price[
            (self.dl.e_price['station_id'] == zone_id) &
            (self.dl.e_price['time'] <= timestamp)
        ].sort_values('time', ascending=False)
        
        if len(e_price_data) > 0:
            e_price = e_price_data.iloc[0]['price']
        else:
            # 如果没有实时数据，使用平均价格
            e_price = self.dl.e_price[self.dl.e_price['station_id'] == zone_id]['price'].mean()
            if np.isnan(e_price):
                e_price = 1.0  # 默认电价
        
        # 获取当前占用率
        occupancy_data = self.dl.occupancy[
            (self.dl.occupancy['station_id'] == zone_id) &
            (self.dl.occupancy['time'] <= timestamp)
        ].sort_values('time', ascending=False)
        
        if len(occupancy_data) > 0:
            occupancy = occupancy_data.iloc[0]['occupancy']
        else:
            # 使用历史平均占用率
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            hourly_data = self.dl.hour_zone_occupancy[
                (self.dl.hour_zone_occupancy['station_id'] == zone_id) &
                (self.dl.hour_zone_occupancy['hour'] == hour)
            ]
            if len(hourly_data) > 0:
                occupancy = hourly_data.iloc[0]['avg_hourly_occupancy']
            else:
                occupancy = 50  # 默认占用率50%
        
        # 计算基本吸引力
        if 'charging_capacity' in zone_info:
            effective_capacity = zone_info['charging_capacity'] * (1 - occupancy/100)
        else:
            # 如果没有容量数据，使用默认值
            effective_capacity = 10 * (1 - occupancy/100)
            
        base_attraction = self.params['alpha'] * effective_capacity / (e_price + 0.01)
        
        # 时间依赖因子（工作日与周末模式不同）
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # 时间因子：工作日高峰时段更高，周末分布更均匀
        if is_weekend:
            time_factor = 0.7 + 0.3 * np.sin(np.pi * (hour - 10) / 12)  # 周末10点到下午4点高峰
        else:
            # 工作日双峰（早高峰和晚高峰）
            morning_peak = np.exp(-0.5 * ((hour - 8) / 2) ** 2)  # 早高峰约8点
            evening_peak = np.exp(-0.5 * ((hour - 18) / 3) ** 2)  # 晚高峰约6点
            time_factor = 0.5 + 0.5 * max(morning_peak, evening_peak)
        
        attraction = base_attraction * (1 + self.params['theta'] * time_factor)
        
        # 添加天气影响
        if include_weather:
            weather_data = self.dl.weather[self.dl.weather['time'] <= timestamp].sort_values('time', ascending=False)
            if len(weather_data) > 0:
                temp = weather_data.iloc[0]['temperature']
                
                # 根据天气调整吸引力（极端温度时人们更倾向于去有空调的地方充电）
                weather_factor = 1 + self.params['delta'] * (0.5 - temp)
                attraction *= weather_factor
        
        # 添加POI影响 - 检查附近POI密集程度
        if include_poi and hasattr(self.dl, 'poi'):
            # 从区域获取坐标
            if 'latitude' in zone_info and 'longitude' in zone_info:
                zone_lat = zone_info['latitude']
                zone_lon = zone_info['longitude']
                
                # 计算所有POI到该区域的距离
                poi_distances = []
                for _, poi in self.dl.poi.iterrows():
                    poi_lat = poi['latitude']
                    poi_lon = poi['longitude']
                    distance = haversine((zone_lat, zone_lon), (poi_lat, poi_lon))
                    if distance < 2:  # 2公里内的POI
                        poi_distances.append(distance)
                
                # POI密度影响因子
                poi_count = len(poi_distances)
                poi_factor = 1 + self.params['eta'] * (2 / (1 + np.exp(-0.02 * poi_count)) - 1)
                attraction *= poi_factor
            
        return max(0.001, attraction)  # 确保最小吸引力
        
    def calculate_impedance(self, origin_id, dest_id, timestamp):
        """计算从起点到目的地充电的阻抗（成本）"""
        # 获取区域间距离
        origin_idx = self.zone_index_map.get(origin_id)
        dest_idx = self.zone_index_map.get(dest_id)
        
        if origin_idx is not None and dest_idx is not None:
            try:
                distance = self.dl.distance[origin_idx, dest_idx]
            except (IndexError, KeyError):
                # 如果距离矩阵中没有这些ID，回退到坐标的直线距离
                distance = self._calculate_distance_from_coords(origin_id, dest_id)
        else:
            distance = self._calculate_distance_from_coords(origin_id, dest_id)
        
        # 获取目的地等待时间
        occupancy_data = self.dl.occupancy[
            (self.dl.occupancy['station_id'] == dest_id) &
            (self.dl.occupancy['time'] <= timestamp)
        ].sort_values('time', ascending=False)
        
        if len(occupancy_data) > 0 and 'wait_time' in occupancy_data.columns:
            wait_time = occupancy_data.iloc[0]['wait_time']
        else:
            # 从占用率估计等待时间
            if len(occupancy_data) > 0:
                occupancy = occupancy_data.iloc[0]['occupancy']
                wait_time = max(0, np.exp(occupancy / 25) - 1)  # 指数关系
            else:
                wait_time = 5  # 默认等待时间
        
        # 获取目的地服务价格
        s_price_data = self.dl.s_price[
            (self.dl.s_price['station_id'] == dest_id) &
            (self.dl.s_price['time'] <= timestamp)
        ].sort_values('time', ascending=False)
        
        if len(s_price_data) > 0:
            s_price = s_price_data.iloc[0]['service_price']
        else:
            s_price = self.dl.s_price[self.dl.s_price['station_id'] == dest_id]['service_price'].mean()
            if np.isnan(s_price):
                s_price = 1.0  # 默认服务价格
        
        # 获取当前天气因子
        weather_data = self.dl.weather[self.dl.weather['time'] <= timestamp].sort_values('time', ascending=False)
        if len(weather_data) > 0 and 'temperature' in weather_data.columns:
            temp = weather_data.iloc[0]['temperature']
            weather_factor = 1 + 0.2 * (0.5 - temp)  # 极端温度下阻抗更高
        else:
            weather_factor = 1.0
        
        # 计算总阻抗
        base_impedance = (
            self.params['gamma'] * (distance + 0.1) +  # 空间距离（加小常数避免除零）
            self.params['beta'] * (wait_time + s_price * 0.1)  # 时间和经济成本
        )
        
        return max(0.01, base_impedance * weather_factor)  # 确保最小阻抗
    
    def _calculate_distance_from_coords(self, origin_id, dest_id):
        """从坐标计算两点之间的距离"""
        origin_info = self.dl.zones[self.dl.zones['zone_id'] == origin_id]
        dest_info = self.dl.zones[self.dl.zones['zone_id'] == dest_id]
        
        if len(origin_info) > 0 and len(dest_info) > 0 and 'latitude' in origin_info.columns:
            origin_coords = (origin_info.iloc[0]['latitude'], origin_info.iloc[0]['longitude'])
            dest_coords = (dest_info.iloc[0]['latitude'], dest_info.iloc[0]['longitude'])
            return haversine(origin_coords, dest_coords)
        else:
            # 默认距离
            return 10.0
    
    def predict_probability(self, origin_id, timestamp, candidate_zones=None):
        """预测从给定起点选择每个充电站的概率"""
        if candidate_zones is None:
            candidate_zones = self.dl.zones['zone_id'].unique()
        
        attractions = []
        impedances = []
        zone_ids = []
        
        for dest_id in candidate_zones:
            zone_ids.append(dest_id)
            a = self.calculate_attraction(dest_id, timestamp)
            c = self.calculate_impedance(origin_id, dest_id, timestamp)
            attractions.append(a)
            impedances.append(c)
        
        # 使用Huff模型公式计算效用
        attractions = np.array(attractions)
        impedances = np.array(impedances)
        utilities = attractions / np.power(impedances, self.params['lambda'])
        
        # 处理数值问题
        utilities = np.nan_to_num(utilities, nan=0.001, posinf=100, neginf=0.001)
        utilities = np.clip(utilities, 0.001, 100)  # 限制极值
        
        # 转换为概率
        sum_utilities = np.sum(utilities)
        if sum_utilities > 0:
            probabilities = utilities / sum_utilities
        else:
            probabilities = np.ones_like(utilities) / len(utilities)  # 如果效用都为0，返回均匀分布
        
        # 以Series形式返回结果
        return pd.Series(probabilities, index=zone_ids)
    
    def predict_flow(self, origin_id, demand, timestamp, candidate_zones=None):
        """预测从起点到目的地的电动汽车充电需求流"""
        probabilities = self.predict_probability(origin_id, timestamp, candidate_zones)
        flows = probabilities * demand
        return flows

    def predict_all_flows(self, demands, timestamp, candidate_zones=None):
        """预测所有起点-终点对之间的流量"""
        all_flows = []
        
        # 限制候选区域数量以提高性能
        if candidate_zones is None:
            candidate_zones = list(self.dl.zones['zone_id'].unique())
        
        if len(candidate_zones) > 50:  # 最多处理50个目的地区域
            logger.warning(f"候选区域过多({len(candidate_zones)})，限制为50个以提高性能")
            candidate_zones = candidate_zones[:50]
        
        # 限制起点数量
        if len(demands) > 30:  # 最多处理30个起点
            logger.warning(f"起点数量过多({len(demands)})，限制为30个以提高性能")
            limited_demands = dict(list(demands.items())[:30])
        else:
            limited_demands = demands
        
        # 添加错误处理
        try:
            for origin_id, demand in tqdm(limited_demands.items(), desc="预测充电流量"):
                flows = self.predict_flow(origin_id, demand, timestamp, candidate_zones)
                flow_df = pd.DataFrame({
                    'origin_id': origin_id,
                    'dest_id': flows.index,
                    'flow': flows.values,
                    'timestamp': timestamp
                })
                all_flows.append(flow_df)
        except Exception as e:
            logger.error(f"预测充电流量时出错: {e}")
            return pd.DataFrame(columns=['origin_id', 'dest_id', 'flow', 'timestamp'])
            
        if all_flows:
            return pd.concat(all_flows, ignore_index=True)
        else:
            return pd.DataFrame(columns=['origin_id', 'dest_id', 'flow', 'timestamp'])
    
    def set_params(self, params):
        """设置模型参数"""
        self.params.update(params)
        return self

# 三、时空预测模块
class STPredictor:
    """时空预测器"""
    
    def __init__(self, data_loader):
        """初始化时空预测器"""
        self.dl = data_loader
        self.models = {}
        
    def predict_occupancy(self, zone_id, future_periods=24, include_weather=True):
        """预测特定区域的未来占用率"""
        # 获取该区域的历史数据
        zone_data = self.dl.occupancy[self.dl.occupancy['station_id'] == zone_id].copy()
        
        if len(zone_data) < 24:  # 需要足够的数据
            logger.warning(f"区域{zone_id}的数据不足。使用平均模式。")
            # 回退到平均模式
            current_hour = pd.Timestamp.now().hour
            next_hours = [(current_hour + i) % 24 for i in range(future_periods)]
            
            if hasattr(self.dl, 'hour_zone_occupancy'):
                hourly_patterns = self.dl.hour_zone_occupancy[
                    self.dl.hour_zone_occupancy['station_id'] == zone_id
                ]
                if len(hourly_patterns) > 0:
                    forecasts = []
                    for h in next_hours:
                        hour_data = hourly_patterns[hourly_patterns['hour'] == h]
                        if len(hour_data) > 0:
                            forecast = hour_data['avg_hourly_occupancy'].values[0]
                        else:
                            forecast = 50  # 默认值
                        forecasts.append(forecast)
                    
                    # 创建预测DataFrame
                    future_timestamps = pd.date_range(
                        start=pd.Timestamp.now(), 
                        periods=future_periods, 
                        freq='H'
                    )
                    return pd.DataFrame({
                        'ds': future_timestamps,
                        'yhat': forecasts,
                        'yhat_lower': [max(0, f - 15) for f in forecasts],
                        'yhat_upper': [min(100, f + 15) for f in forecasts]
                    })
            
            # 如果以上都失败，返回常数预测
            future_timestamps = pd.date_range(
                start=pd.Timestamp.now(), 
                periods=future_periods, 
                freq='H'
            )
            return pd.DataFrame({
                'ds': future_timestamps,
                'yhat': [50] * future_periods,
                'yhat_lower': [35] * future_periods,
                'yhat_upper': [65] * future_periods
            })
        
        # 准备Prophet的数据
        prophet_data = zone_data.rename(columns={
            'time': 'ds',
            'occupancy': 'y'
        })
        
        # 添加回归变量
        regressors = []
        if include_weather and hasattr(self.dl, 'weather'):
            # 合并天气数据
            weather_data = self.dl.weather.copy()
            if 'time' in weather_data.columns:
                weather_data = weather_data.rename(columns={'time': 'ds'})
                weather_vars = {'temperature': 'T', 'humidity': 'U'}
                
                for target_col, source_col in weather_vars.items():
                    if source_col in weather_data.columns:
                        prophet_data = prophet_data.merge(
                            weather_data[['ds', source_col]], 
                            on='ds', 
                            how='left'
                        )
                        prophet_data[target_col] = prophet_data[source_col]
                        prophet_data[target_col] = prophet_data[target_col].fillna(method='ffill').fillna(method='bfill')
                        regressors.append(target_col)
        
        # 添加时间特征
        prophet_data['hour'] = prophet_data['ds'].dt.hour
        prophet_data['day_of_week'] = prophet_data['ds'].dt.dayofweek
        prophet_data['weekend'] = prophet_data['day_of_week'].isin([5, 6]).astype(int)
        regressors.extend(['hour', 'weekend'])
        
        # 训练Prophet模型
        model = Prophet(
            interval_width=0.95,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # 添加回归变量
        for regressor in regressors:
            model.add_regressor(regressor)
        
        model.fit(prophet_data)
        self.models[f'occupancy_{zone_id}'] = model
        
        # 创建未来数据框
        future = model.make_future_dataframe(periods=future_periods, freq='H')
        
        # 为未来数据框添加回归变量值
        future['hour'] = future['ds'].dt.hour
        future['day_of_week'] = future['ds'].dt.dayofweek
        future['weekend'] = future['day_of_week'].isin([5, 6]).astype(int)
        
        if include_weather and regressors:
            for col in ['temperature', 'humidity']:
                if col in regressors:
                    # 简单起见，使用最后已知的值
                    last_value = prophet_data[col].iloc[-1]
                    future[col] = last_value
        
        # 生成预测
        forecast = model.predict(future)
        
        # 确保占用率在有效范围内(0-100%)
        forecast['yhat'] = np.clip(forecast['yhat'], 0, 100)
        forecast['yhat_lower'] = np.clip(forecast['yhat_lower'], 0, 100)
        forecast['yhat_upper'] = np.clip(forecast['yhat_upper'], 0, 100)
        
        # 只返回未来时段
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_periods)
    
    def predict_zone_demand(self, zone_id, timestamp, model_type='time_series'):
        """预测特定时间特定区域的充电需求"""
        if model_type == 'time_series':
            # 查找历史充电量数据中的模式
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # 在历史数据中查找类似时间
            if hasattr(self.dl, 'volume'):
                similar_times = self.dl.volume[
                    (self.dl.volume['station_id'] == zone_id) &
                    (self.dl.volume['hour'] == hour) &
                    (self.dl.volume['weekend'] == is_weekend)
                ]
                
                if len(similar_times) > 0:
                    return similar_times['volume'].mean()
            
            # 如果没有找到类似的时间，使用基于区域的简单估计
            if hasattr(self.dl, 'zones') and 'capacity' in self.dl.zones.columns:
                zone_cap = self.dl.zones[self.dl.zones['zone_id'] == zone_id]['capacity'].values[0]
                # 基础需求是容量的30%
                base_demand = zone_cap * 0.3
                
                # 根据一天中的时间调整
                time_factor = 1.0
                if hour >= 7 and hour <= 9:  # 早高峰
                    time_factor = 1.5
                elif hour >= 17 and hour <= 19:  # 晚高峰
                    time_factor = 1.8
                elif hour >= 23 or hour <= 5:  # 夜间
                    time_factor = 0.3
                
                # 根据周末调整
                weekend_factor = 0.7 if is_weekend else 1.0
                
                return base_demand * time_factor * weekend_factor
            
            # 非常基本的回退值
            return 10  # 默认值
        
        elif model_type == 'demographic':
            # 更复杂的模型，使用人口统计数据、POI等
            # 这需要比基础仓库更多的数据
            return 15  # 占位符
        
        else:
            logger.warning(f"未知的模型类型：{model_type}。使用默认需求。")
            return 10  # 默认值

# 四、模型优化
class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, huff_model, data_loader, validation_data=None):
        """初始化模型优化器"""
        self.model = huff_model
        self.dl = data_loader
        self.validation_data = validation_data
        
    def objective_function(self, params):
        """参数优化的目标函数"""
        # 更新模型参数
        param_dict = {
            'alpha': params[0],
            'beta': params[1],
            'gamma': params[2],
            'lambda': params[3],
            'delta': params[4] if len(params) > 4 else self.model.params['delta'],
            'eta': params[5] if len(params) > 5 else self.model.params['eta'],
            'theta': params[6] if len(params) > 6 else self.model.params['theta']
        }
        self.model.set_params(param_dict)
        
        # 如果有验证数据，用它计算损失
        if self.validation_data is not None:
            loss = self._compute_validation_loss()
        else:
            # 否则使用历史数据估计损失
            loss = self._compute_historical_loss()
            
        logger.info(f"参数: {param_dict}, 损失: {loss}")
        return loss
    
    def _compute_validation_loss(self):
        """使用验证数据计算损失"""
        # 假设validation_data有列：origin_id, dest_id, timestamp, actual_flow
        loss = 0
        
        for _, row in self.validation_data.iterrows():
            origin_id = row['origin_id']
            dest_id = row['dest_id']
            timestamp = row['timestamp']
            actual_flow = row['actual_flow']
            
            # 获取所有预测概率
            probs = self.model.predict_probability(origin_id, timestamp)
            pred_prob = probs.get(dest_id, 0)
            
            # 计算损失（使用平方误差）
            error = (pred_prob - actual_flow) ** 2
            loss += error
            
        return loss / len(self.validation_data)
    
    def _compute_historical_loss(self):
        """使用历史数据模式计算损失"""
        # 采样一些区域对和时间戳
        if not hasattr(self.dl, 'volume') or len(self.dl.volume) == 0:
            return 1000  # 如果没有历史数据，则损失高
            
        sample_zones = np.random.choice(self.dl.zones['zone_id'].unique(), 
                                        size=min(5, len(self.dl.zones)), 
                                        replace=False)
        sample_times = np.random.choice(self.dl.volume['time'].unique(), 
                                         size=min(10, len(self.dl.volume['time'].unique())), 
                                         replace=False)
        
        total_loss = 0
        count = 0
        
        for zone_id in sample_zones:
            for timestamp in sample_times:
                # 获取此时间此区域的所有充电量
                actual_volume = self.dl.volume[
                    (self.dl.volume['station_id'] == zone_id) & 
                    (self.dl.volume['time'] == timestamp)
                ]['volume'].sum()
                
                if actual_volume > 0:
                    # 获取预测值
                    demand = {z: 10 for z in self.dl.zones['zone_id'].unique()}  # 假设所有区域的需求相等
                    flows = self.model.predict_all_flows(demand, timestamp)
                    pred_volume = flows[flows['dest_id'] == zone_id]['flow'].sum()
                    
                    # 添加损失
                    rel_error = abs(pred_volume - actual_volume) / (actual_volume + 1)
                    total_loss += rel_error
                    count += 1
        
        if count == 0:
            return 1000  # 如果没有有效样本，则损失高
            
        return total_loss / count
    
    def optimize(self, n_calls=50, random_state=42):
        """优化模型参数"""
        # 定义参数边界
        param_bounds = [
            (0.1, 1.0),     # alpha: 容量权重
            (0.1, 1.0),     # beta: 价格敏感度
            (0.1, 1.0),     # gamma: 空间阻抗
            (0.5, 2.5),     # lambda: 距离衰减
            (0.0, 0.5),     # delta: 天气影响
            (0.0, 0.5),     # eta: POI权重
            (0.0, 0.5)      # theta: 时间因子
        ]
        
        logger.info("开始参数优化...")
        
        # 运行贝叶斯优化
        result = gp_minimize(
            self.objective_function,
            param_bounds,
            n_calls=n_calls,
            random_state=random_state,
            verbose=True
        )
        
        # 提取优化后的参数
        opt_params = {
            'alpha': result.x[0],
            'beta': result.x[1],
            'gamma': result.x[2],
            'lambda': result.x[3],
            'delta': result.x[4],
            'eta': result.x[5],
            'theta': result.x[6]
        }
        
        logger.info(f"优化完成。最佳参数: {opt_params}")
        
        # 用优化参数更新模型
        self.model.set_params(opt_params)
        
        return opt_params

# 五、可视化与分析
class EVHuffVisualizer:
    """EV Huff模型可视化器"""
    
    def __init__(self, huff_model, data_loader):
        """初始化可视化器"""
        self.model = huff_model
        self.dl = data_loader
        
    # 1. 修改 EVHuffVisualizer 类的 plot_attraction_map 方法
    def plot_attraction_map(self, timestamp, save_path=None, max_zones=50):
        """绘制特定时间的充电站吸引力地图，限制区域数量以提高性能"""
        # 检查是否有空间数据
        if not hasattr(self.dl, 'zones') or 'latitude' not in self.dl.zones.columns:
            logger.error("无法绘制吸引力地图：区域数据中缺少空间坐标。")
            return None
        
        # 限制区域数量
        all_zones = self.dl.zones['zone_id'].unique()
        if len(all_zones) > max_zones:
            logger.warning(f"区域过多({len(all_zones)})，限制为{max_zones}个以加快可视化")
            zones_to_plot = all_zones[:max_zones]
        else:
            zones_to_plot = all_zones
        
        # 计算每个区域的吸引力
        attractions = []
        logger.info(f"计算{len(zones_to_plot)}个区域的吸引力...")
        
        for zone_id in tqdm(zones_to_plot, desc="计算区域吸引力"):
            # 禁用POI计算以提高速度
            attraction = self.model.calculate_attraction(zone_id, timestamp, 
                                                        include_weather=True, 
                                                        include_poi=False)
            attractions.append({'zone_id': zone_id, 'attraction': attraction})
        
        attraction_df = pd.DataFrame(attractions)
        
        # 与区域数据合并
        plot_data = self.dl.zones.merge(attraction_df, on='zone_id', how='right')
        
        # 创建条形图而非地图，速度更快
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data.sort_values('attraction', ascending=False).head(20).plot(
            kind='bar', 
            x='zone_id', 
            y='attraction', 
            ax=ax,
            color='orange'
        )
        plt.title(f'吸引力最高的20个电动汽车充电区域 ({timestamp})')
        plt.xlabel('区域ID')
        plt.ylabel('吸引力值')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
        return fig
    
    # 2. 修改 EVHuffVisualizer 类的 plot_flow_map 方法
    def plot_flow_map(self, origin_id, timestamp, save_path=None, max_dests=20):
        """绘制特定时间从特定起点的流量分布图 - 优化版本"""
        # 计算流量概率
        probabilities = self.model.predict_probability(origin_id, timestamp)
        
        # 只保留概率最高的几个目的地
        top_probs = probabilities.sort_values(ascending=False).head(max_dests)
        
        # 创建用于绘图的DataFrame
        flow_data = pd.DataFrame({
            'zone_id': top_probs.index,
            'probability': top_probs.values
        })
        
        # 创建条形图
        fig, ax = plt.subplots(figsize=(12, 6))
        flow_data.plot(
            kind='bar', 
            x='zone_id', 
            y='probability', 
            ax=ax,
            color='skyblue'
        )
        plt.title(f'从区域 {origin_id} 出发最可能的{max_dests}个目的地 ({timestamp})')
        plt.xlabel('目的地区域ID')
        plt.ylabel('概率')
        plt.xticks(rotation=45)
        
        if save_path:
            # 确保保存为png格式
            if save_path.endswith('.html'):
                save_path = save_path.replace('.html', '.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        return fig

# 六、集成应用
class DynamicHuffEVApp:
    """动态Huff EV应用"""
    
    def __init__(self, data_path='C:/Users/zhang/Desktop/DQN/data/'):
        """初始化完整应用"""
        # 初始化组件
        self.data_loader = UrbanEVDataLoader(data_path)
        
    def load_and_preprocess(self):
        """加载和预处理所有数据"""
        self.data_loader.load_datasets()
        self.data_loader.preprocess()
        self.data_loader.add_temporal_features()
        self.data_loader.create_spatiotemporal_features()
        
        # 在数据加载后初始化模型
        self.huff_model = DynamicHuffEV(self.data_loader)
        self.st_predictor = STPredictor(self.data_loader)
        self.optimizer = ModelOptimizer(self.huff_model, self.data_loader)
        self.visualizer = EVHuffVisualizer(self.huff_model, self.data_loader)
        
        return self
    
    def optimize_model(self, n_calls=50):
        """优化模型参数"""
        return self.optimizer.optimize(n_calls=n_calls)
    
    def run_single_prediction(self, origin_id, timestamp, demand=None, candidate_zones=None):
        """对特定时间的起点运行单次预测"""
        if demand is None:
            demand = self.st_predictor.predict_zone_demand(origin_id, timestamp)
            
        return self.huff_model.predict_flow(origin_id, demand, timestamp, candidate_zones)
    
    def run_batch_predictions(self, origin_ids, timestamp, demands=None, candidate_zones=None):
        """对特定时间的多个起点运行预测"""
        if demands is None:
            demands = {}
            for oid in origin_ids:
                demands[oid] = self.st_predictor.predict_zone_demand(oid, timestamp)
                
        return self.huff_model.predict_all_flows(demands, timestamp, candidate_zones)
    
    def evaluate_charging_station_scenario(self, scenario_zones, timestamp, origins=None, candidate_zones=None):
        """评估充电站布局方案"""
        # 备份原始区域数据
        original_zones = self.data_loader.zones.copy()
        
        # 应用方案更改
        for zone_id, capacity in scenario_zones.items():
            idx = self.data_loader.zones.index[self.data_loader.zones['zone_id'] == zone_id].tolist()
            if idx:
                self.data_loader.zones.loc[idx[0], 'capacity'] = capacity
                self.data_loader.zones.loc[idx[0], 'charging_capacity'] = capacity * 0.85
                
        # 若未提供起点，则使用所有区域
        if origins is None:
            origins = self.data_loader.zones['zone_id'].unique()
            
        # 估计需求
        demands = {}
        for oid in origins:
            demands[oid] = self.st_predictor.predict_zone_demand(oid, timestamp)
            
        # 运行预测
        flows = self.huff_model.predict_all_flows(demands, timestamp, candidate_zones)
        
        # 计算指标
        metrics = {
            'total_flow': flows['flow'].sum(),
            'covered_demand': 0,
            'average_distance': 0,
            'max_utilization': 0
        }
        
        # 计算每个目的地的利用率
        dest_flows = flows.groupby('dest_id')['flow'].sum().reset_index()
        dest_flows = dest_flows.merge(
            self.data_loader.zones[['zone_id', 'charging_capacity']], 
            left_on='dest_id', 
            right_on='zone_id',
            how='left'
        )
        dest_flows['utilization'] = dest_flows['flow'] / dest_flows['charging_capacity']
        
        metrics['max_utilization'] = dest_flows['utilization'].max()
        metrics['avg_utilization'] = dest_flows['utilization'].mean()
        metrics['overloaded_stations'] = len(dest_flows[dest_flows['utilization'] > 1])
        
        # 计算平均距离
        total_distance = 0
        total_weight = 0
        
        for _, row in flows.iterrows():
            origin_id = row['origin_id']
            dest_id = row['dest_id']
            flow = row['flow']
            
            # 获取距离
            distance = self.huff_model._calculate_distance_from_coords(origin_id, dest_id)
            
            total_distance += distance * flow
            total_weight += flow
            
        if total_weight > 0:
            metrics['average_distance'] = total_distance / total_weight
            
        # 计算覆盖率（可以被服务的需求百分比）
        if sum(demands.values()) > 0:
            metrics['demand_coverage'] = metrics['total_flow'] / sum(demands.values())
        else:
            metrics['demand_coverage'] = 0
            
        # 恢复原始区域数据
        self.data_loader.zones = original_zones
        
        return metrics
    
    def visualize_results(self, timestamp, origin_id=None, save_dir=None):
        """创建模型结果的可视化"""
        figures = {}
        
        # 如果需要，创建保存目录
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 创建吸引力地图
        attraction_save_path = os.path.join(save_dir, f'attraction_map_{timestamp.strftime("%Y%m%d_%H%M")}.png') if save_dir else None
        figures['attraction_map'] = self.visualizer.plot_attraction_map(timestamp, save_path=attraction_save_path)
        
        # 如果提供了起点，创建流量地图
        if origin_id is not None:
            flow_save_path = os.path.join(save_dir, f'flow_map_from_{origin_id}_{timestamp.strftime("%Y%m%d_%H%M")}.html') if save_dir else None
            figures['flow_map'] = self.visualizer.plot_flow_map(origin_id, timestamp, save_path=flow_save_path)
            
        return figures

# 主执行程序
def main():
    """主函数，演示模型功能"""
    # 初始化应用
    app = DynamicHuffEVApp(data_path='C:/Users/zhang/Desktop/DQN/data/')
    app.load_and_preprocess()
    
    # 选择一个样本时间戳
    try:
        sample_timestamp = app.data_loader.occupancy_wide['time'].iloc[0]
        logger.info(f"使用时间戳: {sample_timestamp}")
    except (IndexError, AttributeError):
        sample_timestamp = pd.Timestamp('2022-01-01 08:00:00')
        logger.info(f"未找到实际时间戳，使用默认值: {sample_timestamp}")
    
    # 跳过参数优化，直接使用默认参数
    logger.info("使用默认模型参数，跳过优化步骤")
    
    # 选择少量区域运行示例
    if len(app.data_loader.zones) > 0:
        # 只选择前10个区域作为目的地
        candidate_zones = app.data_loader.zones['zone_id'].unique()[:10].tolist()
        logger.info(f"使用{len(candidate_zones)}个候选区域进行测试")
        
        # 只选择第一个区域作为起点
        sample_origin = app.data_loader.zones['zone_id'].iloc[0]
        logger.info(f"为起点区域{sample_origin}运行预测...")
        
        # 运行单次预测
        flows = app.run_single_prediction(
            origin_id=sample_origin, 
            timestamp=sample_timestamp,
            candidate_zones=candidate_zones
        )
        
        # 显示结果
        top_dests = flows.sort_values(ascending=False).head(5)
        logger.info(f"从区域{sample_origin}出发的前5个目的地:")
        for dest_id, flow in top_dests.items():
            logger.info(f"  区域 {dest_id}: {flow:.2f}")
        
        # 可视化结果 - 这一步可能会比较耗时
        logger.info("创建可视化...")
        try:
            app.visualize_results(
                timestamp=sample_timestamp, 
                origin_id=sample_origin, 
                save_dir='C:/Users/zhang/Desktop/DQN/results/'
            )
            logger.info("可视化创建完成")
        except Exception as e:
            logger.error(f"创建可视化时出错: {e}")
    
    logger.info("完成!")

if __name__ == "__main__":
    main()
