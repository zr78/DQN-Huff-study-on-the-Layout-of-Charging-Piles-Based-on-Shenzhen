import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from haversine import haversine
import os
import logging
import joblib
import time
from tqdm import tqdm
import warnings
import functools

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings("ignore")

# 添加缓存装饰器
def cache_result(func):
    """缓存函数结果的装饰器"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 创建缓存键
        key = str(args) + str(sorted(kwargs.items()))
        
        # 检查缓存
        if key in cache:
            return cache[key]
        
        # 计算结果
        result = func(*args, **kwargs)
        
        # 存入缓存
        cache[key] = result
        return result
    
    # 添加清除缓存的方法
    wrapper.clear_cache = lambda: cache.clear()
    
    return wrapper

class SimpleEVDataLoader:
    """用于加载和预处理电动汽车数据的简化加载器"""
    
    def __init__(self, data_path='./data/'):
        """初始化数据加载器"""
        self.data_path = data_path
        
    def load_datasets(self):
        """加载所有需要的数据集"""
        logger.info("加载数据集...")
        
        # 加载必要的文件
        files_to_load = ['inf.csv', 'distance.csv', 'occupancy.csv', 'e_price.csv']
        
        for file in tqdm(files_to_load, desc="加载数据文件"):
            try:
                if file == 'inf.csv':
                    self.inf = pd.read_csv(os.path.join(self.data_path, file))
                elif file == 'distance.csv':
                    self.distance_df = pd.read_csv(os.path.join(self.data_path, file))
                elif file == 'occupancy.csv':
                    self.occupancy_wide = pd.read_csv(os.path.join(self.data_path, file))
                    self.occupancy_wide['time'] = pd.to_datetime(self.occupancy_wide['time'])
                elif file == 'e_price.csv':
                    self.e_price_wide = pd.read_csv(os.path.join(self.data_path, file))
                    self.e_price_wide['time'] = pd.to_datetime(self.e_price_wide['time'])
            except Exception as e:
                logger.warning(f"加载文件 {file} 时出错: {e}")
        
        logger.info("数据集加载完成")
        return self

    def preprocess(self):
        """预处理数据集"""
        logger.info("预处理数据集...")
        
        # 创建充电站信息表
        if hasattr(self, 'inf'):
            self.zones = self.inf.copy()
            # 根据实际数据结构调整字段名
            if 'station_id' in self.zones.columns and 'zone_id' not in self.zones.columns:
                self.zones.rename(columns={'station_id': 'zone_id'}, inplace=True)
                logger.info("将'station_id'重命名为'zone_id'")
        else:
            # 从occupancy数据创建基本区域信息
            if hasattr(self, 'occupancy_wide'):
                station_ids = [col for col in self.occupancy_wide.columns if col != 'time']  # 排除时间列
                self.zones = pd.DataFrame({'zone_id': station_ids})
                logger.info(f"从occupancy数据创建了包含{len(station_ids)}个区域的基本信息表")
            else:
                logger.warning("无法创建区域信息表，缺少必要数据")
                self.zones = pd.DataFrame(columns=['zone_id'])
        
        # 转换占用率数据为长格式
        if hasattr(self, 'occupancy_wide'):
            try:
                logger.info("转换占用率数据为长格式...")
                self.occupancy = pd.melt(
                    self.occupancy_wide,
                    id_vars=['time'],
                    var_name='station_id',
                    value_name='occupancy'
                )
                logger.info(f"转换后占用率数据包含 {len(self.occupancy)} 行")
            except Exception as e:
                logger.error(f"转换占用率数据时出错: {e}")
                self.occupancy = pd.DataFrame(columns=['time', 'station_id', 'occupancy'])
        else:
            self.occupancy = pd.DataFrame(columns=['time', 'station_id', 'occupancy'])
        
        # 转换电价数据为长格式
        if hasattr(self, 'e_price_wide'):
            try:
                logger.info("转换电价数据为长格式...")
                self.e_price = pd.melt(
                    self.e_price_wide,
                    id_vars=['time'],
                    var_name='station_id',
                    value_name='price'
                )
                logger.info(f"转换后电价数据包含 {len(self.e_price)} 行")
            except Exception as e:
                logger.error(f"转换电价数据时出错: {e}")
                self.e_price = pd.DataFrame(columns=['time', 'station_id', 'price'])
        else:
            self.e_price = pd.DataFrame(columns=['time', 'station_id', 'price'])
        
        # 确保zone_id是字符串类型
        if 'zone_id' in self.zones.columns:
            self.zones['zone_id'] = self.zones['zone_id'].astype(str)
            logger.info("确保zone_id是字符串类型")
        
        # 确保station_id与zone_id字段匹配
        if 'station_id' in self.occupancy.columns:
            self.occupancy['station_id'] = self.occupancy['station_id'].astype(str)
        if 'station_id' in self.e_price.columns:
            self.e_price['station_id'] = self.e_price['station_id'].astype(str)
        
        # 添加时间特征
        if len(self.occupancy) > 0:
            self.occupancy['hour'] = self.occupancy['time'].dt.hour
            self.occupancy['day_of_week'] = self.occupancy['time'].dt.dayofweek
            self.occupancy['weekend'] = self.occupancy['day_of_week'].isin([5, 6]).astype(int)
        
        # 预计算距离矩阵
        self._precompute_distance_matrix()
        
        logger.info(f"数据预处理完成, 包含 {len(self.zones)} 个充电站区域")
        return self
    
    def _precompute_distance_matrix(self):
        """预计算距离矩阵以提高性能"""
        logger.info("预计算距离矩阵...")
        
        # 如果已经有距离数据，则不需要预计算
        if hasattr(self, 'distance_df') and len(self.distance_df) > 0:
            logger.info("使用已有的距离数据")
            return
        
        # 检查是否有坐标数据
        if not hasattr(self, 'zones') or 'latitude' not in self.zones.columns or 'longitude' not in self.zones.columns:
            logger.warning("没有坐标数据，无法预计算距离矩阵")
            return
        
        # 创建距离矩阵
        zone_ids = self.zones['zone_id'].values
        n_zones = len(zone_ids)
        
        # 创建坐标数组
        coords = self.zones[['latitude', 'longitude']].values
        
        # 计算距离矩阵
        distances = np.zeros((n_zones, n_zones))
        for i in tqdm(range(n_zones), desc="计算距离矩阵"):
            for j in range(i+1, n_zones):
                dist = haversine((coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1]))
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 创建距离数据框
        distance_data = []
        for i, origin_id in enumerate(zone_ids):
            for j, dest_id in enumerate(zone_ids):
                if i != j:
                    distance_data.append({
                        'origin_id': origin_id,
                        'dest_id': dest_id,
                        'distance': distances[i, j]
                    })
        
        self.distance_df = pd.DataFrame(distance_data)
        logger.info(f"距离矩阵预计算完成，包含 {len(distance_data)} 个距离值")
    
    def get_zone_id_mapping(self):
        """创建一个从数字索引到zone_id的映射"""
        zone_ids = self.zones['zone_id'].values
        return {i: zone_id for i, zone_id in enumerate(zone_ids)}
    
    def get_zone_index_mapping(self):
        """创建一个从zone_id到数字索引的映射"""
        zone_ids = self.zones['zone_id'].values
        return {zone_id: i for i, zone_id in enumerate(zone_ids)}


class SimpleHuffEV:
    """简化版电动汽车充电站选择模型"""
    
    def __init__(self, data_loader):
        """初始化模型"""
        self.dl = data_loader
        # 默认参数设置
        self.params = {
            'alpha': 0.6,  # 容量吸引力权重
            'beta': 0.35,  # 价格敏感度
            'gamma': 0.4,  # 空间阻抗因子
            'lambda': 1.5,  # 距离衰减参数
            'theta': 0.25   # 时间依赖因子
        }
        # 创建zone_id映射
        self.zone_id_map = self.dl.get_zone_id_mapping()
        self.zone_index_map = self.dl.get_zone_index_mapping()
        
        # 添加缓存
        self.attraction_cache = {}  # 格式: (zone_id, hour_timestamp) -> attraction
        self.impedance_cache = {}   # 格式: (origin_id, dest_id, hour_timestamp) -> impedance
        self.probability_cache = {} # 格式: (origin_id, hour_timestamp, candidates_hash) -> probabilities
        self.cache_stats = {"hits": 0, "misses": 0}
                
    @cache_result
    def calculate_attraction(self, zone_id, timestamp):
        """计算特定时间下充电站区域的吸引力"""
        # 转换时间戳为小时级别缓存键
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp)
        elif isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        cache_ts = pd.Timestamp(timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
        
        # 缓存键
        cache_key = (zone_id, cache_ts)
        
        # 检查缓存
        if cache_key in self.attraction_cache:
            self.cache_stats["hits"] += 1
            return self.attraction_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        
        # 获取区域信息
        zone_info = None
        if hasattr(self.dl, 'zones'):
            zone_info = self.dl.zones[self.dl.zones['zone_id'] == zone_id]
            if len(zone_info) > 0:
                zone_info = zone_info.iloc[0]
        
        if zone_info is None:
            logger.warning(f"未找到区域ID {zone_id} 的信息")
            self.attraction_cache[cache_key] = 0.001
            return 0.001  # 返回默认最小值
        
        # 获取电价
        e_price = 1.0  # 默认电价
        if hasattr(self.dl, 'e_price'):
            e_price_data = self.dl.e_price[
                (self.dl.e_price['station_id'] == zone_id) & 
                (self.dl.e_price['time'] <= timestamp)
            ].sort_values('time', ascending=False)
            
            if len(e_price_data) > 0:
                e_price = e_price_data.iloc[0]['price']
        
        # 获取当前占用率
        occupancy = 50  # 默认占用率50%
        if hasattr(self.dl, 'occupancy'):
            occupancy_data = self.dl.occupancy[
                (self.dl.occupancy['station_id'] == zone_id) & 
                (self.dl.occupancy['time'] <= timestamp)
            ].sort_values('time', ascending=False)
            
            if len(occupancy_data) > 0:
                occupancy = occupancy_data.iloc[0]['occupancy']
        
        # 计算基本吸引力
        effective_capacity = 10  # 默认值
        if isinstance(zone_info, pd.Series):
            if 'charging_capacity' in zone_info:
                effective_capacity = zone_info['charging_capacity'] * (1 - occupancy/100)
            elif 'charge_count' in zone_info:
                effective_capacity = zone_info['charge_count'] * (1 - occupancy/100)
        
        # 使用确定性计算，不再使用随机因子
        # 改用基于zone_id的伪随机计算，确保每个充电站有不同但确定的吸引力
        zone_factor = 0.8 + 0.4 * (hash(str(zone_id)) % 1000) / 1000.0
        
        # 计算基础吸引力
        base_attraction = self.params['alpha'] * effective_capacity / (e_price + 0.01)
        
        # 时间依赖因子（工作日与周末模式不同）
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # 简化时间因子计算
        if is_weekend:
            time_factor = 0.7 + 0.3 * np.sin(np.pi * (hour - 10) / 12)
        else:
            morning_peak = np.exp(-0.5 * ((hour - 8) / 2) ** 2)
            evening_peak = np.exp(-0.5 * ((hour - 18) / 3) ** 2)
            time_factor = 0.5 + 0.5 * max(morning_peak, evening_peak)
        
        # 计算最终吸引力
        attraction = base_attraction * (1 + self.params['theta'] * time_factor) * zone_factor
        
        # 确保吸引力在合理范围内
        attraction = max(0.001, min(100.0, attraction))
        
        # 存入缓存
        self.attraction_cache[cache_key] = attraction
        return attraction
    
    @cache_result
    def calculate_impedance(self, origin_id, dest_id, timestamp):
        """计算从起点到目的地充电的阻抗"""
        # 转换为小时级别的缓存键
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp)
        elif isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
            
        cache_ts = pd.Timestamp(timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
        
        # 缓存键
        cache_key = (origin_id, dest_id, cache_ts)
        
        # 检查缓存
        if cache_key in self.impedance_cache:
            self.cache_stats["hits"] += 1
            return self.impedance_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        
        try:
            # 获取区域间距离
            distance = self._calculate_distance(origin_id, dest_id)
            
            # 获取目的地等待时间
            wait_time = 5.0  # 默认等待时间
            if hasattr(self.dl, 'occupancy'):
                occupancy_data = self.dl.occupancy[
                    (self.dl.occupancy['station_id'] == dest_id) & 
                    (self.dl.occupancy['time'] <= timestamp)
                ].sort_values('time', ascending=False)
                
                if len(occupancy_data) > 0:
                    occupancy = occupancy_data.iloc[0]['occupancy']
                    wait_time = max(0, np.exp(occupancy / 25) - 1)  # 指数关系
            
            # 确保值是有效数字
            if not np.isfinite(distance): distance = 10.0
            if not np.isfinite(wait_time): wait_time = 5.0
            
            # 计算总阻抗
            impedance = (
                self.params['gamma'] * (distance + 0.1) +  # 空间距离（加小常数避免除零）
                self.params['beta'] * wait_time  # 时间成本
            )
            
            result = max(0.01, impedance)  # 确保最小阻抗
            self.impedance_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"计算区域 {origin_id} 到 {dest_id} 的阻抗时出错: {e}")
            return 10.0  # 默认阻抗
    
    def _calculate_distance(self, origin_id, dest_id):
        """计算两个区域之间的距离"""
        # 方法1: 使用distance_df
        if hasattr(self.dl, 'distance_df'):
            try:
                row = self.dl.distance_df[
                    (self.dl.distance_df['origin_id'] == origin_id) & 
                    (self.dl.distance_df['dest_id'] == dest_id)
                ]
                if len(row) > 0 and 'distance' in row.columns:
                    return row.iloc[0]['distance']
            except Exception:
                pass
                
        # 方法2: 从坐标计算
        try:
            origin_info = self.dl.zones[self.dl.zones['zone_id'] == origin_id]
            dest_info = self.dl.zones[self.dl.zones['zone_id'] == dest_id]
            
            if len(origin_info) > 0 and len(dest_info) > 0 and 'latitude' in origin_info.columns:
                origin_lat = origin_info.iloc[0]['latitude']
                origin_lon = origin_info.iloc[0]['longitude']
                dest_lat = dest_info.iloc[0]['latitude']
                dest_lon = dest_info.iloc[0]['longitude']
                
                if all(np.isfinite(x) for x in [origin_lat, origin_lon, dest_lat, dest_lon]):
                    return haversine((origin_lat, origin_lon), (dest_lat, dest_lon))
        except Exception:
            pass
            
        # 默认距离
        return 10.0
    
    @cache_result
    def predict_probability(self, origin_id, timestamp, candidate_zones=None):
        """预测用户从起点选择各个候选区域的概率"""
        if candidate_zones is None:
            candidate_zones = self.dl.zones['zone_id'].unique()
            
        # 创建缓存键
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp)
        elif isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
            
        cache_ts = pd.Timestamp(timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
        candidates_hash = hash(tuple(sorted(candidate_zones)))
        cache_key = (origin_id, cache_ts, candidates_hash)
        
        # 检查缓存
        if cache_key in self.probability_cache:
            self.cache_stats["hits"] += 1
            return self.probability_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        
        attractions = []
        impedances = []
        zone_ids = []
        
        for dest_id in candidate_zones:
            try:
                a = self.calculate_attraction(dest_id, timestamp)
                c = self.calculate_impedance(origin_id, dest_id, timestamp)
                
                # 只添加有效值
                if a is not None and c is not None and np.isfinite(a) and np.isfinite(c):
                    zone_ids.append(dest_id)
                    attractions.append(float(a))
                    impedances.append(float(c))
            except Exception as e:
                logger.warning(f"计算区域 {dest_id} 的概率时出错: {e}")
        
        # 检查是否有有效数据
        if not attractions or not impedances:
            logger.warning(f"没有有效的区域数据，返回均匀分布")
            # 返回均匀分布
            return pd.Series([1.0/len(candidate_zones)] * len(candidate_zones), index=candidate_zones)
        
        # 转换为numpy数组
        attractions = np.array(attractions, dtype=float)
        impedances = np.array(impedances, dtype=float)
        
        # 处理零或负阻抗
        impedances[impedances <= 0] = 0.001
        
        # 计算效用
        utilities = attractions / np.power(impedances, self.params['lambda'])
        
        # 处理数值问题
        utilities = np.nan_to_num(utilities, nan=0.001, posinf=100, neginf=0.001)
        utilities = np.clip(utilities, 0.001, 100)  # 限制极值
        
        # 转换为概率
        sum_utilities = np.sum(utilities)
        if sum_utilities > 0:
            probabilities = utilities / sum_utilities
        else:
            probabilities = np.ones_like(utilities) / len(utilities)
        
        # 以Series形式返回结果
        result = pd.Series(probabilities, index=zone_ids)
        self.probability_cache[cache_key] = result
        return result
    
    def predict_flow(self, origin_id, demand, timestamp, candidate_zones=None):
        """预测从起点到目的地的电动汽车充电需求流"""
        probabilities = self.predict_probability(origin_id, timestamp, candidate_zones)
        flows = probabilities * demand
        return flows

    def generate_recommendations(self, user_location, timestamp, top_n=5):
        """为用户生成充电站推荐列表"""
        # 用户位置可以是一个区域ID或坐标
        if isinstance(user_location, (int, str)):
            origin_id = user_location
        else:
            # 如果是坐标，找最近的区域
            origin_id = self._find_nearest_zone(user_location)
        
        # 预测概率
        probabilities = self.predict_probability(origin_id, timestamp)
        
        # 添加等待时间信息
        recommendations = []
        for zone_id, prob in probabilities.items():
            wait_time = self._estimate_wait_time(zone_id, timestamp)
            recommendations.append({
                'zone_id': zone_id,
                'probability': prob,
                'estimated_wait': wait_time
            })
        
        # 排序并取前N个
        recommendations.sort(key=lambda x: x['probability'] / (x['estimated_wait'] + 1), reverse=True)
        return recommendations[:top_n]
    
    def _find_nearest_zone(self, location):
        """找到离给定位置最近的区域"""
        logger.info(f"正在查找最近的充电站区域...")
        
        # 如果location已经是区域ID，直接返回
        if isinstance(location, (int, str, np.integer)):
            logger.info(f"输入位置已经是区域ID: {location}")
            return location
            
        if not hasattr(self.dl, 'zones') or 'latitude' not in self.dl.zones.columns:
            # 如果没有坐标数据，返回第一个区域
            default_zone = self.dl.zones['zone_id'].iloc[0]
            logger.warning(f"没有坐标数据，使用默认区域: {default_zone}")
            return default_zone
        
        try:
            lat, lon = location
            min_dist = float('inf')
            nearest_zone = None
            
            logger.info(f"正在计算与位置({lat}, {lon})最近的充电站...")
            for _, zone in self.dl.zones.iterrows():
                if pd.notna(zone.get('latitude')) and pd.notna(zone.get('longitude')):
                    dist = haversine((lat, lon), (zone['latitude'], zone['longitude']))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_zone = zone['zone_id']
            
            if nearest_zone is not None:
                logger.info(f"找到最近的充电站区域: {nearest_zone}，距离: {min_dist:.2f}公里")
                return nearest_zone
        except Exception as e:
            logger.warning(f"计算最近区域时出错: {e}")
        
        default_zone = self.dl.zones['zone_id'].iloc[0]
        logger.warning(f"使用默认区域: {default_zone}")
        return default_zone
    
    def _estimate_wait_time(self, zone_id, timestamp):
        """估计特定时间特定区域的等待时间"""
        if hasattr(self.dl, 'occupancy'):
            data = self.dl.occupancy[
                (self.dl.occupancy['station_id'] == zone_id) & 
                (self.dl.occupancy['time'] <= timestamp)
            ].sort_values('time', ascending=False)
            
            if len(data) > 0:
                occupancy = data.iloc[0]['occupancy']
                # 使用简化公式估计等待时间（分钟）
                return max(0, np.exp(occupancy / 25) - 1) * 10
        
        return 5.0  # 默认等待时间

    def save_cache(self, file_path='model_cache.pkl'):
        """将缓存保存到磁盘"""
        logger.info(f"保存模型缓存至 {file_path}...")
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 准备保存数据
        cache_data = {
            'attraction_cache': self.attraction_cache,
            'impedance_cache': self.impedance_cache,
            'probability_cache': self.probability_cache,
            'params': self.params
        }
        
        # 保存到文件
        with open(file_path, 'wb') as f:
            joblib.dump(cache_data, f)
            
        logger.info(f"缓存已保存。包含{len(self.attraction_cache)}个吸引力, "
                   f"{len(self.impedance_cache)}个阻抗, "
                   f"{len(self.probability_cache)}个概率分布")
    
    def load_cache(self, file_path='model_cache.pkl'):
        """从磁盘加载缓存"""
        if not os.path.exists(file_path):
            logger.warning(f"缓存文件 {file_path} 不存在，使用空缓存")
            return False
            
        logger.info(f"从 {file_path} 加载模型缓存...")
        
        try:
            # 加载缓存
            with open(file_path, 'rb') as f:
                cache_data = joblib.load(f)
            
            # 更新模型缓存
            self.attraction_cache = cache_data['attraction_cache']
            self.impedance_cache = cache_data['impedance_cache']
            self.probability_cache = cache_data.get('probability_cache', {})
            self.params = cache_data.get('params', self.params)
            
            logger.info(f"缓存加载完成。包含{len(self.attraction_cache)}个吸引力, "
                       f"{len(self.impedance_cache)}个阻抗, "
                       f"{len(self.probability_cache)}个概率分布")
            return True
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return False

# 演示使用方法
def demo():
    # 设置数据路径
    data_loader = SimpleEVDataLoader(data_path='./data/')
    
    # 加载和预处理数据
    data_loader.load_datasets()
    data_loader.preprocess()
    
    # 创建模型
    model = SimpleHuffEV(data_loader)
    
    # 尝试加载缓存
    model.load_cache('model_cache.pkl')
    
    # 示例：计算吸引力
    timestamp = pd.Timestamp('2023-01-01 12:00:00')
    zone_id = data_loader.zones['zone_id'].iloc[0]
    attraction = model.calculate_attraction(zone_id, timestamp)
    print(f"区域 {zone_id} 在 {timestamp} 的吸引力: {attraction:.4f}")
    
    # 示例：生成推荐
    user_location = data_loader.zones['zone_id'].iloc[0]  # 使用第一个区域作为用户位置
    recommendations = model.generate_recommendations(user_location, timestamp)
    print("推荐充电站:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. 区域 {rec['zone_id']}: 概率 {rec['probability']:.2f}, 预计等待 {rec['estimated_wait']:.1f}分钟")
    
    # 保存缓存
    model.save_cache('model_cache.pkl')

if __name__ == "__main__":
    demo() 