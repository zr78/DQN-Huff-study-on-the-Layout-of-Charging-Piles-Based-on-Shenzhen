import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pickle

class ExpertKnowledge:
    """充电桩布局优化的专家知识模块"""
    
    def __init__(self):
        self.best_layout = None
        self.best_reward = -float('inf')
        self.layout_history = []
        self.reward_history = []
        self.expert_actions_taken = 0
        self.total_actions = 0
    
    def expert_initialization(self, env):
        """
        使用专家知识初始化充电桩布局
        
        Args:
            env: 环境对象
            
        Returns:
            list: 初始化的充电桩布局
        """
        # 计算总充电桩数量 - 使用正确的属性
        if hasattr(env, 'charging_piles'):
            # 使用现有充电桩列表的总和
            total_charging_piles = sum(env.charging_piles)
        else:
            # 如果没有现有布局，设置一个默认值
            total_charging_piles = int(env.num_stations * 1.5)  # 平均每个站点1.5个充电桩
        
        # 初始化布局 - 基于需求、可达性和吸引力
        init_layout = [0] * env.num_stations
        
        # 获取需求数据
        if hasattr(env, '_dynamic_demand'):
            demand = env._dynamic_demand()
        else:
            # 如果没有需求方法，创建一个均匀分布
            demand = [1] * env.num_stations
        
        # 获取可达性
        if hasattr(env, '_calculate_accessibility') and hasattr(env, 'station_pos'):
            accessibility = env._calculate_accessibility(env.station_pos)
        else:
            # 默认可达性均匀
            accessibility = [1] * env.num_stations
        
        # 获取站点吸引力
        if hasattr(env, 'station_attractions'):
            attractions = env.station_attractions
        elif hasattr(env, 'attractions'):
            attractions = env.attractions
        else:
            # 默认吸引力均匀
            attractions = [1] * env.num_stations
        
        # 确保所有列表长度一致
        min_len = min(len(demand), len(accessibility), len(attractions), env.num_stations)
        demand = demand[:min_len]
        accessibility = accessibility[:min_len]
        attractions = attractions[:min_len]
        
        # 计算组合分数
        scores = []
        for i in range(min_len):
            # 组合分数 = 0.5*需求 + 0.3*可达性 + 0.2*吸引力
            score = 0.5 * demand[i] + 0.3 * accessibility[i] + 0.2 * attractions[i]
            scores.append(score)
        
        # 归一化分数
        total_score = sum(scores) + 1e-10
        normalized_scores = [s / total_score for s in scores]
        
        # 按照分数分配充电桩
        remaining_piles = total_charging_piles
        
        # 首先，确保每个站点至少有一个充电桩
        for i in range(min_len):
            init_layout[i] = 1
            remaining_piles -= 1
        
        # 然后，按照归一化分数分配剩余充电桩
        for i in range(min_len):
            # 分配基于分数的充电桩数量
            allocated = int(normalized_scores[i] * remaining_piles)
            init_layout[i] += allocated
            
        # 分配任何剩余的充电桩到得分最高的站点
        if sum(init_layout) < total_charging_piles:
            remaining = total_charging_piles - sum(init_layout)
            sorted_indices = sorted(range(min_len), key=lambda i: scores[i], reverse=True)
            for i in range(remaining):
                init_layout[sorted_indices[i % min_len]] += 1
        
        # 确保不超过站点容量限制
        if hasattr(env, 'station_capacity'):
            for i in range(min_len):
                if i < len(env.station_capacity):
                    init_layout[i] = min(init_layout[i], env.station_capacity[i])
        
        return init_layout

    
    def calculate_expert_guidance_reward(self, env):
        """计算基于专家知识的指导奖励
        
        Args:
            env: 充电环境实例
            
        Returns:
            float: 基于专家知识的奖励值
        """
        expert_reward = 0.0
        
        # 1. 需求匹配奖励 - 惩罚充电桩与需求的不匹配
        demand = env._dynamic_demand()
        demand_array = np.array(demand, dtype=float) + 1e-5  # 避免除零
        charging_piles_array = np.array(env.charging_piles, dtype=float)
        
        # 计算每个站点的供需比
        supply_demand_ratio = charging_piles_array / demand_array
        
        # 惩罚比例过高或过低的站点
        oversupply_penalty = -0.2 * np.sum(np.maximum(0, supply_demand_ratio - 1.5))
        undersupply_penalty = -0.3 * np.sum(np.maximum(0, 1.0 - supply_demand_ratio))
        
        # 鼓励适度的供需比 (0.8-1.2范围内最优)
        optimal_supply = np.sum((supply_demand_ratio >= 0.8) & (supply_demand_ratio <= 1.2))
        optimal_ratio_reward = 0.5 * optimal_supply / env.num_stations
        
        # 2. 考虑站点的地理位置重要性
        if hasattr(env, 'station_importance'):
            importance_array = np.array(env.station_importance, dtype=float)
            # 鼓励在重要站点投放更多充电桩
            importance_match = np.sum(importance_array * charging_piles_array) / np.sum(charging_piles_array)
            importance_reward = 0.4 * importance_match
        else:
            importance_reward = 0
        
        # 3. 空间分布合理性 - 惩罚充电桩分布过于集中
        if hasattr(env, 'station_pos'):
            # 计算充电桩分布的基尼系数
            sorted_piles = np.sort(charging_piles_array)
            cumsum_piles = np.cumsum(sorted_piles)
            gini = (np.sum((2 * np.arange(1, env.num_stations + 1) - env.num_stations - 1) * sorted_piles) / 
                    (env.num_stations * np.sum(sorted_piles)))
            
            # 惩罚过高的不平等性 (适度的不平等是可以接受的，但过高不行)
            distribution_reward = -0.3 * max(0, gini - 0.4)
        else:
            distribution_reward = 0
        
        # 4. 确保最低服务水平 - 每个站点至少有一定数量的充电桩
        min_piles_required = 1  # 每个站点最低要求
        stations_below_min = np.sum(charging_piles_array < min_piles_required)
        min_service_penalty = -0.2 * stations_below_min / env.num_stations
        
        # 5. 历史表现相关 - 如果有历史效果好的布局，鼓励与其相似
        if self.best_layout is not None:
            historical = np.array(self.best_layout, dtype=float)
            similarity = 1.0 - np.sum(np.abs(charging_piles_array - historical)) / (2 * np.sum(historical))
            historical_reward = 0.2 * similarity
        else:
            historical_reward = 0
        
        # 组合所有专家知识奖励
        expert_reward = (oversupply_penalty + undersupply_penalty + optimal_ratio_reward + 
                        importance_reward + distribution_reward + min_service_penalty + 
                        historical_reward)
        
        return expert_reward
    
    def combine_rewards(self, original_reward, expert_reward, episode_count):
        """结合原始奖励和专家奖励
        
        Args:
            original_reward: 原始环境奖励
            expert_reward: 专家知识奖励
            episode_count: 当前训练回合数
            
        Returns:
            float: 最终组合奖励
        """
        # 随着训练进行，逐渐减少专家奖励的权重
        expert_weight = max(0.2, 0.8 - 0.6 * min(1.0, episode_count / 200))
        
        final_reward = (1 - expert_weight) * original_reward + expert_weight * expert_reward
        
        # 调试输出
        if episode_count % 10 == 0:
            print(f"奖励组成: 原始={original_reward:.2f}, 专家={expert_reward:.2f}, "
                  f"专家权重={expert_weight:.2f}, 最终={final_reward:.2f}")
        
        return final_reward
    
    def select_expert_action(self, state, env, epsilon):
        """根据专家知识选择动作
        
        Args:
            state: 当前环境状态
            env: 环境实例
            epsilon: 当前探索率
            
        Returns:
            tuple: (是否使用专家动作, 专家选择的动作)
        """
        # 随着训练进行，减少专家干预
        expert_threshold = max(0.1, 0.8 * epsilon)  # 探索率高时更多使用专家
        
        if np.random.random() < expert_threshold:
            # 使用专家规则选择动作
            num_stations = env.num_stations
            
            # 将状态重组为每个站点的特征
            station_states = []
            for i in range(num_stations):
                # 假设状态向量中每6个元素代表一个站点的特征
                station_states.append(state[i*6:(i+1)*6])
            
            # 提取站点特征
            # 假设：station_states[i][0] = 当前充电桩数量
            #      station_states[i][1] = 需求量
            #      station_states[i][2] = 站点容量
            #      station_states[i][3] = 可达性
            #      station_states[i][4] = 吸引力
            #      station_states[i][5] = 利用率
            
            current_piles = np.array([s[0] for s in station_states])
            demand = np.array([s[1] for s in station_states])
            capacity = np.array([s[2] for s in station_states])
            accessibility = np.array([s[3] for s in station_states])
            attraction = np.array([s[4] for s in station_states])
            utilization = np.array([s[5] for s in station_states])
            
            # 计算供需差距
            supply_demand_gap = demand - current_piles
            
            # 计算站点评分
            station_scores = (
                0.4 * supply_demand_gap +  # 需求缺口
                0.2 * (capacity - current_piles) +  # 剩余容量
                0.2 * accessibility +  # 可达性
                0.1 * attraction +  # 吸引力
                0.1 * (1 - utilization)  # 低利用率惩罚
            )
            
            # 专家决策逻辑
            if np.random.random() < 0.7:  # 70%概率增加充电桩
                # 找出评分最高的站点，增加充电桩
                best_station = np.argmax(station_scores)
                # 动作编码：0到num_stations-1为增加操作
                self.expert_actions_taken += 1
                self.total_actions += 1
                return True, best_station
            else:  # 30%概率减少充电桩
                # 找出评分最低且有充电桩的站点，减少充电桩
                removal_scores = -station_scores
                # 确保只考虑有充电桩的站点
                for i in range(len(removal_scores)):
                    if current_piles[i] <= 0:
                        removal_scores[i] = -np.inf
                
                if np.max(removal_scores) > -np.inf:
                    worst_station = np.argmax(removal_scores)
                    # 动作编码：num_stations到2*num_stations-1为减少操作
                    self.expert_actions_taken += 1
                    self.total_actions += 1
                    return True, worst_station + num_stations
                else:
                    # 如果没有可减少的站点，改为增加操作
                    best_station = np.argmax(station_scores)
                    self.expert_actions_taken += 1
                    self.total_actions += 1
                    return True, best_station
        
        # 不使用专家规则
        self.total_actions += 1
        return False, None
    
    def track_environment_state(self, env):
        """追踪环境状态，记录最佳布局
        
        Args:
            env: 环境实例
        """
        # 记录当前状态
        current_state = {
            'charging_piles': env.charging_piles.copy(),
            'demand': env._dynamic_demand(),
            'reward': env._calculate_reward()
        }
        
        self.layout_history.append(current_state)
        self.reward_history.append(current_state['reward'])
        
        # 更新最佳布局
        if current_state['reward'] > self.best_reward:
            self.best_reward = current_state['reward']
            self.best_layout = env.charging_piles.copy()
            
            # 每当发现更好的布局时，尝试分析其特点
            self._analyze_best_layout(env)
            
    def _analyze_best_layout(self, env):
        """分析最佳布局的特点
        
        Args:
            env: 环境实例
        """
        if self.best_layout is None:
            return
            
        # 计算当前最佳布局的特点
        demand = env._dynamic_demand()
        
        # 计算供需比
        supply_demand_ratio = np.array(self.best_layout) / (np.array(demand) + 1e-5)
        
        # 计算统计特征
        avg_ratio = np.mean(supply_demand_ratio)
        std_ratio = np.std(supply_demand_ratio)
        min_ratio = np.min(supply_demand_ratio)
        max_ratio = np.max(supply_demand_ratio)
        
        # 获取充电桩站点特征
        if hasattr(env, 'station_attractions'):
            high_attraction_ratio = np.mean(supply_demand_ratio[np.array(env.station_attractions) > np.median(env.station_attractions)])
            low_attraction_ratio = np.mean(supply_demand_ratio[np.array(env.station_attractions) <= np.median(env.station_attractions)])
        else:
            high_attraction_ratio = low_attraction_ratio = avg_ratio
        
        # 保存分析结果，用于专家知识引导
        self.layout_analysis = {
            'avg_supply_demand_ratio': avg_ratio,
            'std_supply_demand_ratio': std_ratio,
            'min_supply_demand_ratio': min_ratio,
            'max_supply_demand_ratio': max_ratio,
            'high_attraction_ratio': high_attraction_ratio,
            'low_attraction_ratio': low_attraction_ratio
        }
        
        print(f"\n最佳布局分析:")
        print(f"平均供需比: {avg_ratio:.2f}")
        print(f"供需比标准差: {std_ratio:.2f}")
        print(f"高吸引力站点供需比: {high_attraction_ratio:.2f}")
        print(f"低吸引力站点供需比: {low_attraction_ratio:.2f}")
    
    def save_best_layout(self, save_path="output"):
        """保存最佳布局数据
        
        Args:
            save_path: 保存路径
        """
        if self.best_layout is None:
            print("没有最佳布局可保存")
            return
            
        os.makedirs(save_path, exist_ok=True)
        
        # 保存为二进制文件
        with open(os.path.join(save_path, "best_expert_layout.pkl"), 'wb') as f:
            pickle.dump({
                'layout': self.best_layout,
                'reward': self.best_reward,
                'analysis': self.layout_analysis if hasattr(self, 'layout_analysis') else None
            }, f)
            
        print(f"最佳布局已保存至 {save_path}/best_expert_layout.pkl")
        
        # 保存为可读文本
        with open(os.path.join(save_path, "best_expert_layout.txt"), 'w') as f:
            f.write(f"最佳奖励: {self.best_reward}\n\n")
            f.write("站点充电桩分布:\n")
            for i, piles in enumerate(self.best_layout):
                f.write(f"站点 {i}: {piles} 个充电桩\n")
                
            if hasattr(self, 'layout_analysis'):
                f.write("\n布局分析:\n")
                for key, value in self.layout_analysis.items():
                    f.write(f"{key}: {value:.2f}\n")
    
    def plot_expert_metrics(self, output_path):
        """绘制专家指标图表"""
        # 修改条件判断
        if hasattr(self, 'best_layout') and self.best_layout is not None:
            # 绘制最佳布局图表
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(self.best_layout)), self.best_layout)
            plt.title('最佳充电桩布局')
            plt.xlabel('站点索引')
            plt.ylabel('充电桩数量')
            plt.savefig(os.path.join(output_path, 'best_layout.png'))
            plt.close()
            
            # 如果有历史数据
            if hasattr(self, 'history') and len(self.history) > 0:
                # 绘制历史奖励图表
                plt.figure(figsize=(10, 6))
                rewards = [h.get('reward', 0) for h in self.history]
                plt.plot(rewards)
                plt.title('奖励历史')
                plt.xlabel('回合')
                plt.ylabel('奖励')
                plt.savefig(os.path.join(output_path, 'reward_history.png'))
                plt.close()
                
                # 其他可能的图表...
        else:
            print("没有最佳布局数据可供绘图")


