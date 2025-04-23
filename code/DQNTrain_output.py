import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 设置最后一个回合数（根据您的训练调整，例如299）
episode = 299

# 1. 计算充电站利用率
print("=== 计算充电站利用率 ===")
summary_file = f"output/station_summary_ep{episode}.csv"
try:
    # 加载充电站摘要文件
    df = pd.read_csv(summary_file)
    # 计算每站利用率：charging_piles / capacity
    df['utilization'] = df['charging_piles'] / df['capacity']
    mean_utilization = df['utilization'].mean()
    std_utilization = df['utilization'].std()
    print(f"平均利用率: {mean_utilization:.2f}")
    print(f"利用率标准差: {std_utilization:.2f}")

    # 可视化利用率分布（可选）
    plt.hist(df['utilization'], bins=20, color='skyblue', edgecolor='black')
    plt.title('充电站利用率分布')
    plt.xlabel('利用率')
    plt.ylabel('站点数')
    plt.savefig('output/utilization_distribution.png')
    plt.close()
except FileNotFoundError:
    print(f"错误：文件 {summary_file} 不存在，请检查训练输出。")

# 2. 计算奖励提升
print("\n=== 计算奖励提升 ===")
try:
    # 加载奖励历史文件
    rewards = np.load('output/reward_history.npy')
    initial_reward = np.mean(rewards[:10])  # 前10回合平均奖励
    final_reward = np.mean(rewards[-10:])   # 最后10回合平均奖励
    improvement = (final_reward - initial_reward) / abs(initial_reward) * 100
    print(f"初始平均奖励: {initial_reward:.2f}")
    print(f"最终平均奖励: {final_reward:.2f}")
    print(f"奖励提升百分比: {improvement:.2f}%")

    # 可视化奖励曲线（可选）
    plt.plot(rewards, color='green')
    plt.title('训练奖励曲线')
    plt.xlabel('回合数')
    plt.ylabel('奖励值')
    plt.savefig('output/reward_curve.png')
    plt.close()
except FileNotFoundError:
    print("错误：reward_history.npy 文件不存在，无法计算奖励提升。")

# 3. 计算需求满足率
print("\n=== 计算需求满足率 ===")
try:
    # 加载最佳环境状态
    with open('output/best_env_state.pkl', 'rb') as f:
        env_state = pickle.load(f)
    charging_piles = env_state['charging_piles']
    demand = env_state['demand']
    # 计算需求满足比例，限制在0-1之间
    demand_ratio = charging_piles / (demand + 1e-5)  # 避免除零
    satisfaction = np.mean(np.clip(demand_ratio, 0, 1))
    print(f"需求满足率: {satisfaction:.2f}")
except FileNotFoundError:
    print("错误：best_env_state.pkl 文件不存在，无法计算需求满足率。")

# 4. 专家知识验证
print("\n=== 专家知识验证 ===")
try:
    # 导入环境模块（需替换为您的实际模块名）
    from DQNtrain_model import DynamicChargingEnv
    env = DynamicChargingEnv(data_path="data")
    env.charging_piles = df['charging_piles'].values
    # 调用验证方法
    result_df = env.validate_against_analysis()
    print("专家知识匹配结果：")
    print(result_df)
except ImportError:
    print("错误：无法导入环境模块，请检查模块名称。")
except AttributeError:
    print("错误：环境类中未定义 validate_against_analysis 方法。")
except NameError:
    print("错误：请先确保 station_summary 数据已加载。")
    
#5. 可视化选项（可选，需根据您的代码调整）
print("\n=== 生成可视化图表 ===")
try:
    from DQNtrain_model import Visualization  # 替换为实际模块名
    Visualization.plot_charging_stations(env, episode=episode, save_path='output')
    Visualization.plot_heatmap(env, episode=episode, save_path='output')
    print("可视化图表已保存至 output 目录。")
except ImportError:
    print("错误：无法导入可视化模块，请检查模块名称。")
