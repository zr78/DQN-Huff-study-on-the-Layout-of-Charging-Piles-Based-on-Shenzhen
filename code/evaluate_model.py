import torch
import numpy as np
import pickle
import os
import time

# --- 导入必要的组件 ---
# 假设你的环境在 'DQNtrain_model.py' 中
from DQNtrain_model import DynamicChargingEnv
# 假设你的 DQN 代理在 'DQNtrain_model.py' 中
from DQNtrain_model import DQNAgent
# 假设你的专家知识在 'expert_knowledge.py' 中
# 评估时可能不需要 ExpertKnowledge 类本身，
# 除非环境在初始化时需要它。
# from expert_knowledge import ExpertKnowledge # 如果环境需要，取消注释

# --- 配置参数（必须与训练时匹配） ---
# 环境配置
env_config = {
    # 'num_stations': 10, # 通常不需要，除非无法加载 station_inf.csv
    'area_size': 1000,      # 必须与训练时使用的 area_size 匹配
    'data_path': "data",    # 必须指向包含训练数据的文件夹
    # 添加你的 DynamicChargingEnv 初始化所需的任何其他参数
}

# DQN Agent 配置
# ==================>> 重要: 替换下面的占位符 <<==================
# 基于 env.num_stations = 1682 重新计算！请务必核实计算逻辑！
state_dim = 10092 # TODO: 仔细检查！(示例值 = 6 * 1682)
action_dim = 3364 # TODO: 仔细检查！(示例值 = 2 * 1682)
# =================================================================

hidden_dim = 128             # 这个值之前确认是正确的
# !! 模型路径已修改 !!
model_path = 'output/best_model.pth' # 你的最佳模型路径
# !! 修改专家布局路径 !!
expert_layout_path = 'output/best_expert_layout.pkl' # 专家布局路径


# 评估配置
num_eval_episodes = 10        # 评估运行的回合数
max_steps_per_episode = 100   # 每回合最大步数 (防止无限循环)
render_env = False            # 是否在评估时渲染环境 (如果你的环境实现了 render 方法)
render_delay = 0.1            # 渲染时的延迟 (秒)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper function to process state (if needed) ---
# 根据你在训练中处理状态的方式调整此函数
def process_state(state_data): # Renamed argument for clarity
    """将环境返回的状态数据 (NumPy array) 处理成 DQN 需要的张量。"""
    try:
        # Ensure it's a NumPy array (it should be, based on debug output)
        state_array = np.array(state_data, dtype=np.float32)
        # state_array should now have shape (10092,)

        # Convert directly to FloatTensor, add batch dimension, and move to device
        # This should result in shape [1, 10092]
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(device)
        return state_tensor
    except Exception as e:
        # It's good practice to print the problematic data
        print(f"Error processing state data: {state_data}")
        print(f"Exception: {e}")
        # Return a zero tensor of the correct shape on error
        return torch.zeros((1, state_dim), device=device)


# --- 初始化 ---
print("Initializing environment and agent...")
try:
    env = DynamicChargingEnv(**env_config)
    print(f"Environment initialized with {env.num_stations} stations.")
    # 更新警告检查逻辑 (可选，但推荐)
    if state_dim != 6 * env.num_stations: # 使用你确认的计算逻辑
         print(f"Warning: state_dim ({state_dim}) might not match expected calculation (e.g., 6 * {env.num_stations} = {6*env.num_stations}). Double-check state_dim calculation.")
    if action_dim != 2 * env.num_stations: # 使用你确认的计算逻辑
         print(f"Warning: action_dim ({action_dim}) might not match expected calculation (e.g., 2 * {env.num_stations} = {2*env.num_stations}). Double-check action_dim calculation.")

except Exception as e:
    print(f"Error initializing environment: {e}")
    exit()

try:
    # !! 再次修改 DQNAgent 调用，移除 device 参数 !!
    # 现在只传递 state_dim, action_dim, hidden_dim
    dqn_agent = DQNAgent(state_dim, action_dim, hidden_dim)
except Exception as e:
    # 更新 state_dim 和 action_dim 的值以匹配你当前的设置
    print(f"Error initializing DQNAgent (check state_dim={state_dim}, action_dim={action_dim}): {e}")
    exit()


# --- 加载模型 ---
if os.path.exists(model_path):
    print(f"Loading model weights from: {model_path}")
    try:
        # !! 确保这里的 device 与模型保存时一致，或让 PyTorch 自动处理 !!
        # map_location=device 仍然是需要的，告诉 PyTorch 在哪里加载模型
        dqn_agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        dqn_agent.policy_net.to(device) # !! 显式将网络移动到目标设备 !!
        dqn_agent.policy_net.eval() # 设置为评估模式
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights (check state_dim/action_dim match saved model): {e}")
        exit()
else:
    print(f"Error: Model file not found at {model_path}")
    exit()

# --- 评估循环 ---
print(f"\nRunning evaluation for {num_eval_episodes} episodes...")
all_episode_rewards = []
final_layouts = []
total_steps = 0

for episode in range(num_eval_episodes):
    state_tuple = env.reset() # 获取初始状态元组

    # <<<--- 添加打印语句: 检查 env.reset() 的输出 ---<<<
    print(f"Debug: Initial state_tuple type: {type(state_tuple)}")
    print(f"Debug: Initial state_tuple content (first 20 elements if long): {str(state_tuple)[:200]}") # 打印部分内容防止过长
    # >>>------------------------------------------------->>>

    episode_reward = 0
    done = False
    step_count = 0

    while not done and step_count < max_steps_per_episode:
        if render_env and hasattr(env, 'render'):
            env.render()
            time.sleep(render_delay)

        # 1. 处理 *当前* 的 state_tuple 为网络所需的张量
        state = process_state(state_tuple) # process_state 内部仍用 state_tuple[0] (待确认)

        # <<<--- 调试打印语句: 检查 process_state 的输出 ---<<<
        print(f"Debug: Shape of state tensor being fed to network: {state.shape}")
        # >>>--------------------------------------------------->>>

        # 2. 使用处理后的 state 从网络获取动作
        with torch.no_grad(): # 评估时不需要计算梯度
            try:
                # 使用正确的网络属性名（例如 policy_net）
                q_values = dqn_agent.policy_net(state)
                # 选择 Q 值最高的动作 (贪婪策略)
                action = q_values.argmax(dim=1).item()
                print(f"Debug: Chosen action: {action}") # 打印选择的动作
            except RuntimeError as e:
                 print(f"RuntimeError during network forward pass: {e}")
                 print(f"Input state tensor shape was: {state.shape}")
                 break # 发生错误时退出当前回合
            except Exception as e: # 捕获其他潜在错误
                print(f"Error during network forward pass or action selection: {e}")
                break

        # 3. 在环境中执行动作，获取 *下一个* 状态元组及其他信息
        try:
            next_state_tuple, raw_reward, done, info = env.step(action)

            # <<<--- 添加打印语句: 检查 env.step() 的输出 ---<<<
            print(f"Debug: Next state_tuple type: {type(next_state_tuple)}")
            print(f"Debug: Next state_tuple content (first 20 elements if long): {str(next_state_tuple)[:200]}") # 打印部分内容
            print(f"Debug: Raw reward: {raw_reward}, Done: {done}, Info: {info}")
            # >>>---------------------------------------------->>>

        except Exception as e:
            print(f"\nError during environment step (action={action}): {e}")
            print("Likely issue: action_dim mismatch or error in env.step() logic.")
            done = True # 强制结束此回合
            raw_reward = -1000 # 惩罚错误步骤
            next_state_tuple = state_tuple # 发生错误时保持当前 state_tuple

        # 4. 更新奖励和步数计数
        episode_reward += raw_reward # 累加环境奖励
        step_count += 1

        # 5. 为下一次迭代做准备：下一个状态成为当前状态
        state_tuple = next_state_tuple

    # --- while 循环结束后 ---
    all_episode_rewards.append(episode_reward)
    if hasattr(env, 'charging_piles'):
        final_layouts.append(env.charging_piles.copy())
    else:
         final_layouts.append("Layout info not available")

    total_steps += step_count
    print(f"Episode {episode + 1}/{num_eval_episodes} finished in {step_count} steps. Raw Reward: {episode_reward:.4f}")

# --- 结果 ---
if all_episode_rewards:
    average_reward = sum(all_episode_rewards) / len(all_episode_rewards)
    print(f"\n--- Evaluation Summary ---")
    print(f"Average Raw Environment Reward over {num_eval_episodes} episodes: {average_reward:.4f}")
    print(f"Average Steps per Episode: {total_steps / num_eval_episodes:.2f}")
else:
    print("\n--- Evaluation Summary ---")
    print("No episodes completed successfully.")

# 显示最后一个回合的布局示例
if final_layouts:
    print(f"\nFinal Layout/State from Last Episode:")
    print(final_layouts[-1])

# --- 与专家布局比较 (可选) ---
if os.path.exists(expert_layout_path):
    print(f"\n--- Expert Layout Comparison ---")
    try:
        with open(expert_layout_path, 'rb') as f:
            expert_layout = pickle.load(f)
        print(f"Loaded Expert Layout: ")
        print(expert_layout)

        # TODO: 如果你的环境支持，实现静态布局评估
        # 这需要一种方法来计算给定固定布局的环境奖励
        # 例如: expert_static_reward = env.calculate_reward_for_static_layout(expert_layout)
        #       dqn_final_static_reward = env.calculate_reward_for_static_layout(final_layouts[-1])
        #       print(f"Estimated Static Reward for Expert Layout: {expert_static_reward}")
        #       print(f"Estimated Static Reward for Last DQN Layout: {dqn_final_static_reward}")
        print("\nNote: Static reward calculation for direct comparison not implemented.")
        print("Compare the average dynamic DQN reward and the final layouts visually/conceptually with the expert layout.")

    except Exception as e:
        print(f"Could not load or process expert layout from {expert_layout_path}: {e}")
else:
    print(f"\nExpert layout file not found at {expert_layout_path}. Skipping comparison.")








# 关闭环境（如果需要）
if hasattr(env, 'close'):
    env.close()

print("\nEvaluation complete.")