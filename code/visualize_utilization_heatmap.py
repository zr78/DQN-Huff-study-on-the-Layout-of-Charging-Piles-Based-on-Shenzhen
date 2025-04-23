"""
基于DQN优化结果生成时空热力图的独立脚本（改进版）
# 笔记：充电站利用率热力图可视化脚本

## 脚本目的 (visualize_utilization_heatmap.py)

该 Python 脚本的主要目的是：

1.  **加载数据**: 读取与充电站相关的各种数据，包括站点基础信息（位置坐标）、历史占用率（或利用率）、充电桩数量布局等。
2.  **加载布局**:
    *   加载一个“基准”布局（代表当前或专家设定的充电桩数量分布），从 `output/best_expert_layout.txt` 文件读取。
    *   加载一个“优化”布局（代表通过某种算法优化后的充电桩数量分布，目前代码中使用基准布局作为替代进行测试）。
3.  **计算利用率**: 针对一个特定的时间点（例如，脚本中设定为中午 12 点），根据加载的占用率数据和两种不同的充电桩布局（基准 vs 优化），计算每个充电站的利用率。利用率通常表示为 `实际占用桩数 / 总桩数` 或直接使用处理后的占用率数据。
4.  **生成可视化图**: 创建两张地理散点图（通常称为热力图），将每个充电站表示为一个点，点的位置由其经纬度决定，点的颜色深浅代表该站在该特定小时和特定布局下的利用率高低。
5.  **保存图表**: 将生成的两张图（基准利用率图和优化利用率图）保存为 PNG 图片文件。

## 图表用途 (利用率热力图)

生成的两张图表（`baseline_utilization_heatmap_h12.png` 和 `optimized_utilization_heatmap_h12.png`）主要有以下用途：

1.  **空间分布可视化**: 直观展示充电站在地理空间上的分布情况。
2.  **利用率可视化**: 通过颜色（例如，从紫色/蓝色代表低利用率到黄色/绿色代表高利用率）清晰地显示在特定时间点（中午 12 点），各个充电站的繁忙程度。
3.  **布局对比**: 核心用途是比较“基准布局”和“优化布局”的效果。通过对比两张图：
    *   可以看出优化算法是否有效地提高了整体或特定区域的充电桩利用率。
    *   可以识别出在不同布局下，哪些站点的利用率过高（可能需要增加桩）或过低（可能资源浪费）。
    *   可以评估优化策略是否使得资源分配更加均衡。
4.  **识别瓶颈与闲置**: 帮助快速定位利用率持续过高（热点，可能服务不足）或过低（冷点，可能桩数过多或位置不佳）的区域。
5.  **辅助决策**: 为充电网络的规划、管理和优化提供直观的数据支持，帮助决策者评估不同布局方案的优劣。

**总结**: 该脚本和图表共同构成了一个分析工具，用于理解和比较不同充电桩布局方案下，充电站在特定时间的空间利用率表现，从而评估优化策略的有效性。

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.interpolate import Rbf
import seaborn as sns
import os

from DQNtrain_model import DynamicChargingEnv, DataLoader
from Huff import DynamicHuffEV, UrbanEVDataLoader 
import logging 

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UtilizationVisualizer:
    def __init__(self, data_path="data"):
        """初始化可视化工具，加载环境和数据"""
        try:
            self.env = DynamicChargingEnv(data_path=data_path)
            # Ensure data is loaded and preprocessed if the env doesn't do it automatically
            if not hasattr(self.env.data_loader, 'occupancy') or self.env.data_loader.occupancy is None:
                 logger.info("Data seems unloaded in env, running full load/preprocess sequence...")
                 self.env.data_loader.load_datasets().preprocess().add_temporal_features().create_spatiotemporal_features()
                 logger.info("Data loading/preprocessing complete via visualizer init.")

            # Get station info needed for plotting
            self.station_info = self.env.data_loader.station_inf[['station_id', 'longitude', 'latitude']].copy()
            # Ensure station_id is string type for consistent merging/lookup
            self.station_info['station_id'] = self.station_info['station_id'].astype(str)
            self.station_info.set_index('station_id', inplace=True)
            logger.info(f"从环境加载了 {len(self.station_info)} 个站点的坐标信息")

        except Exception as e:
            logger.error(f"初始化 UtilizationVisualizer 时出错: {e}", exc_info=True)
            raise # Re-raise the exception to stop execution if init fails

    def load_expert_layout(self, file_path):
        """
        从文本文件加载专家布局（基准布局）。
        适应包含标题行和描述性文本的格式，并在非站点数据行停止。
        """
        logger.info(f"尝试从 {file_path} 加载专家布局 (适应特定格式)...")
        pile_counts = []
        try:
            with open(file_path, 'r', encoding='gbk') as f: 
                lines = f.readlines()

            start_line_index = 3
            if len(lines) <= start_line_index:
                logger.error(f"文件 {file_path} 的行数不足，无法提取布局数据。")
                return None

            # Process lines containing pile counts
            for i, line in enumerate(lines[start_line_index:]):
                line_num = start_line_index + i + 1 # Actual line number in file
                line = line.strip()
                if not line:
                    continue # Skip blank lines

                # --- Add Check: Stop if line doesn't start with expected prefix ---
                if not line.startswith("站点 "): # Check if the line starts correctly
                    logger.info(f"在行 {line_num} 检测到非站点数据 ('{line[:30]}...')，停止解析布局。")
                    break # Exit the loop

                try:
                    parts = line.split(':')
                    if len(parts) == 2:
                        count_part = parts[1].strip()
                        space_index = count_part.find(' ')
                        if space_index != -1:
                           count_str = count_part[:space_index]
                           pile_counts.append(int(count_str))
                        else:
                           # Fallback for "X个充电桩" without space
                           import re
                           match = re.match(r"\d+", count_part)
                           if match:
                               pile_counts.append(int(match.group(0)))
                           else:
                               logger.warning(f"无法从行 {line_num} 解析充电桩数量 (格式错误): '{line}'")
                               # --- Add Break: Stop if parsing fails for a line that should be data ---
                               logger.info(f"由于行 {line_num} 格式错误，停止解析布局。")
                               break # Exit the loop if format is wrong after "站点 " prefix
                    else:
                        logger.warning(f"无法从行 {line_num} 解析格式 (缺少 ':'): '{line}'")
                        # --- Add Break: Stop if parsing fails for a line that should be data ---
                        logger.info(f"由于行 {line_num} 格式错误，停止解析布局。")
                        break # Exit the loop if format is wrong after "站点 " prefix

                except ValueError:
                    logger.warning(f"在行 {line_num} 遇到非整数值: '{line}'")
                    # --- Add Break: Stop if parsing fails for a line that should be data ---
                    logger.info(f"由于行 {line_num} 格式错误，停止解析布局。")
                    break # Exit the loop if conversion fails
                except Exception as parse_err:
                    logger.warning(f"解析行 {line_num} 时出错: '{line}' - {parse_err}")
                     # --- Add Break: Stop if parsing fails for a line that should be data ---
                    logger.info(f"由于行 {line_num} 解析时出错，停止解析布局。")
                    break # Exit on other parsing errors

            # --- Check if the correct number of stations was parsed ---
            # This is an optional sanity check within this function
            expected_stations = len(self.env.data_loader.station_ids) # Get expected count
            if len(pile_counts) != expected_stations:
                logger.warning(f"解析得到的布局值数量 ({len(pile_counts)}) 与预期的站点数量 ({expected_stations}) 不匹配。请检查文件 {file_path} 的格式。")
                # Depending on requirements, you might want to return None here
                # return None

            if not pile_counts:
                logger.error(f"未能从文件 {file_path} 提取任何有效的充电桩数量。")
                return None

            pile_counts_array = np.array(pile_counts, dtype=int)
            logger.info(f"成功从文件解析了 {len(pile_counts_array)} 个布局值。")

            return {'layout': pile_counts_array}

        except FileNotFoundError:
            logger.error(f"专家布局文件未找到: {file_path}")
            return None
        except UnicodeDecodeError as ude:
             logger.error(f"文件 {file_path} 的编码错误: {ude}. 请检查文件是否以 GBK 或 UTF-8 保存。")
             return None
        except Exception as e:
            logger.error(f"加载专家布局时发生意外错误: {e}", exc_info=True)
            return None
        
    def _load_models(self):
        """加载必要的模型和数据"""
        data_loader = UrbanEVDataLoader(self.data_path)
        data_loader.load_datasets().preprocess().add_temporal_features().create_spatiotemporal_features()
        self.huff_model = DynamicHuffEV(data_loader)
        self.station_data = DataLoader.load_charging_station_data(self.data_path)
        self.station_pos = self.station_data['positions']
        self.num_stations = self.station_data['count']

    def _load_map(self):
        """加载深圳市地图"""
        try:
            self.shenzhen_map = gpd.read_file(self.map_path)
        except Exception as e:
            print(f"错误：无法加载地图文件 {self.map_path}，错误信息：{e}")
            self.shenzhen_map = None

    def _calculate_utilization(self, pile_counts, hour):
        """
        获取给定小时和充电桩布局下的利用率字典。
        假设 occupancy.csv 中的值已经是利用率。
        """
        utilization_dict = {}
        missing_occupancy_data = False
        occupancy_lookup = {} # Initialize empty

        # 1. Access Processed Occupancy Data (assumed to be utilization rates)
        try:
            # Use the env's data_loader
            if not hasattr(self.env, 'data_loader') or \
               not hasattr(self.env.data_loader, 'occupancy') or \
               self.env.data_loader.occupancy is None or \
               self.env.data_loader.occupancy.empty:
                logger.error("Occupancy data (as utilization) is not loaded or empty in env.data_loader.")
                missing_occupancy_data = True
            else:
                occupancy_df = self.env.data_loader.occupancy
                required_cols = ['hour', 'station_id', 'occupancy'] # 'occupancy' column now holds the rate
                if not all(col in occupancy_df.columns for col in required_cols):
                    logger.error(f"Occupancy data missing required columns: {required_cols}")
                    missing_occupancy_data = True
                else:
                    hourly_data = occupancy_df[occupancy_df['hour'] == hour]
                    # Create lookup: {station_id: utilization_rate}
                    # Ensure station_id in the index is the correct type (string or int) matching pile_counts keys
                    key_type = type(next(iter(pile_counts))) if pile_counts else str # Get type from dict keys, default str
                    hourly_data = hourly_data.astype({'station_id': key_type}) # Convert before creating Series/dict

                    occupancy_lookup = pd.Series(hourly_data.occupancy.values, index=hourly_data.station_id).to_dict()
                    logger.debug(f"Hour {hour}: Found occupancy/utilization data for {len(occupancy_lookup)} stations.")

        except Exception as e:
             logger.error(f"Error accessing or processing occupancy data for hour {hour}: {e}", exc_info=True)
             missing_occupancy_data = True

        # If occupancy data for the hour is missing entirely, we can't proceed
        if missing_occupancy_data and not occupancy_lookup:
             logger.warning(f"Cannot get utilization for hour {hour} due to missing occupancy data. Returning empty dict.")
             return {}

        # 2. Get Utilization for Stations in the Layout
        station_ids_in_layout = list(pile_counts.keys())
        logger.info(f"Getting utilization for {len(station_ids_in_layout)} stations in layout for hour {hour}.")

        # ... (rest of _calculate_utilization remains the same) ...
        for station_id, capacity in pile_counts.items():
            try:
                capacity = float(capacity)
            except (ValueError, TypeError):
                 logger.warning(f"Station {station_id}: Invalid capacity '{capacity}'. Setting utilization to NaN.")
                 utilization_dict[station_id] = float('nan')
                 continue

            if capacity > 0:
                utilization_rate = occupancy_lookup.get(station_id, float('nan')) # Default to NaN if not found
                if pd.isna(utilization_rate):
                    logger.warning(f"Station {station_id} (Capacity: {capacity}): Utilization data not found for hour {hour}. Setting to NaN.")
                    utilization_dict[station_id] = float('nan')
                else:
                    try:
                         utilization_rate = float(utilization_rate)
                         if not (0 <= utilization_rate <= 1):
                              logger.warning(f"Station {station_id}: Raw utilization rate {utilization_rate} out of bounds [0, 1]. Clamping.")
                              utilization_rate = max(0.0, min(1.0, utilization_rate))
                         utilization_dict[station_id] = utilization_rate
                    except (ValueError, TypeError):
                         logger.error(f"Station {station_id}: Invalid utilization rate '{utilization_rate}' found in data. Setting to NaN.")
                         utilization_dict[station_id] = float('nan')
            else:
                utilization_dict[station_id] = 0.0 # Station exists but has no capacity, utilization is 0

        if not utilization_dict:
             logger.warning(f"Utilization dictionary is empty after processing layout for hour {hour}.")

        return utilization_dict


    def _plot_heatmap(self, utilization_dict, title, filename):
        """绘制利用率热力图"""
        if not utilization_dict:
             logger.warning(f"利用率字典为空，无法绘制热力图 '{title}'。")
             return

        # Convert dict to Series for easier merging
        util_series = pd.Series(utilization_dict, name='utilization')
        util_series.index.name = 'station_id' # Ensure index has a name for merging

         # Ensure the index type matches station_info's index type (string)
        if not pd.api.types.is_string_dtype(util_series.index):
             util_series.index = util_series.index.astype(str)


        # Merge with station coordinates
        plot_data = pd.merge(self.station_info, util_series, left_index=True, right_index=True, how='inner')

        if plot_data.empty:
             logger.warning(f"利用率数据与站点坐标合并后为空，无法绘制热力图 '{title}'。")
             logger.warning(f"Station Info Index sample: {self.station_info.index[:5]}")
             logger.warning(f"Utilization Series Index sample: {util_series.index[:5]}")
             return

        logger.info(f"准备绘制热力图 '{title}'，包含 {len(plot_data)} 个站点的数据。")

        plt.figure(figsize=(12, 10))
        # Use station coordinates for scatter plot, color by utilization
        scatter = plt.scatter(plot_data['longitude'], plot_data['latitude'],
                              c=plot_data['utilization'], # Color based on utilization
                              cmap='viridis',             # Colormap (e.g., viridis, plasma, inferno, magma)
                              s=50,                       # Size of points
                              vmin=0.0, vmax=1.0,         # Fix color scale from 0 to 1
                              alpha=0.7)                  # Point transparency

        plt.colorbar(scatter, label='Utilization Rate') # Add color bar
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Ensure output directory exists
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)

        try:
            plt.savefig(full_path)
            logger.info(f"热力图已保存至: {full_path}")
        except Exception as e:
            logger.error(f"保存热力图时出错 {full_path}: {e}")
        plt.close() # Close the figure to free memory


    def generate_heatmaps(self, baseline_piles, optimized_piles, specific_hour=12):
        """生成基准和优化后布局在特定小时的利用率热力图"""
        logger.info(f"为小时 {specific_hour} 生成利用率热力图...")

        # Ensure the layout dictionaries are valid before proceeding
        if not isinstance(baseline_piles, dict):
             logger.error(f"基准布局不是字典类型 (是 {type(baseline_piles)})，无法计算利用率。")
             base_util_dict = {} # Set empty
        else:
             base_util_dict = self._calculate_utilization(baseline_piles, specific_hour)

        if not isinstance(optimized_piles, dict):
             logger.error(f"优化布局不是字典类型 (是 {type(optimized_piles)})，无法计算利用率。")
             opt_util_dict = {} # Set empty
        else:
             opt_util_dict = self._calculate_utilization(optimized_piles, specific_hour)


        # Check if dictionaries are populated before plotting
        if not base_util_dict:
             logger.error(f"无法生成基准利用率热力图，因为计算结果为空或基准布局无效 (小时 {specific_hour})。")
        else:
            valid_base_utils = [u for u in base_util_dict.values() if pd.notna(u)]
            avg_base_util = sum(valid_base_utils) / len(valid_base_utils) if valid_base_utils else float('nan')
            logger.info(f"小时 {specific_hour} - 基准平均利用率: {avg_base_util:.4f}")
            self._plot_heatmap(base_util_dict, f"Baseline Utilization (Hour {specific_hour})", f"baseline_utilization_heatmap_h{specific_hour}.png")

        if not opt_util_dict:
             logger.error(f"无法生成优化利用率热力图，因为计算结果为空或优化布局无效 (小时 {specific_hour})。")
        else:
             valid_opt_utils = [u for u in opt_util_dict.values() if pd.notna(u)]
             avg_opt_util = sum(valid_opt_utils) / len(valid_opt_utils) if valid_opt_utils else float('nan')
             logger.info(f"小时 {specific_hour} - 优化平均利用率: {avg_opt_util:.4f}")
             self._plot_heatmap(opt_util_dict, f"Optimized Utilization (Hour {specific_hour})", f"optimized_utilization_heatmap_h{specific_hour}.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try: # Add try block for better error reporting during setup
        visualizer = UtilizationVisualizer(data_path="data") # This initializes self.env and data_loader

        # --- Create Baseline Piles Dictionary ---
        baseline_piles = {} # Initialize as empty dict
        expert_layout_path = "output/best_expert_layout.txt" # Adjust path if needed
        try:
            # This call should now work
            baseline_layout_data = visualizer.load_expert_layout(expert_layout_path)
            logger.info(f"Loaded expert layout data type: {type(baseline_layout_data)}")

            if isinstance(baseline_layout_data, dict) and 'layout' in baseline_layout_data:
                layout_array = baseline_layout_data['layout']
                logger.info(f"Layout array type: {type(layout_array)}, shape: {getattr(layout_array, 'shape', 'N/A')}")

                if isinstance(layout_array, (np.ndarray, list, tuple)):
                    station_ids_str = visualizer.env.data_loader.station_ids.astype(str)
                    if len(station_ids_str) == len(layout_array):
                        baseline_piles = dict(zip(station_ids_str, layout_array))
                        logger.info(f"成功创建基准布局字典: {len(baseline_piles)} 个站点")
                        logger.info(f"基准布局字典示例: {list(baseline_piles.items())[:5]}")
                    else:
                        logger.error(f"基准布局加载失败: 站点ID数量 ({len(station_ids_str)}) 与布局数组长度 ({len(layout_array)}) 不匹配。")
                else:
                     logger.error(f"基准布局加载失败: 'layout'键的值不是预期的数组类型 (是 {type(layout_array)})。")
            elif baseline_layout_data is None:
                 logger.error(f"基准布局加载失败: load_expert_layout 返回 None (可能文件未找到或格式错误)。")
            else:
                 logger.error(f"基准布局加载失败: load_expert_layout 未返回包含'layout'键的字典 (返回类型: {type(baseline_layout_data)})。")

        except FileNotFoundError: # This might be redundant if load_expert_layout handles it, but good practice
             logger.error(f"基准布局文件未找到: {expert_layout_path}")
        except Exception as e:
             logger.error(f"加载或处理基准布局时出错: {e}", exc_info=True) # Log traceback
             # Ensure baseline_piles is empty on error
             baseline_piles = {}


        # --- Create Optimized Piles Dictionary ---
        optimized_piles = {} # Initialize as empty dict
        # --- Use the CSV file from DQN training results ---
        optimized_layout_path = "output/station_summary_ep299.csv" # <--- Path to your results CSV

        # --- NEW: Load optimized layout from CSV using pandas ---
        try:
            logger.info(f"尝试从 CSV 文件加载优化布局: {optimized_layout_path}")
            # Read the CSV file
            optimized_df = pd.read_csv(optimized_layout_path)

            # Check if required columns exist
            required_cols = ['station_id', 'charging_piles']
            if not all(col in optimized_df.columns for col in required_cols):
                logger.error(f"优化布局 CSV 文件 '{optimized_layout_path}' 缺少必需的列: {required_cols}")
                optimized_piles = {} # Ensure empty
            else:
                # Ensure station_id is string type for consistency
                optimized_df['station_id'] = optimized_df['station_id'].astype(str)
                # Ensure charging_piles is integer type
                optimized_df['charging_piles'] = optimized_df['charging_piles'].astype(int)

                # Create the dictionary: {station_id: charging_piles}
                optimized_piles = pd.Series(optimized_df.charging_piles.values, index=optimized_df.station_id).to_dict()

                # Optional: Verify against the environment's station IDs if needed
                env_station_ids = set(visualizer.env.data_loader.station_ids.astype(str))
                loaded_station_ids = set(optimized_piles.keys())
                if env_station_ids != loaded_station_ids:
                    logger.warning(f"优化布局中的站点ID ({len(loaded_station_ids)}) 与环境中的站点ID ({len(env_station_ids)}) 不完全匹配。")
                    logger.warning(f"仅存在于优化布局中的站点: {loaded_station_ids - env_station_ids}")
                    logger.warning(f"仅存在于环境中的站点: {env_station_ids - loaded_station_ids}")
                    # Depending on requirements, you might filter optimized_piles here
                    # optimized_piles = {k: v for k, v in optimized_piles.items() if k in env_station_ids}
                    # logger.info(f"过滤后，优化布局字典包含 {len(optimized_piles)} 个与环境匹配的站点")


                logger.info(f"成功从 CSV 创建优化布局字典: {len(optimized_piles)} 个站点")
                logger.info(f"优化布局字典示例: {list(optimized_piles.items())[:5]}")

        except FileNotFoundError:
             logger.error(f"优化布局 CSV 文件未找到: {optimized_layout_path}")
             optimized_piles = {} # Ensure empty
        except pd.errors.EmptyDataError:
             logger.error(f"优化布局 CSV 文件为空: {optimized_layout_path}")
             optimized_piles = {} # Ensure empty
        except Exception as e:
             logger.error(f"加载或处理优化布局 CSV 时出错: {e}", exc_info=True) # Log traceback
             optimized_piles = {} # Ensure empty


        # --- Generate Heatmaps (only if at least one layout is valid and non-empty) ---
        # Check specifically if the dictionaries are non-empty after attempting to load
        if baseline_piles or optimized_piles: # Check if either dict has content
             logger.info("尝试生成热力图...")
             # Ensure you pass the correctly loaded (or empty) dictionaries
             visualizer.generate_heatmaps(baseline_piles=baseline_piles,
                                          optimized_piles=optimized_piles,
                                          specific_hour=12)
        else:
            logger.error("基准和优化布局均为空或加载失败，无法生成热力图。")

    except Exception as main_err:
        logger.error(f"可视化脚本执行期间发生顶层错误: {main_err}", exc_info=True)