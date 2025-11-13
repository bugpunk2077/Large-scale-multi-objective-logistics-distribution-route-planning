import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from tqdm import tqdm
import sys
import glob

# 强制使用 UTF-8 输出，避免 Windows PowerShell 中文乱码
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    # 旧 Python 版本或环境可能不支持 reconfigure，忽略
    pass

# matplotlib 中文显示设置（如果系统中存在对应字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建必要的目录
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)

from models.gmoea import GMOEA
from utils.data_loader import load_vrp_instance, VRPProblem, load_ground_truth
from utils.evaluator import evaluate_solution_set
from utils.visualization import (
    plot_training_history,
    plot_pareto_front,
    plot_solution_routes,
    plot_combined_pareto_fronts,
    plot_combined_training_histories,
)

def main():
    print("基于神经网络的GMOEA物流配送路径规划 - 双目标版本")
    print("目标函数：1. 最小化总行驶距离  2. 最小化使用的车辆数")
    print("=" * 70)

    # 参数调优方案：根据实例规模与收敛情况选择合适的配置档位
    # 当前问题分析：700代稳定在1757.67，2000代仅微弱改进到1676.23（约5%）-> 早熟收敛、多样性丧失
    # 对策：增加人口多样性、提高探索强度、强化局部搜索、避免早熟
    
    # 选择配置档位（可选择其中一个或自定义）
    config_preset = 'conservative'  # 可选: 'conservative' / 'balanced' / 'aggressive'
    
    # 定义三个预设配置档位
    config_presets = {
        'conservative': {
            # 适用于快速验证、小规模实验
            # 目标：快速找到可行解，时间成本低
            'population_size': 50,
            'max_generations': 30,
            'crossover_rate': 0.7,      # 较低交叉率以保持多样性
            'mutation_rate': 0.3,       # 较高变异率增加探索
            'local_search_freq': 1,     # 每代执行一次局部搜索（轻量化）
            'local_search_depth': 'light',  # 'light' (2-opt仅)  / 'medium' / 'heavy'
        },
        'balanced': {
            # 适用于中等规模VRPTW实例（100个客户左右）
            # 目标：平衡探索与开发，在合理时间内得到近优解
            'population_size': 150,
            'max_generations': 1000,
            'crossover_rate': 0.75,     # 中等交叉率
            'mutation_rate': 0.35,      # 增强变异以抵抗早熟
            'local_search_freq': 2,     # 每2代执行一次局部搜索
            'local_search_depth': 'medium',  # 2-opt + 路径合并
        },
        'aggressive': {
            # 适用于寻求最优解、计算资源充足的情况
            # 目标：深度优化，尽可能接近全局最优
            'population_size': 300,
            'max_generations': 3000,
            'crossover_rate': 0.8,      # 中等偏高交叉率
            'mutation_rate': 0.4,       # 最高变异率，持续引入新血
            'local_search_freq': 1,     # 每代执行局部搜索（高强度）
            'local_search_depth': 'heavy',  # 完整2-opt + 激进路径合并 + 邻域重构
        },
    }
    
    # 使用预设或自定义
    config = {
        **config_presets[config_preset],
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'batch_size': 16,
        'train_interval': 5,
        # 行为克隆（BC）预训练选项：当为 True 且存在同名 .sol 文件时，先进行监督预训练
        'use_behavior_cloning': False,
        'bc_epochs': 10,
        'bc_batch_size': 8,
        'min_vehicles': 5,
        'max_vehicles': 30,
        'balance_strategy': 'min_vehicles',
        'use_nn': True,
        'verbose': True,
        'print_interval': 50,  # 每50代打印一次（减少输出）
        'use_kmeans':True,
        'kmeans_k':10,
        'kmeans_seed':0,
        'debug_bc':False,
    }
    
    print(f"\n使用配置档位: {config_preset}")
    print(f"参数调优说明:")
    print(f"  - population_size: {config['population_size']} (更大的人口增加多样性，探索更广的解空间)")
    print(f"  - max_generations: {config['max_generations']} (更多代数允许更深度的搜索)")
    print(f"  - mutation_rate: {config['mutation_rate']} (更高变异率延缓收敛速度，避免早熟)")
    print(f"  - local_search_freq: {config.get('local_search_freq', 2)} (更频繁的局部搜索加速收敛)\n")

    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 查找实例文件（支持 data/*.txt 和 data/*.vrp），若没有则使用内置示例
    default_file = 'data/lc101.txt'
    instance_files = sorted(glob.glob('data/*.txt') + glob.glob('data/*.vrp'))
    if os.path.exists(default_file) and len(instance_files) <= 1:
        instance_files = [default_file]

    if not instance_files:
        print("未在 data/ 目录下找到实例文件，后续将使用内置示例数据进行一次训练")

    overall_start = time.time()
    overall_results = {}
    overall_pareto = {}

    for fp in instance_files or [None]:
        if fp is None:
            print("\n=== 使用内置示例数据训练 ===")
            problem = VRPProblem()
            base = 'example'
        else:
            print(f"\n=== 训练实例: {fp} ===")
            try:
                problem = load_vrp_instance(fp)
            except Exception as e:
                print(f"加载实例 {fp} 失败: {e}\n使用示例数据代替")
                problem = VRPProblem()
            base = os.path.splitext(os.path.basename(fp))[0]

        print("\n初始化GMOEA算法...")
        # 如果配置中要求行为克隆预训练，则先启用 NN
        if config.get('use_behavior_cloning', False):
            config['use_nn'] = True
        algorithm = GMOEA(problem, config)

        # 若启用行为克隆并且存在同名 .sol 文件，则载入并执行预训练
        if config.get('use_behavior_cloning', False):
            sol_path = os.path.join('data', f"{base}.sol")
            if os.path.exists(sol_path):
                print(f"检测到监督解文件: {sol_path}，开始行为克隆预训练 ({config.get('bc_epochs')} epochs)")
                sols = load_ground_truth(sol_path)
                routes = sols.get(base)
                if routes:
                    # 数据增强：把启发式解也加入行为克隆样本以增加多样性和局部策略
                    mixed_dataset = []
                    # ground-truth 解（可能为单个解）
                    mixed_dataset.append(routes)
                    try:
                        # 最近邻启发式
                        nn_sol = algorithm.nearest_neighbor_solution()
                        if nn_sol:
                            mixed_dataset.append(nn_sol)
                    except Exception:
                        pass
                    try:
                        # 节约法启发式
                        s_sol = algorithm.savings_algorithm_solution()
                        if s_sol:
                            mixed_dataset.append(s_sol)
                    except Exception:
                        pass
                    try:
                        # 随机解一个作为多样性样本
                        r_sol = algorithm.generate_random_solution()
                        if r_sol:
                            mixed_dataset.append(r_sol)
                    except Exception:
                        pass

                    try:
                        algorithm.train_supervised(mixed_dataset, epochs=config.get('bc_epochs', 10), batch_size=config.get('bc_batch_size', 8))
                    except Exception as e:
                        print(f"行为克隆训练失败: {e}")
                else:
                    print(f"未能从 {sol_path} 解析到路线，跳过 BC 预训练。")
            else:
                print(f"配置要求行为克隆但未找到 {sol_path}，跳过 BC 预训练。")

        print("开始训练...")
        start_time = time.time()
        pareto_front, population, training_history = algorithm.run_training()
        end_time = time.time()

        duration = end_time - start_time
        results = {
            'config': config,
            'training_time': duration,
            'evaluation_results': evaluate_solution_set(pareto_front, problem),
            'training_history': training_history,
        }

        with open(f'results/{base}_training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 收集到 overall_results 中，用于合并绘图
        overall_results[base] = results
        overall_pareto[base] = pareto_front

        # 可视化（仅保存，不弹窗）
        plot_training_history(training_history, f'results/{base}_training_history.png', show=False)
        plot_pareto_front(pareto_front, f'results/{base}_pareto_front.png', show=False)
        if pareto_front:
            # 计算每个目标的范围用于归一化
            objectives = np.array([sol.objectives for sol in pareto_front])
            obj_ranges = objectives.max(axis=0) - objectives.min(axis=0)
            obj_ranges = np.where(obj_ranges > 0, obj_ranges, 1.0)  # 避免除零
            
            # 选择一个相对平衡的解（归一化后的三个目标之和最小）
            balanced_sol = min(pareto_front, 
                key=lambda x: sum((np.array(x.objectives) - objectives.min(axis=0)) / obj_ranges))
            
            # 保存三种不同的路径图
            plot_solution_routes(balanced_sol, problem, 
                f'results/{base}_balanced_routes.png', show=False,
                title_prefix="平衡的")
            
            best_distance_sol = min(pareto_front, key=lambda x: x.objectives[0])
            plot_solution_routes(best_distance_sol, problem, 
                f'results/{base}_shortest_distance_routes.png', show=False,
                title_prefix="最短距离的")

        # 记录最后一个实例的一些信息（供单实例路径图显示）
        last_base = base
        last_problem = problem
        last_best_distance = best_distance_sol if pareto_front else None

        print(f"实例 {base} 训练完成, 耗时 {duration:.1f}s, pareto 大小: {len(pareto_front)}")

    overall_end = time.time()
    print(f"\n总训练流程完成! 总耗时: {overall_end - overall_start:.2f}秒 ({(overall_end - overall_start)/60:.2f}分钟)")

    # 全部训练完成后，生成并展示两个合并窗口：
    # 1) 合并训练历史（每个实例一条曲线）
    # 2) 合并 Pareto 前沿（在子图中并列展示三维与二维投影）
    try:
        if overall_results:
            print("\n生成合并训练历史与合并Pareto前沿图 (仅保存，不直接弹窗)...")
            # 先只保存合并图到文件，严格禁止在这里弹出多个窗口
            combined_hist_path = 'results/combined_training_histories.png'
            combined_pareto_path = 'results/combined_pareto_fronts.png'
            plot_combined_training_histories(overall_results, save_path=combined_hist_path, show=False)
            plot_combined_pareto_fronts(overall_pareto, save_path=combined_pareto_path, show=False)

            # 统一在此处只打开两个窗口来显示已保存的合并图像（先关闭所有现有figure以防止残留窗口）
            try:
                import matplotlib.image as mpimg
                plt.close('all')

                img1 = mpimg.imread(combined_hist_path)
                img2 = mpimg.imread(combined_pareto_path)

                fig1 = plt.figure(figsize=(12, 8))
                plt.imshow(img1)
                plt.axis('off')
                fig1.suptitle('合并训练历史')

                fig2 = plt.figure(figsize=(14, 8))
                plt.imshow(img2)
                plt.axis('off')
                fig2.suptitle('合并 Pareto 前沿')

                # 仅一次 plt.show() 调用，会同时弹出上面两个窗口（严格为两个）
                plt.show()
            except Exception as e:
                print(f"显示合并图像失败: {e}")

            # 注意：只保存最后一个实例的最佳路径图，不弹窗，避免在多实例训练时生成过多窗口
            if last_best_distance is not None:
                plot_solution_routes(last_best_distance, last_problem, f'results/{last_base}_best_distance_routes.png', show=False)
    except Exception as e:
        print(f"展示合并可视化时出错: {e}")

if __name__ == "__main__":
    main()