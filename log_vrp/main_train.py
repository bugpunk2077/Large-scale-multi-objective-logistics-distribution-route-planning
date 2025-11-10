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
from utils.data_loader import load_vrp_instance, VRPProblem
from utils.evaluator import evaluate_solution_set
from utils.visualization import (
    plot_training_history,
    plot_pareto_front,
    plot_solution_routes,
    plot_combined_pareto_fronts,
    plot_combined_training_histories,
)

def main():
    print("基于神经网络的GMOEA物流配送路径规划 - 完整训练版本")
    print("=" * 70)

    # 配置参数
    config = {
        'population_size': 50,      # 种群大小
        'max_generations': 1000,      # 最大代数
        'hidden_dim': 64,           # 神经网络隐藏层维度
        'learning_rate': 0.001,     # 学习率
        'crossover_rate': 0.8,      # 交叉概率
        'mutation_rate': 0.2,       # 变异概率
        'batch_size': 8,            # 训练批量大小
        'train_interval': 20,       # 训练间隔（代）
        'min_vehicles': 5,
        'max_vehicles': 30,         # 车辆数量上下限
        'verbose': True,
        'print_interval': 10,
    }

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
        algorithm = GMOEA(problem, config)

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