import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_training_history(history, save_path=None, show=False):
    """绘制训练历史"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制损失曲线
    if history['policy_loss']:
        ax1.plot(history['policy_loss'], label='策略损失')
        ax1.plot(history['value_loss'], label='价值损失')
        ax1.set_xlabel('训练周期')
        ax1.set_ylabel('损失')
        ax1.set_title('训练损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 绘制目标函数进化
    if history['avg_objectives']:
        objectives = np.array(history['avg_objectives'])
        ax2.plot(objectives[:, 0], label='总距离')
        ax2.plot(objectives[:, 1], label='最长路径时间')
        ax2.plot(objectives[:, 2], label='车辆数')
        ax2.set_xlabel('代数')
        ax2.set_ylabel('目标值')
        ax2.set_title('平均目标函数进化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 绘制超体积进化
    if history['hv_history']:
        ax3.plot(history['hv_history'])
        ax3.set_xlabel('代数')
        ax3.set_ylabel('超体积')
        ax3.set_title('超体积指标进化')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")

    if show:
        plt.show()

def plot_pareto_front(pareto_front, save_path=None, show=False):
    """绘制Pareto前沿"""
    if not pareto_front:
        print("没有Pareto解可可视化")
        return
        
    objectives = np.array([sol.objectives for sol in pareto_front])
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D散点图
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                         c=objectives[:, 0], cmap='viridis', alpha=0.7)
    ax1.set_xlabel('总距离')
    ax1.set_ylabel('最长路径时间')
    ax1.set_zlabel('车辆数')
    ax1.set_title('三维Pareto前沿')
    
    # 2D投影 - 距离 vs 时间
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(objectives[:, 0], objectives[:, 1], 
                          c=objectives[:, 2], cmap='viridis', alpha=0.7)
    ax2.set_xlabel('总距离')
    ax2.set_ylabel('最长路径时间')
    ax2.set_title('总距离 vs 最长路径时间')
    plt.colorbar(scatter2, ax=ax2, label='车辆数')
    
    # 2D投影 - 距离 vs 车辆
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(objectives[:, 0], objectives[:, 2], 
                          c=objectives[:, 1], cmap='plasma', alpha=0.7)
    ax3.set_xlabel('总距离')
    ax3.set_ylabel('车辆数')
    ax3.set_title('总距离 vs 车辆数')
    plt.colorbar(scatter3, ax=ax3, label='最长路径时间')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto前沿图已保存至: {save_path}")

    if show:
        plt.show()


def plot_combined_pareto_fronts(pareto_dict, save_path=None, show=False):
    """绘制多个实例或多次运行得到的 Pareto 前沿（在同一张图中用颜色区分）。
    pareto_dict: {name: list_of_solutions}
    """
    # 收集所有点
    all_objs = []
    labels = []
    for name, sols in pareto_dict.items():
        if not sols:
            continue
        objs = np.array([s.objectives for s in sols])
        all_objs.append((name, objs))

    if not all_objs:
        print("没有Pareto解可可视化")
        return

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    cmap = plt.get_cmap('tab10')
    for i, (name, objs) in enumerate(all_objs):
        color = cmap(i % 10)
        ax1.scatter(objs[:, 0], objs[:, 1], objs[:, 2], c=[color], label=name, alpha=0.7)
        ax2.scatter(objs[:, 0], objs[:, 1], c=[color], label=name, alpha=0.7)
        ax3.scatter(objs[:, 0], objs[:, 2], c=[color], label=name, alpha=0.7)

    ax1.set_xlabel('总距离')
    ax1.set_ylabel('最长路径时间')
    ax1.set_zlabel('车辆数')
    ax1.set_title('合并三维Pareto前沿')

    ax2.set_xlabel('总距离')
    ax2.set_ylabel('最长路径时间')
    ax2.set_title('总距离 vs 最长路径时间')

    ax3.set_xlabel('总距离')
    ax3.set_ylabel('车辆数')
    ax3.set_title('总距离 vs 车辆数')

    ax2.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"合并Pareto前沿图已保存至: {save_path}")

    if show:
        plt.show()


def plot_combined_training_histories(results_dict, save_path=None, show=False):
    """绘制多个实例训练历史的合并图（每个实例一条曲线）。
    results_dict: {name: result_dict} where result_dict contains 'training_history'
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    cmap = plt.get_cmap('tab10')
    for i, (name, res) in enumerate(results_dict.items()):
        hist = res.get('training_history', {})
        color = cmap(i % 10)

        policy = hist.get('policy_loss', [])
        value = hist.get('value_loss', [])
        if policy:
            axes[0].plot(policy, label=f'{name}-policy', color=color)
        if value:
            axes[0].plot(value, label=f'{name}-value', color=color, linestyle='--')

        avg_obj = hist.get('avg_objectives', [])
        if avg_obj:
            arr = np.array(avg_obj)
            axes[1].plot(arr[:, 0], label=f'{name}-distance', color=color)

    axes[0].set_title('策略/价值 损失（多实例覆盖）')
    axes[0].set_xlabel('训练周期')
    axes[0].set_ylabel('损失')
    axes[0].legend(fontsize='small')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('平均总距离（多实例覆盖）')
    axes[1].set_xlabel('代数')
    axes[1].set_ylabel('平均总距离')
    axes[1].legend(fontsize='small')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"合并训练历史图已保存至: {save_path}")

    if show:
        plt.show()

def plot_solution_routes(solution, problem, save_path=None, show=False, title_prefix=""):
    """绘制解决方案路径"""
    if not solution:
        print("没有解决方案可可视化")
        return
        
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制客户点（带需求标注）
    customers_x = []
    customers_y = []
    demands = []
    n_nodes = len(problem.coordinates)
    for i in range(1, n_nodes):
        x, y = problem.coordinates[i]
        customers_x.append(x)
        customers_y.append(y)
        # 保护性取值：若 demands 数组长度不足或值异常，使用 0 填充
        try:
            d = float(problem.demands[i])
        except Exception:
            d = 0.0
        # 负需求不合理——保留原值用于诊断，但绘图时以 0 为下限
        demands.append(d)
    # 为可视化将负值裁剪到 0，以免颜色条误导
    demands_for_color = [max(0.0, float(d)) for d in demands]
    scatter = ax.scatter(customers_x, customers_y, c=demands_for_color, cmap='YlOrRd', 
                      s=100, alpha=0.7, label='客户点')
    plt.colorbar(scatter, ax=ax, label='需求量 (绘图时负值显示为0)')
    
    # 绘制仓库
    depot_x, depot_y = problem.coordinates[0]
    ax.scatter([depot_x], [depot_y], c='red', s=200, marker='s', label='仓库')
    
    # 绘制路径
    colors = plt.cm.tab10(np.linspace(0, 1, len(solution.routes)))
    
    for i, route in enumerate(solution.routes):
        if len(route) <= 2:  # 只有仓库
            continue
            
        route_x = [problem.coordinates[node][0] for node in route]
        route_y = [problem.coordinates[node][1] for node in route]
        
        ax.plot(route_x, route_y, 'o-', color=colors[i], linewidth=2, 
                markersize=6, label=f'车辆 {i+1}')
    
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    
    # 计算每条路径的负载（保护性计算，忽略越界或非法索引）
    route_loads = []
    invalid_indices = []
    for route in solution.routes:
        load = 0.0
        for node in route:
            if node == 0:
                continue
            try:
                if 0 <= int(node) < len(problem.demands):
                    load += float(problem.demands[int(node)])
                else:
                    invalid_indices.append(node)
            except Exception:
                invalid_indices.append(node)
        route_loads.append(load)
    
    # 计算负载统计
    total_load = sum(route_loads)
    avg_load = total_load / len([r for r in solution.routes if len(r) > 2])
    load_std = np.std([load for load in route_loads if load > 0])
    
    title = (f'{title_prefix}配送路径方案\n'
             f'总距离: {solution.objectives[0]:.1f}, '
             f'最长路径: {solution.objectives[1]:.1f}, '
             f'车辆数: {solution.objectives[2]}\n'
             f'平均负载: {avg_load:.1f}, 负载标准差: {load_std:.1f}')
    ax.set_title(title)
    
    # 为每条路径添加负载信息到图例，并在图上标注节点序号以便调试
    handles, labels = ax.get_legend_handles_labels()
    color_map = plt.cm.tab10
    for i, (route, load) in enumerate(zip(solution.routes, route_loads)):
        if len(route) > 2:  # 只显示非空路径
            color = color_map(i % 10)
            handles.append(plt.Line2D([0], [0], color=color, 
                          label=f'车辆 {i+1} (负载: {load:.1f})'))
            # 在每个客户点附近写上其索引（小字体，便于检查分配）
            for node in route:
                if node == 0:
                    continue
                try:
                    idx = int(node)
                    x, y = problem.coordinates[idx]
                    ax.text(x, y, str(idx), fontsize=6, color=color)
                except Exception:
                    # 忽略越界或非整数索引
                    pass
    ax.legend(handles=handles, title="红色越深表示需求越大")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"路径图已保存至: {save_path}")
        # 同路径图一起保存调试用 JSON，包含 routes、route_loads、demands 摘要
        try:
            import json
            json_path = os.path.splitext(save_path)[0] + '_debug.json'
            debug_data = {
                'routes': solution.routes,
                'route_loads': route_loads,
                'demands_snapshot': [float(d) for d in demands],
                'invalid_indices': invalid_indices
            }
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(debug_data, jf, ensure_ascii=False, indent=2)
            print(f"调试数据已保存至: {json_path}")
        except Exception as e:
            print(f"保存调试 JSON 失败: {e}")

    if show:
        plt.show()