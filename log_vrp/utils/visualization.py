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
    if history.get('avg_objectives'):
        try:
            objectives = np.array(history['avg_objectives'])
            if objectives.size and objectives.ndim == 2 and objectives.shape[1] >= 3:
                ax2.plot(objectives[:, 0], label='总距离')
                ax2.plot(objectives[:, 1], label='最长路径时间')
                ax2.plot(objectives[:, 2], label='车辆数')
        except Exception:
            # 避免因空数组/NaN导致的 numpy 警告
            pass
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

    fig = plt.figure(figsize=(10, 5))

    # 2D投影 - 距离 vs 车辆
    ax = fig.add_subplot(111)
    scatter = ax.scatter(objectives[:, 0], objectives[:, 1], cmap='viridis', alpha=0.7)
    ax.set_xlabel('总距离')
    ax.set_ylabel('车辆数')
    ax.set_title('总距离 vs 车辆数')
    plt.colorbar(scatter, ax=ax, label='Pareto解')

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

    # 目前为双目标（总距离, 车辆数）: 绘制单张 2D 图即可，用颜色区分不同实例
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    cmap = plt.get_cmap('tab10')
    for i, (name, objs) in enumerate(all_objs):
        color = cmap(i % 10)
        # 保护性：确保 objs 至少有 2 列
        if objs.ndim == 1:
            xs = objs[0:1]
            ys = objs[1:2] if objs.shape[0] > 1 else np.zeros_like(xs)
        else:
            xs = objs[:, 0]
            ys = objs[:, 1] if objs.shape[1] > 1 else np.zeros(len(xs))
        ax.scatter(xs, ys, c=[color], label=name, alpha=0.7)

    ax.set_xlabel('总距离')
    ax.set_ylabel('车辆数')
    ax.set_title('合并 Pareto 前沿（总距离 vs 车辆数）')
    ax.legend()
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
    """绘制解决方案路径，支持 pickup (+) / delivery (-) 需求"""
    if not solution:
        print("没有解决方案可可视化")
        return
        
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 分别绘制 pickup (+) 和 delivery (-) 点
    pickups_x = []
    pickups_y = []
    pickup_demands = []
    deliveries_x = []
    deliveries_y = []
    delivery_demands = []
    
    n_nodes = len(problem.coordinates)
    for i in range(1, n_nodes):
        x, y = problem.coordinates[i]
        try:
            d = float(problem.demands[i])
            if d > 0:  # Pickup 点
                pickups_x.append(x)
                pickups_y.append(y)
                pickup_demands.append(d)
            else:  # Delivery 点
                deliveries_x.append(x)
                deliveries_y.append(y)
                delivery_demands.append(abs(d))  # 取绝对值用于颜色映射
        except Exception:
            continue  # 跳过无效点
    # 分别绘制 pickup 和 delivery 点（用不同形状和颜色）
    if pickups_x:
        scatter_pickup = ax.scatter(pickups_x, pickups_y, 
                                  c=pickup_demands, cmap='Reds', 
                                  marker='^', s=100, alpha=0.7, 
                                  label='Pickup点')
        plt.colorbar(scatter_pickup, ax=ax, label='Pickup需求量')
    
    if deliveries_x:
        scatter_delivery = ax.scatter(deliveries_x, deliveries_y,
                                    c=delivery_demands, cmap='Blues',
                                    marker='v', s=100, alpha=0.7,
                                    label='Delivery点')
        plt.colorbar(scatter_delivery, ax=ax, label='Delivery需求量')
    
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
                ni = int(node)
                if 0 <= ni < len(problem.demands):
                    load += float(problem.demands[ni])
                else:
                    invalid_indices.append(node)
            except Exception:
                invalid_indices.append(node)
        route_loads.append(load)

    # 计算负载统计（带保护）
    non_empty_routes = [r for r in solution.routes if len(r) > 2]
    total_load = sum(route_loads) if route_loads else 0.0
    valid_loads = [load for load in route_loads if load is not None]
    if non_empty_routes:
        try:
            avg_load = total_load / len(non_empty_routes)
        except Exception:
            avg_load = 0.0
    else:
        avg_load = 0.0

    if valid_loads:
        try:
            load_std = float(np.nanstd(valid_loads))
        except Exception:
            load_std = 0.0
    else:
        load_std = 0.0
    
    # 保护性访问 objectives（双目标：distance, vehicles）
    try:
        dist_val = float(solution.objectives[0]) if len(solution.objectives) > 0 else 0.0
    except Exception:
        dist_val = 0.0
    try:
        veh_cnt = int(solution.objectives[1]) if len(solution.objectives) > 1 else len(solution.routes)
    except Exception:
        veh_cnt = len(solution.routes)

    title = (f'{title_prefix}配送路径方案\n'
             f'总距离: {dist_val:.1f}, '
             f'车辆数: {veh_cnt}\n'
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
                'pickups': [(x, y, d) for x, y, d in zip(pickups_x, pickups_y, pickup_demands)] if pickups_x else [],
                'deliveries': [(x, y, d) for x, y, d in zip(deliveries_x, deliveries_y, delivery_demands)] if deliveries_x else [],
                'invalid_indices': invalid_indices
            }
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(debug_data, jf, ensure_ascii=False, indent=2)
            print(f"调试数据已保存至: {json_path}")
        except Exception as e:
            print(f"保存调试 JSON 失败: {e}")

    if show:
        plt.show()