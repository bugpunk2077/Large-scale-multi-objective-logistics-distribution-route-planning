import numpy as np
from scipy.spatial.distance import cdist

def calculate_hypervolume(pareto_front, ref_point):
    """计算超体积指标"""
    if not pareto_front:
        return 0.0
    
    objectives = np.array([sol.objectives for sol in pareto_front])
    
    # 归一化目标值
    normalized_obj = objectives / ref_point
    
    # 简化的超体积计算
    volume = 0.0
    for i in range(len(normalized_obj)):
        contribution = np.prod(1.0 - normalized_obj[i])
        volume += contribution
        
    return volume / len(pareto_front)

def calculate_igd(pareto_front, true_pareto):
    """计算反转世代距离"""
    if not pareto_front or not true_pareto:
        return float('inf')
    
    approx_set = np.array([sol.objectives for sol in pareto_front])
    true_set = np.array(true_pareto)
    
    distances = cdist(true_set, approx_set)
    min_distances = np.min(distances, axis=1)
    
    return np.mean(min_distances)

def evaluate_solution_set(pareto_front, problem):
    """评估解集质量"""
    if not pareto_front:
        return {
            'num_solutions': 0,
            'hypervolume': 0,
            'best_distance': float('inf'),
            'best_time': float('inf'),
            'best_vehicles': float('inf')
        }
    
    objectives = np.array([sol.objectives for sol in pareto_front])
    
    # 参考点（可以根据问题调整）
    ref_point = [problem.distance_matrix.max() * 10, 
                 problem.distance_matrix.max() * 5, 
                 problem.num_vehicles * 2]
    
    return {
        'num_solutions': len(pareto_front),
        'hypervolume': calculate_hypervolume(pareto_front, ref_point),
        'best_distance': objectives[:, 0].min(),
        'best_time': objectives[:, 1].min(),
        'best_vehicles': objectives[:, 2].min(),
        'avg_distance': objectives[:, 0].mean(),
        'avg_time': objectives[:, 1].mean(),
        'avg_vehicles': objectives[:, 2].mean()
    }