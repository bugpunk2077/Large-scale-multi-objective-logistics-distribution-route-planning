import numpy as np
from scipy.spatial.distance import cdist


def calculate_hypervolume(pareto_front, ref_point):
    """计算超体积 (简化实现，假设最小化问题)。

    说明：ref_point 应该是一个与目标维度相同的数组/列表，且每一维大于 pareto 中的目标值。
    本函数对每个解计算 (ref - obj) 的体积近似并求和作为超体积近似值。
    """
    if not pareto_front:
        return 0.0

    objs = np.array([sol.objectives for sol in pareto_front], dtype=float)
    ref = np.array(ref_point, dtype=float)
    # 保护性：若维度不匹配尝试修正（若无法匹配则返回 0）
    if ref.ndim != 1 or ref.shape[0] != objs.shape[1]:
        # 如果 ref 是标量或单值，尝试扩展为与 objs 相同的维度
        try:
            if np.isscalar(ref_point):
                ref = np.full((objs.shape[1],), float(ref_point), dtype=float)
        except Exception:
            return 0.0
        if ref.shape[0] != objs.shape[1]:
            return 0.0

    # 如果参考点在任何维度上都不比 Pareto 的最大值更糟（即 ref <= max(obj)），
    # 则自动扩展参考点以确保正体积：使用 max(ref, max(obj)) * margin
    try:
        max_vals = objs.max(axis=0)
        # margin 保守取 1.05（5%）以确保 ref 在 Pareto 之外
        margin = 1.05
        ref = np.maximum(ref, max_vals) * margin
    except Exception:
        # 若计算失败，保守返回 0
        return 0.0

    # 避免负值（若某目标超过参考点，则把差值截为 0）
    diffs = np.maximum(ref - objs, 0.0)
    volumes = np.prod(diffs, axis=1)
    # 返回平均超体积（便于不同规模比较）
    return float(np.mean(volumes))


def calculate_igd(pareto_front, true_pareto):
    """计算反转世代距离（IGD）。

    true_pareto: iterable of objective vectors (list of lists or np.array)
    """
    if not pareto_front or not true_pareto:
        return float('inf')

    approx_set = np.array([sol.objectives for sol in pareto_front], dtype=float)
    true_set = np.array(true_pareto, dtype=float)

    if approx_set.size == 0 or true_set.size == 0:
        return float('inf')

    distances = cdist(true_set, approx_set)
    min_distances = np.min(distances, axis=1)
    return float(np.mean(min_distances))


def evaluate_solution_set(pareto_front, problem):
    """评估解集质量并返回可序列化的 summary 字典。

    双目标：距离（最小化）和车辆数（最小化）
    """
    summary = {
        'num_solutions': 0,
        'hypervolume': 0.0,
        'best_distance': None,
        'best_vehicles': None,
        'avg_distance': None,
        'avg_vehicles': None,
    }

    if not pareto_front:
        return summary

    objs = np.array([sol.objectives for sol in pareto_front], dtype=float)

    # 参考点：距离和车辆数（双目标）
    maxd = float(np.max(problem.distance_matrix)) if hasattr(problem, 'distance_matrix') else 1.0
    ref_point = np.array([maxd * 10.0, max(1.0, float(getattr(problem, 'num_vehicles', 1))) * 2.0])

    hv = calculate_hypervolume(pareto_front, ref_point)

    # 统计车辆数
    vehicles_list = []
    for sol in pareto_front:
        v = None
        try:
            # 双目标情况下，objectives[1] 是车辆数
            v = int(sol.objectives[1])
        except Exception:
            try:
                v = sum(1 for r in getattr(sol, 'routes', []) if len(r) > 2)
            except Exception:
                v = None
        vehicles_list.append(v if v is not None else float('inf'))

    summary['num_solutions'] = len(pareto_front)
    summary['hypervolume'] = float(hv)
    summary['best_distance'] = float(np.min(objs[:, 0]))
    summary['best_vehicles'] = int(np.min(vehicles_list)) if vehicles_list else None
    summary['avg_distance'] = float(np.mean(objs[:, 0]))
    summary['avg_vehicles'] = float(np.mean(vehicles_list)) if vehicles_list else None

    return summary