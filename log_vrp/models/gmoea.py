"""
简洁且自洽的 GMOEA 实现（多目标进化）

说明：
- 与 `main_train.py` / `utils.evaluator.py` 接口兼容。
- NSGA-II 风格的非支配排序 + 拥挤距离选择。
- 初始化使用：节约法(savings)、最近邻(nearest neighbor)、随机构造。
- 进化算子：路径混合交叉、随机交换变异、修复以保证覆盖所有客户。
- 保留神经网络接入占位，但当前不主动调用外部 NN（避免构造不匹配导致运行错误）。
"""

import random
import time
import numpy as np
from utils.evaluator import calculate_hypervolume
import torch
import torch.nn as nn
from utils.data_loader import compute_kmeans
from models.neural_network import NeuralVRPSolver


class Solution:
    """简单容器：routes + objectives

    routes: List[List[int]]，每条路径以 0（仓库）开始和结束
    objectives: [total_distance, num_vehicles]（双目标：最小化距离与车辆数）
    """

    def __init__(self, routes, objectives=None):
        self.routes = routes
        self.objectives = objectives

    def evaluate(self, problem):
        total_distance = 0.0
        for route in self.routes:
            if len(route) <= 2:
                continue
            route_dist = 0.0
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                route_dist += problem.distance_matrix[a][b]
            total_distance += route_dist
        num_vehicles = sum(1 for r in self.routes if len(r) > 2)
        self.objectives = [total_distance, num_vehicles]
        return self.objectives


class GMOEA:
    """简洁、可运行的多目标进化框架（非支配排序 + 拥挤距离）"""

    def __init__(self, problem, config=None):
        self.problem = problem
        self.config = config or {}
        self.population_size = int(self.config.get('population_size', 50))
        self.max_generations = int(self.config.get('max_generations', 100))
        self.crossover_rate = float(self.config.get('crossover_rate', 0.8))
        self.mutation_rate = float(self.config.get('mutation_rate', 0.2))
        self.train_interval = int(self.config.get('train_interval', 20))
        # 局部搜索配置（防止早熟收敛）
        self.local_search_freq = int(self.config.get('local_search_freq', 2))  # 每N代执行一次
        self.local_search_depth = self.config.get('local_search_depth', 'heavy')  # 'light' / 'medium' / 'heavy'
        # 输出与打印控制
        self.verbose = bool(self.config.get('verbose', True))
        self.print_interval = int(self.config.get('print_interval', 10))

        # 训练历史（interface 兼容）
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'avg_objectives': [],
            'hv_history': []
        }

        # --- 神经网络（可选） ---
        self.use_nn = bool(self.config.get('use_nn', True))
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.nn_model = None
        self.nn_optimizer = None
        if self.use_nn:
            try:
                # 基础节点特征维度（默认 x,y,demand,earliest,latest）
                base_node_dim = int(self.config.get('node_dim', 5))
                # 如果启用了 k-means 聚类特征，自动为每节点添加两个特征：
                # 距离到簇中心、簇规模归一化（因此 node_dim += 2）
                if bool(self.config.get('use_kmeans', False)):
                    node_dim = base_node_dim + 2
                else:
                    node_dim = base_node_dim

                hidden_dim = int(self.config.get('hidden_dim', 64))
                n_actions = len(self.problem.coordinates)
                # 双目标 (distance, vehicles)
                self.nn_model = NeuralVRPSolver(node_dim=node_dim, hidden_dim=hidden_dim, n_actions=n_actions, n_objectives=2)
                self.nn_model.to(self.device)
                lr = float(self.config.get('learning_rate', 1e-3))
                self.nn_optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=lr)
            except Exception as e:
                print(f"初始化神经网络失败，已禁用 NN 集成: {e}")
                self.use_nn = False

    def run_training(self):
        start = time.time()
        population = self.initialize_population()
        if not population:
            raise RuntimeError('初始化种群失败，请检查数据或生成策略')

        for gen in range(self.max_generations):
            # 评估
            for ind in population:
                if ind.objectives is None:
                    ind.evaluate(self.problem)

            # 记录平均目标
            avg = np.mean([ind.objectives for ind in population], axis=0)
            self.training_history['avg_objectives'].append(avg.tolist())

            # 计算 Pareto 与 Hypervolume
            fronts = self.fast_non_dominated_sort(population)
            pareto = fronts[0] if fronts else []
            if pareto:
                # 根据当前 Pareto 动态构建参考点，确保参考点严格位于所有解之上
                objs = np.array([s.objectives for s in pareto], dtype=float)
                # 以 Pareto 最大值为基准，乘以一个安全裕度
                max_vals = objs.max(axis=0)
                # 若某一维的 max_val 非正则退回到问题的 distance_matrix 或 num_vehicles
                try:
                    maxd = float(np.max(self.problem.distance_matrix)) if hasattr(self.problem, 'distance_matrix') else 1.0
                except Exception:
                    maxd = 1.0
                fallback = np.array([maxd * 10.0, max(1.0, float(getattr(self.problem, 'num_vehicles', 1)))])
                # 使用 max(max_vals, fallback) 并乘以 margin
                ref_arr = np.maximum(max_vals, fallback) * 1.1
                ref = ref_arr.tolist()
                hv = calculate_hypervolume(pareto, ref)
            else:
                hv = 0.0
            self.training_history['hv_history'].append(float(hv))

            # 实时打印（可配置） - 改为每代输出以满足需求
            if self.verbose:
                # population 的当前最小/均值目标
                try:
                    pop_objs = np.array([ind.objectives for ind in population], dtype=float)
                    bests = pop_objs.min(axis=0)
                    avgs = pop_objs.mean(axis=0)
                    pareto_size = len(pareto)
                    print(f"Gen {gen+1}/{self.max_generations} | Pareto {pareto_size} | HV {hv:.4g} | best(dist,veh) = ({bests[0]:.2f}, {int(bests[1])}) | avg = ({avgs[0]:.2f}, {avgs[1]:.2f})")
                except Exception:
                    print(f"Gen {gen+1}/{self.max_generations} | Pareto {len(pareto)} | HV {hv:.4g}")

            # NN 训练（若启用则按间隔训练并记录损失）
            if self.use_nn and (gen % self.train_interval == 0):
                ploss, vloss = self.train_nn(population)
                # 记录到 history（保持与早期接口兼容）
                self.training_history['policy_loss'].append(float(ploss))
                self.training_history['value_loss'].append(float(vloss))
            elif not self.use_nn and (gen % self.train_interval == 0):
                # 保持原有占位行为
                self.training_history['policy_loss'].append(0.0)
                self.training_history['value_loss'].append(0.0)

            # 产生子代并选择下一代
            # 按 local_search_freq 决定是否对子代应用局部搜索
            apply_local_search = (gen % self.local_search_freq == 0)
            offspring = self.evolve_population(population, apply_local_search=apply_local_search)
            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)
            new_pop = []
            for f in fronts:
                if len(new_pop) + len(f) <= self.population_size:
                    new_pop.extend(f)
                else:
                    crowd = self.calculate_crowding_distance(f)
                    order = np.argsort(-crowd)
                    need = self.population_size - len(new_pop)
                    for i in order[:need]:
                        new_pop.append(f[int(i)])
                    break
            population = new_pop

        elapsed = time.time() - start
        print(f"GMOEA 完成: {self.max_generations} 代, 耗时 {elapsed:.1f}s, 种群大小 {len(population)}")
        final_fronts = self.fast_non_dominated_sort(population)
        pareto_front = final_fronts[0] if final_fronts else []
        return pareto_front, population, self.training_history

    def initialize_population(self):
        population = []
        # 尝试节约法种子
        try:
            routes = self.savings_algorithm_solution()
            if routes:
                routes = self.balance_routes(routes)
                s = Solution(routes)
                s.evaluate(self.problem)
                population.append(s)
        except Exception:
            pass

        # 填充：最近邻、随机与（可选）神经网络混合
        while len(population) < self.population_size:
            if self.use_nn and random.random() < 0.3:
                try:
                    routes = self.generate_solution_with_nn()
                except Exception:
                    routes = None
            elif random.random() < 0.4:
                routes = self.nearest_neighbor_solution()
            else:
                routes = self.generate_random_solution()
            # 仅在需要时对路由做平衡/打包；balance_strategy 控制是均匀分配还是尽量压缩车辆数
            routes = self.balance_routes(routes)
            s = Solution(routes)
            s.evaluate(self.problem)
            population.append(s)
        return population

    def generate_solution_with_nn(self):
        """用当前 NN 模型生成一个解（贪心策略）。返回 routes 列表。"""
        if not self.use_nn or self.nn_model is None:
            raise RuntimeError('NN 未初始化')

        self.nn_model.eval()
        node_feats = self._build_node_features()
        n_nodes = node_feats.shape[0]
        node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        unvisited = set(range(1, self.problem.num_customers + 1))
        routes = []

        while unvisited:
            cur = [0]
            load = 0.0
            pos = 0
            while True:
                visited_mask = torch.zeros((1, n_nodes), dtype=torch.bool, device=self.device)
                for v in range(n_nodes):
                    if v not in unvisited and v != 0:
                        visited_mask[0, v] = True
                # also mask depot as visited so policy won't pick it mid-route
                visited_mask[0, 0] = True

                current_tensor = torch.tensor([pos], dtype=torch.long, device=self.device)
                try:
                    with torch.no_grad():
                        action_probs, action_logits = self.nn_model(node_feats_t, current_tensor, visited_mask)
                except Exception:
                    break

                probs = action_probs.squeeze(0).cpu().numpy()
                # choose highest-prob feasible candidate
                cand_indices = np.argsort(-probs)
                chosen = None
                for idx in cand_indices:
                    if idx == 0:
                        continue
                    if idx not in unvisited:
                        continue
                    d = float(self.problem.demands[idx])
                    if load + d <= self.problem.vehicle_capacity:
                        chosen = int(idx)
                        break

                if chosen is None:
                    # 无可行客户，结束本车路线
                    break

                cur.append(chosen)
                load += float(self.problem.demands[chosen])
                unvisited.remove(chosen)
                pos = chosen

            cur.append(0)
            if len(cur) > 2:
                routes.append(cur)
            else:
                # 无法选到新客户，避免死循环
                break

        # 若仍有未访问客户，按贪心补上
        for m in sorted(unvisited):
            routes.append([0, m, 0])

        return routes

    def generate_random_solution(self):
        customers = list(range(1, self.problem.num_customers + 1))
        random.shuffle(customers)
        routes = []
        cur = [0]
        load = 0
        for c in customers:
            d = self.problem.demands[c]
            if load + d > self.problem.vehicle_capacity and len(cur) > 1:
                cur.append(0)
                routes.append(cur)
                cur = [0]
                load = 0
            cur.append(c)
            load += d
        if len(cur) > 1:
            cur.append(0)
            routes.append(cur)
        return routes

    def nearest_neighbor_solution(self):
        unvisited = set(range(1, self.problem.num_customers + 1))
        routes = []
        while unvisited:
            cur = [0]
            pos = 0
            load = 0
            while True:
                cand = [c for c in unvisited if load + self.problem.demands[c] <= self.problem.vehicle_capacity]
                if not cand:
                    break
                nxt = min(cand, key=lambda c: self.problem.distance_matrix[pos][c])
                cur.append(nxt)
                load += self.problem.demands[nxt]
                unvisited.remove(nxt)
                pos = nxt
            cur.append(0)
            routes.append(cur)
        return routes

    def evolve_population(self, population, apply_local_search=True):
        offspring = []
        target = max(1, self.population_size // 2)
        while len(offspring) < target:
            p1 = self.tournament_select(population)
            p2 = self.tournament_select(population)
            if random.random() < self.crossover_rate:
                child_routes = self.simple_crossover(p1.routes, p2.routes)
            else:
                child_routes = [r.copy() for r in p1.routes]
            if random.random() < self.mutation_rate:
                child_routes = self.simple_mutation(child_routes)
            child_routes = self.repair_routes(child_routes, apply_local_search=apply_local_search)
            child = Solution(child_routes)
            child.evaluate(self.problem)
            offspring.append(child)
        return offspring

    def tournament_select(self, population, k=2):
        cand = random.sample(population, k)
        cand.sort(key=lambda x: sum(x.objectives))
        return cand[0]

    def simple_crossover(self, r1_set, r2_set):
        all_customers = set(range(1, self.problem.num_customers + 1))
        child = []
        assigned = set()
        # 保持父1的顺序片段
        for r in r1_set:
            seg = [n for n in r if n != 0 and n not in assigned]
            if seg:
                child.append([0] + seg + [0])
                assigned.update(seg)
        # 补充父2的片段
        for r in r2_set:
            seg = [n for n in r if n != 0 and n not in assigned]
            if seg:
                child.append([0] + seg + [0])
                assigned.update(seg)
        # 添加剩余客户
        remaining = list(all_customers - assigned)
        if remaining:
            cur = [0]
            load = 0
            for c in remaining:
                if load + self.problem.demands[c] > self.problem.vehicle_capacity and len(cur) > 1:
                    cur.append(0)
                    child.append(cur)
                    cur = [0]
                    load = 0
                cur.append(c)
                load += self.problem.demands[c]
            if len(cur) > 1:
                cur.append(0)
                child.append(cur)
        # 经过简单拼接后做平衡处理，确保不超过车辆上限
        return self.balance_routes(child)

    def simple_mutation(self, routes):
        nodes = [(i, j) for i, r in enumerate(routes) for j in range(1, len(r) - 1)]
        if len(nodes) < 2:
            return routes
        (r1, p1), (r2, p2) = random.sample(nodes, 2)
        routes[r1][p1], routes[r2][p2] = routes[r2][p2], routes[r1][p1]
        return routes

    def repair_routes(self, routes, apply_local_search=True):
        all_customers = set(range(1, self.problem.num_customers + 1))
        assigned = set()
        cleaned = []
        for r in routes:
            nr = [0]
            for n in r[1:-1]:
                if 1 <= n <= self.problem.num_customers and n not in assigned:
                    nr.append(n)
                    assigned.add(n)
            nr.append(0)
            if len(nr) > 2:
                cleaned.append(nr)
        for m in sorted(all_customers - assigned):
            cleaned.append([0, m, 0])
        
        # 局部搜索：根据配置深度和频率应用
        if apply_local_search:
            improved = self.local_search(cleaned, depth=self.local_search_depth)
        else:
            improved = cleaned
        
        # 最后做一次平衡/打包
        return self.balance_routes(improved)

    def two_opt(self, route):
        """对单条路径执行 2-opt 局部搜索以减少距离。route 为包含 0 起止的节点列表。"""
        if len(route) <= 4:
            return route
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    if j - i == 1:
                        continue
                    newr = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    # 计算两者距离
                    def route_dist(r):
                        d = 0.0
                        for a, b in zip(r[:-1], r[1:]):
                            d += self.problem.distance_matrix[a][b]
                        return d
                    if route_dist(newr) + 1e-6 < route_dist(best):
                        best = newr
                        improved = True
                        break
                if improved:
                    break
        return best

    def local_search(self, routes, depth='medium'):
        """对一组 routes 执行局部改进。depth 控制搜索强度：
        - 'light': 仅执行单轮 2-opt，不做合并
        - 'medium': 执行 2-opt + 基本合并（每两条路径尝试一次）
        - 'heavy': 执行完整 2-opt + 激进合并（多轮迭代）+ 邻域重构
        """
        # 先对每条路径做 2-opt（所有深度都执行）
        routes = [self.two_opt(r) for r in routes]
        
        if depth == 'light':
            # 仅做 2-opt，不做合并
            return routes
        
        # 'medium' 和 'heavy' 都尝试合并
        if depth == 'medium':
            # 基本合并：一轮迭代
            routes = self._merge_routes_once(routes)
        elif depth == 'heavy':
            # 激进合并：多轮迭代 + 重试
            for _ in range(3):  # 最多三轮迭代
                old_len = len(routes)
                routes = self._merge_routes_once(routes)
                if len(routes) == old_len:
                    break  # 无法再合并，退出
        
        return routes
    
    def _merge_routes_once(self, routes):
        """尝试一次合并两条路径。"""
        merged = True
        while merged:
            merged = False
            best_gain = 0.0
            best_pair = None
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    r1 = routes[i]
                    r2 = routes[j]
                    load1 = sum(self.problem.demands[n] for n in r1 if n != 0)
                    load2 = sum(self.problem.demands[n] for n in r2 if n != 0)
                    if load1 + load2 > self.problem.vehicle_capacity:
                        continue
                    # 尝试简单连接 r1[:-1] + r2[1:]
                    cand = r1[:-1] + r2[1:]
                    # 计算距离差
                    def dist_of(r):
                        s = 0.0
                        for a, b in zip(r[:-1], r[1:]):
                            s += self.problem.distance_matrix[a][b]
                        return s
                    old = dist_of(r1) + dist_of(r2)
                    new = dist_of(cand)
                    gain = old - new
                    if gain > best_gain + 1e-6:
                        best_gain = gain
                        best_pair = (i, j, cand)
            if best_pair is not None:
                i, j, cand = best_pair
                if i < j:
                    del routes[j]
                    del routes[i]
                else:
                    del routes[i]
                    del routes[j]
                routes.append(cand)
                merged = True
        return routes

    def balance_routes(self, routes):
        """打包/平衡路径。

        balance_strategy: 可选 'even' 或 'min_vehicles'。
        - 'even' (原有行为)：将客户轮询分配到 problem.num_vehicles 个车上，得到较均匀分布（但可能使用更多车辆）。
        - 'min_vehicles' (默认)：先按需求降序进行 First-Fit 填充，尽量使用更少车辆（遵守 vehicle_capacity），在达到上限时才创建更多车辆。
        返回值为带 0 起止的路径列表。
        """
        if not routes:
            return []

        strategy = self.config.get('balance_strategy', 'min_vehicles')
        customers = [c for r in routes for c in r if c != 0]
        max_v = int(getattr(self.problem, 'num_vehicles', max(1, len(routes))))

        if strategy == 'even':
            # 保持原有轮询分配逻辑
            if max_v <= 0:
                max_v = 1
            new_routes = [[0] for _ in range(max_v)]
            loads = [0.0 for _ in range(max_v)]
            idx = 0
            for c in customers:
                d = float(self.problem.demands[c])
                placed = False
                for attempt in range(max_v):
                    j = (idx + attempt) % max_v
                    if loads[j] + d <= self.problem.vehicle_capacity:
                        new_routes[j].append(c)
                        loads[j] += d
                        placed = True
                        idx = (j + 1) % max_v
                        break
                if not placed:
                    new_routes.append([0, c])
                    loads.append(d)
                    idx = len(new_routes) - 1
            final = []
            for r in new_routes:
                if len(r) > 1:
                    r.append(0)
                    final.append(r)
            return final

        # 默认：min_vehicles -> First-Fit Decreasing (FFD) based on demands
        # 将客户按 demand 降序排序以便先放大需求
        try:
            cust_sorted = sorted(customers, key=lambda c: -float(self.problem.demands[c]))
        except Exception:
            cust_sorted = customers[:]

        new_routes = []  # 每项为不含 depot 的客户列表
        loads = []
        for c in cust_sorted:
            d = float(self.problem.demands[c])
            placed = False
            # 尝试把客户放入已有车队（先找到第一个能放下的）
            for i, load in enumerate(loads):
                if load + d <= self.problem.vehicle_capacity:
                    new_routes[i].append(c)
                    loads[i] += d
                    placed = True
                    break
            if not placed:
                if len(new_routes) < max_v:
                    new_routes.append([c])
                    loads.append(d)
                else:
                    # 已达车辆上限，尽量放入负载最小的车（可能会超过容量）
                    min_i = int(np.argmin(loads)) if loads else 0
                    if loads:
                        new_routes[min_i].append(c)
                        loads[min_i] += d
                    else:
                        new_routes.append([c])
                        loads.append(d)

        # 格式化成带 depot 的路径
        final = []
        for r in new_routes:
            if r:
                final.append([0] + r + [0])
        return final

    # ----------------- 神经网络相关辅助方法 -----------------
    def _build_node_features(self):
        """构建节点特征矩阵：[n_nodes, node_dim]

        特征顺序: x, y, demand, earliest, latest
        返回 numpy 数组 shape (n_nodes, node_dim)
        """
        coords = getattr(self.problem, 'coordinates', [])
        demands = getattr(self.problem, 'demands', [])
        tw = getattr(self.problem, 'time_windows', [])
        n = len(coords)
        feats = np.zeros((n, 5), dtype=float)
        for i in range(n):
            x, y = coords[i]
            d = float(demands[i]) if i < len(demands) else 0.0
            if i < len(tw):
                earliest, latest = tw[i]
            else:
                earliest, latest = 0.0, 1e9
            feats[i, :] = [x, y, d, float(earliest), float(latest)]

        # 可选：基于 k-means 的局部簇特征（距离到簇中心、簇规模归一化）
        try:
            if bool(self.config.get('use_kmeans', False)):
                k = int(self.config.get('kmeans_k', 4))
                labels, centers = compute_kmeans(coords, k=k, random_state=int(self.config.get('kmeans_seed', 0)))
                labels = np.array(labels, dtype=int)
                centers = np.array(centers, dtype=float)
                # 计算每个点到其簇中心的距离
                dists_to_center = np.zeros((n,), dtype=float)
                for i in range(n):
                    lbl = labels[i] if i < len(labels) else 0
                    c = centers[lbl]
                    xi, yi = coords[i]
                    dists_to_center[i] = float(((xi - c[0]) ** 2 + (yi - c[1]) ** 2) ** 0.5)
                # 簇规模归一化（cluster size / max_size）
                cluster_sizes = np.array([np.sum(labels == j) for j in range(centers.shape[0])], dtype=float)
                max_size = float(cluster_sizes.max()) if cluster_sizes.size else 1.0
                size_norm = np.array([cluster_sizes[labels[i]] / max_size for i in range(n)], dtype=float)

                # 将两列特征拼接到 feats
                feats = np.hstack([feats, dists_to_center.reshape(-1, 1), size_norm.reshape(-1, 1)])
        except Exception:
            # 若 k-means 失败则忽略此特征
            pass

        return feats

    def _solution_to_route_tensor(self, solution, max_len):
        """把一个 Solution 的所有客户按访问顺序拼接成索引序列，返回长度为 max_len 的 LongTensor（填充值 -1）。"""
        # flatten all routes into a single customer sequence (exclude depot 0)
        seq = []
        for r in solution.routes:
            for node in r:
                if node != 0:
                    seq.append(int(node))
        seq = seq[:max_len]
        padded = seq + ([-1] * (max_len - len(seq)))
        return torch.LongTensor(padded)

    def train_nn(self, population):
        """使用当前种群训练神经网络（单次更新），返回 (policy_loss, value_loss)。"""
        if not self.use_nn or self.nn_model is None or self.nn_optimizer is None:
            return 0.0, 0.0

        batch_size = int(self.config.get('batch_size', 8))
        # 采样训练样本
        samples = random.sample(population, min(batch_size, len(population)))

        # 构建 node_features（一次性）
        node_feats = self._build_node_features()  # (n_nodes, node_dim)
        n_nodes, node_dim = node_feats.shape
        # 小心内存：只复制必要的 batch 维度
        node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 构建 solution route tensors
        max_route_len = max(1, max((sum(len(r) - 2 for r in s.routes) for s in samples)))
        route_tensors = torch.stack([self._solution_to_route_tensor(s, max_route_len) for s in samples], dim=0).to(self.device)

        # 目标值（objectives）
        targets = torch.tensor([s.objectives for s in samples], dtype=torch.float32, device=self.device)

        self.nn_model.train()
        self.nn_optimizer.zero_grad()

        # 构造规范化因子（参考 evaluator 的 ref_point），用以稳定 Value 网络训练幅度
        try:
            maxd = float(np.max(self.problem.distance_matrix)) if hasattr(self.problem, 'distance_matrix') else 1.0
        except Exception:
            maxd = 1.0
        # 双目标参考点：距离和车辆数
        ref = torch.tensor([maxd * 10.0, max(1.0, float(getattr(self.problem, 'num_vehicles', 1))) * 2.0], dtype=torch.float32, device=self.device)

        # 1) Value 网络预测与损失（使用规范化目标以避免数值爆炸）
        try:
            # 为 value 前向准备批次的 current_node 与 visited_mask
            current_nodes = torch.zeros((len(samples),), dtype=torch.long, device=self.device)
            visited_mask = torch.zeros((len(samples), n_nodes), dtype=torch.bool, device=self.device)
            # 使用 repeat 临时创建 batch 视图 —— 对小规模 batch 是可接受的
            _, _, value_pred = self.nn_model(node_feats_t.repeat(len(samples), 1, 1), current_node=current_nodes, visited_mask=visited_mask, solution_routes=route_tensors)
            value_loss_fn = nn.MSELoss()
            # 规范化预测与目标
            vloss = value_loss_fn(value_pred / ref, targets / ref)
        except Exception:
            vloss = torch.tensor(0.0, device=self.device)

        # 2) Policy 网络的简单行为克隆损失（避免存储大量小张量，使用累加）
        ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        policy_loss_acc = torch.tensor(0.0, device=self.device)
        policy_count = 0
        # 限制单次训练中 policy 前向的最大步数，防止初始代时大量短路展开造成尖峰
        max_policy_steps = int(self.config.get('policy_max_steps', 200))

        for i, s in enumerate(samples):
            # build flattened customer sequence
            seq = []
            for r in s.routes:
                for node in r:
                    if node != 0:
                        seq.append(int(node))
            if not seq:
                continue

            visited = set([0])
            for t_idx, nxt in enumerate(seq):
                current = 0 if t_idx == 0 else seq[t_idx - 1]
                current_tensor = torch.tensor([current], dtype=torch.long, device=self.device)
                visited_mask_step = torch.zeros((1, n_nodes), dtype=torch.bool, device=self.device)
                for v in visited:
                    if v < n_nodes:
                        visited_mask_step[0, int(v)] = True
                try:
                    # 前向只为单步得到 logits
                    # node_feats_t 已在 device 上，无需额外拷贝
                    action_probs, action_logits = self.nn_model(node_feats_t, current_tensor, visited_mask_step)
                    target = torch.tensor([nxt], dtype=torch.long, device=self.device)
                    loss_step = ce_loss_fn(action_logits, target)
                    policy_loss_acc = policy_loss_acc + loss_step
                    policy_count += 1
                except Exception:
                    pass
                visited.add(nxt)
                # 如果已达到最大 policy 步数，跳出所有循环以限制计算量
                if policy_count >= max_policy_steps:
                    break
            if policy_count >= max_policy_steps:
                break

        if policy_count > 0:
            policy_loss = policy_loss_acc / float(policy_count)
        else:
            policy_loss = torch.tensor(0.0, device=self.device)

        total_loss = policy_loss + vloss
        try:
            total_loss.backward()
            self.nn_optimizer.step()
            self.nn_optimizer.zero_grad()
        except Exception:
            pass

        # 如果使用 CUDA，主动释放缓存以降低峰值占用
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 返回标量损失
        try:
            return float(policy_loss.detach().cpu().item()), float(vloss.detach().cpu().item())
        except Exception:
            return 0.0, 0.0

    def train_supervised(self, ground_truth_routes, epochs=5, batch_size=8):
        """使用给定的 ground-truth 路线列表对 policy 网络做行为克隆预训练。

        参数:
          ground_truth_routes: List[List[List[int]]] 或者单一 List[List[int]]（每个路由为 [0,a,b,0]）
          epochs: 训练轮数
          batch_size: 每批样本数量（按解计）
        返回: 无（在控制台打印每 epoch loss）
        """
        if not self.use_nn or self.nn_model is None or self.nn_optimizer is None:
            print("NN 未启用或未初始化，跳过监督预训练。")
            return

        # 统一输入格式：列表的列表（每个元素是一个解：List[routes])
        dataset = []
        if isinstance(ground_truth_routes, dict):
            # dict of instance->routes_list: flatten values
            for v in ground_truth_routes.values():
                if isinstance(v, list):
                    # 如果 v 是 routes 列表（List[List[int]]），把它作为一个样本
                    dataset.append(v)
        elif isinstance(ground_truth_routes, list):
            # 判断是单个解（list of routes) 还是多解的集合
            if ground_truth_routes and isinstance(ground_truth_routes[0], list) and ground_truth_routes and isinstance(ground_truth_routes[0][0] if ground_truth_routes[0] else [], list):
                # list of solutions
                dataset = ground_truth_routes
            else:
                # single solution (list of routes)
                dataset = [ground_truth_routes]
        else:
            print("Unsupported ground_truth_routes format")
            return

        if not dataset:
            print("没有找到可用的监督样本，跳过预训练。")
            return

        node_feats = self._build_node_features()
        n_nodes = node_feats.shape[0]
        node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')

        debug_bc = bool(self.config.get('debug_bc', False))
        for ep in range(int(epochs)):
            random.shuffle(dataset)
            total_loss = 0.0
            batches = 0
            debug_prints = 0
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                self.nn_model.train()
                self.nn_optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device)
                sample_count = 0
                for routes in batch:
                    # flatten sequence of customers (exclude depot)
                    seq = []
                    for r in routes:
                        for n in r:
                            if n != 0:
                                try:
                                    seq.append(int(n))
                                except Exception:
                                    # skip non-int entries
                                    pass
                    if not seq:
                        continue

                    # 保护性过滤：确保所有索引在当前实例的节点范围内（0..n_nodes-1）
                    seq = [int(n) for n in seq if 0 <= int(n) < n_nodes]
                    if not seq:
                        # 如果过滤后为空，跳过该解并打印调试信息（若开启）
                        if debug_bc:
                            print(f"[BC DEBUG] 跳过样本：所有目标索引超出范围或被过滤，原始 routes={routes}")
                        continue
                    visited = set([0])
                    for t_idx, nxt in enumerate(seq):
                        current = 0 if t_idx == 0 else seq[t_idx - 1]
                        current_tensor = torch.tensor([current], dtype=torch.long, device=self.device)
                        visited_mask_step = torch.zeros((1, n_nodes), dtype=torch.bool, device=self.device)
                        for v in visited:
                            if v < n_nodes:
                                visited_mask_step[0, int(v)] = True
                        try:
                            out = self.nn_model(node_feats_t, current_tensor, visited_mask_step)
                            # 支持两种返回格式：(probs, logits) 或 (probs, logits, value)
                            if isinstance(out, tuple) or isinstance(out, list):
                                if len(out) >= 2:
                                    action_logits = out[1]
                                else:
                                    action_logits = out[0]
                            else:
                                action_logits = out

                            # BC debug: 打印 logits/target/visited_mask 的形状和摘要（有限次数）
                            if debug_bc and debug_prints < 5:
                                try:
                                    print(f"[BC DEBUG] epoch{ep+1} batch_idx {i} step {t_idx}: action_logits.shape={getattr(action_logits, 'shape', None)}, target={nxt}, visited_sum={int(visited_mask_step.sum().item())}")
                                except Exception:
                                    pass
                                debug_prints += 1

                            # 目标张量
                            target = torch.tensor([nxt], dtype=torch.long, device=self.device)
                            # 如果 target 超出 logits 维度范围，打印警告（有助于定位 .sol 索引偏移问题）
                            try:
                                if action_logits.dim() >= 2 and (target.item() >= action_logits.size(-1)):
                                    print(f"[BC WARN] target index {target.item()} >= action_logits.size({action_logits.size(-1)}) - 可能索引越界")
                            except Exception:
                                pass

                            loss_step = ce_loss_fn(action_logits, target)
                            batch_loss = batch_loss + loss_step
                            sample_count += 1
                        except Exception as ex:
                            if debug_bc:
                                print(f"[BC DEBUG] step exception: {ex}")
                            # 忽略单步失败
                            pass
                        visited.add(nxt)

                if sample_count > 0:
                    # 保护性处理：规范化并防止非数或异常大值
                    batch_loss = batch_loss / float(max(1, sample_count))
                    try:
                        loss_val = float(batch_loss.detach().cpu().item())
                    except Exception:
                        loss_val = float('nan')

                    # 如果 loss 非有限或过大，记录并用可表示的替代值，同时打印诊断信息
                    if not np.isfinite(loss_val) or loss_val > 1e6:
                        print(f"[BC WARN] 非常规 batch loss 值: {loss_val} (被修正)")
                        loss_val = 1e6

                    try:
                        batch_loss.backward()
                        self.nn_optimizer.step()
                    except Exception:
                        # 若反向或优化器出错，只记录 loss，不中断训练循环
                        pass

                    total_loss += float(loss_val)
                    batches += 1

            # 保护性平均：如果没有有效批次则记为 NaN
            if batches == 0:
                avg_loss = float('nan')
            else:
                avg_loss = total_loss / float(batches)

            # 如果 avg_loss 非有限或异常，打印诊断并替换为可序列化的大值
            if not np.isfinite(avg_loss):
                print(f"[BC WARN] epoch {ep+1} 计算到非有限 avg_loss={avg_loss}")
                store_loss = 1e6
            else:
                store_loss = float(avg_loss)

            print(f"BC epoch {ep+1}/{int(epochs)} avg_loss={store_loss:.6f}")
            # 记录到训练历史以便可视化（保持数值可序列化）
            self.training_history.setdefault('bc_loss', []).append(float(store_loss))

    def fast_non_dominated_sort(self, population):
        if not population:
            return []
        S = {p: [] for p in population}
        n = {p: 0 for p in population}
        fronts = [[]]
        for p in population:
            for q in population:
                if p is q:
                    continue
                if self._dominates_obj(p.objectives, q.objectives):
                    S[p].append(q)
                elif self._dominates_obj(q.objectives, p.objectives):
                    n[p] += 1
            if n[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_f = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_f.append(q)
            i += 1
            if next_f:
                fronts.append(next_f)
            else:
                break
        return fronts

    def _dominates_obj(self, a, b):
        # 最小化问题：a 支配 b 当且仅当 a <= b 对所有目标且至少一个 <
        a = list(a)
        b = list(b)
        le = all(x <= y for x, y in zip(a, b))
        lt = any(x < y for x, y in zip(a, b))
        return le and lt

    def calculate_crowding_distance(self, front):
        if not front:
            return np.array([])
        m = len(front[0].objectives)
        N = len(front)
        distances = np.zeros(N)
        obj = np.array([ind.objectives for ind in front], dtype=float)
        for i in range(m):
            idx = np.argsort(obj[:, i])
            distances[idx[0]] = distances[idx[-1]] = float('inf')
            denom = obj[idx[-1], i] - obj[idx[0], i]
            if denom == 0:
                continue
            for j in range(1, N - 1):
                distances[idx[j]] += (obj[idx[j + 1], i] - obj[idx[j - 1], i]) / denom
        return distances

    def savings_algorithm_solution(self):
        n = self.problem.num_customers
        if n <= 0:
            return []
        savings = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                s = self.problem.distance_matrix[0][i] + self.problem.distance_matrix[0][j] - self.problem.distance_matrix[i][j]
                savings.append((s, i, j))
        savings.sort(reverse=True)
        routes = [[0, i, 0] for i in range(1, n + 1)]
        for _, i, j in savings:
            ri = self.find_route_containing(routes, i)
            rj = self.find_route_containing(routes, j)
            if ri != rj and ri >= 0 and rj >= 0:
                newr = routes[ri][:-1] + routes[rj][1:]
                load = sum(self.problem.demands[c] for c in newr if c != 0)
                if load <= self.problem.vehicle_capacity:
                    a, b = sorted([ri, rj], reverse=True)
                    routes.pop(a)
                    routes.pop(b)
                    routes.append(newr)
        return [r for r in routes if len(r) > 2]

    def find_route_containing(self, routes, customer):
        for idx, r in enumerate(routes):
            if customer in r:
                return idx
        return -1

