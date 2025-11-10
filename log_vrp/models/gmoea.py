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


class Solution:
    """简单容器：routes + objectives

    routes: List[List[int]]，每条路径以 0（仓库）开始和结束
    objectives: [total_distance, max_route_time, num_vehicles]
    """

    def __init__(self, routes, objectives=None):
        self.routes = routes
        self.objectives = objectives

    def evaluate(self, problem):
        total_distance = 0.0
        max_time = 0.0
        for route in self.routes:
            if len(route) <= 2:
                continue
            route_dist = 0.0
            route_time = 0.0
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                route_dist += problem.distance_matrix[a][b]
                route_time += problem.distance_matrix[a][b] / getattr(problem, 'vehicle_speed', 1.0)
                if b != 0 and hasattr(problem, 'service_times') and hasattr(problem, 'time_windows'):
                    route_time = max(route_time, problem.time_windows[b][0])
                    route_time += problem.service_times[b]
            total_distance += route_dist
            max_time = max(max_time, route_time)
        num_vehicles = sum(1 for r in self.routes if len(r) > 2)
        self.objectives = [total_distance, max_time, num_vehicles]
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
                maxd = float(np.max(self.problem.distance_matrix)) if hasattr(self.problem, 'distance_matrix') else 1.0
                ref = [maxd * 10, maxd * 10, max(1, getattr(self.problem, 'num_vehicles', 1))]
                hv = calculate_hypervolume(pareto, ref)
            else:
                hv = 0.0
            self.training_history['hv_history'].append(float(hv))

            # 实时打印（可配置）
            if self.verbose and (gen % self.print_interval == 0 or gen == self.max_generations - 1):
                # population 的当前最小/均值目标
                try:
                    pop_objs = np.array([ind.objectives for ind in population], dtype=float)
                    bests = pop_objs.min(axis=0)
                    avgs = pop_objs.mean(axis=0)
                    pareto_size = len(pareto)
                    print(f"Gen {gen+1}/{self.max_generations} | Pareto {pareto_size} | HV {hv:.4g} | best(dist,time,veh) = ({bests[0]:.2f}, {bests[1]:.2f}, {int(bests[2])}) | avg = ({avgs[0]:.2f}, {avgs[1]:.2f}, {avgs[2]:.2f})")
                except Exception:
                    print(f"Gen {gen+1}/{self.max_generations} | Pareto {len(pareto)} | HV {hv:.4g}")

            # NN 占位：不主动训练，仅填零，保证接口不变
            if gen % self.train_interval == 0:
                self.training_history['policy_loss'].append(0.0)
                self.training_history['value_loss'].append(0.0)

            # 产生子代并选择下一代
            offspring = self.evolve_population(population)
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

        # 填充：最近邻与随机混合
        while len(population) < self.population_size:
            if random.random() < 0.4:
                routes = self.nearest_neighbor_solution()
            else:
                routes = self.generate_random_solution()
            routes = self.balance_routes(routes)
            s = Solution(routes)
            s.evaluate(self.problem)
            population.append(s)
        return population

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

    def evolve_population(self, population):
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
            child_routes = self.repair_routes(child_routes)
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

    def repair_routes(self, routes):
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
        # 最后做一次平衡，尝试让车辆分配更均匀且不超过车辆上限
        return self.balance_routes(cleaned)

    def balance_routes(self, routes):
        """把客户均匀分配到不超过 problem.num_vehicles 的车辆上，尽量满足容量约束。

        算法：将所有客户摊平，然后轮询到 num_vehicles 个车上（尝试尊重容量），
        如果某个客户无法放入任何车（容量不足的极端情况），则会创建额外车道。
        """
        if not routes:
            return []
        customers = [c for r in routes for c in r if c != 0]
        num_v = int(getattr(self.problem, 'num_vehicles', max(1, len(routes))))
        if num_v <= 0:
            num_v = 1

        new_routes = [[0] for _ in range(num_v)]
        loads = [0.0 for _ in range(num_v)]
        idx = 0
        for c in customers:
            d = float(self.problem.demands[c])
            placed = False
            # 尝试从当前索引开始循环寻找能放下的车
            for attempt in range(num_v):
                j = (idx + attempt) % num_v
                if loads[j] + d <= self.problem.vehicle_capacity:
                    new_routes[j].append(c)
                    loads[j] += d
                    placed = True
                    idx = (j + 1) % num_v
                    break
            if not placed:
                # 无法放入任何现有车辆（可能容量太小），新建一条车道来容纳
                new_routes.append([0, c])
                loads.append(d)
                idx = len(new_routes) - 1

        # 关闭路径 (append depot end) 并移除空路径
        final = []
        for r in new_routes:
            if len(r) > 1:
                r.append(0)
                final.append(r)
        return final

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

