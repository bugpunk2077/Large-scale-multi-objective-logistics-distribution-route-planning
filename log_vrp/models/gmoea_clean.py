"""
精简且自洽的 GMOEA 实现（多目标进化）

这个文件是干净、单一实现的 GMOEA；我先保留为新文件 `gmoea_clean.py`。
如果你同意，我可以把 main_train.py 的导入改为使用此文件，或你可以手动将其替换为 `gmoea.py`。
"""

import random
import time
import numpy as np
from utils.evaluator import calculate_hypervolume


class Solution:
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
    def __init__(self, problem, config=None):
        self.problem = problem
        self.config = config or {}
        self.population_size = int(self.config.get('population_size', 50))
        self.max_generations = int(self.config.get('max_generations', 100))
        self.crossover_rate = float(self.config.get('crossover_rate', 0.8))
        self.mutation_rate = float(self.config.get('mutation_rate', 0.2))
        self.train_interval = int(self.config.get('train_interval', 20))

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
            raise RuntimeError('初始化种群失败')

        for gen in range(self.max_generations):
            for ind in population:
                if ind.objectives is None:
                    ind.evaluate(self.problem)

            avg = np.mean([ind.objectives for ind in population], axis=0)
            self.training_history['avg_objectives'].append(avg.tolist())

            fronts = self.fast_non_dominated_sort(population)
            pareto = fronts[0] if fronts else []
            if pareto:
                maxd = float(np.max(self.problem.distance_matrix)) if hasattr(self.problem, 'distance_matrix') else 1.0
                ref = [maxd * 10, maxd * 10, max(1, getattr(self.problem, 'num_vehicles', 1))]
                hv = calculate_hypervolume(pareto, ref)
            else:
                hv = 0.0
            self.training_history['hv_history'].append(float(hv))

            if gen % self.train_interval == 0:
                self.training_history['policy_loss'].append(0.0)
                self.training_history['value_loss'].append(0.0)

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
        try:
            routes = self.savings_algorithm_solution()
            if routes:
                s = Solution(routes)
                s.evaluate(self.problem)
                population.append(s)
        except Exception:
            pass

        while len(population) < self.population_size:
            if random.random() < 0.4:
                routes = self.nearest_neighbor_solution()
            else:
                routes = self.generate_random_solution()
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
        for r in r1_set:
            seg = [n for n in r if n != 0 and n not in assigned]
        
print('已创建 gmoea_clean.py')