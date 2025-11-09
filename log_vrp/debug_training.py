import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from tqdm import tqdm

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 简化的问题定义
class SimpleVRPProblem:
    def __init__(self, num_customers=20):  # 使用更小的规模进行调试
        self.num_customers = num_customers
        self.vehicle_capacity = 200
        self.num_vehicles = 5
        
        # 生成简单数据
        np.random.seed(42)
        self.coordinates = [(0, 0)]  # 仓库
        self.demands = [0]
        
        for i in range(num_customers):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            demand = np.random.randint(1, 20)
            self.coordinates.append((x, y))
            self.demands.append(demand)
        
        # 计算距离矩阵
        self.distance_matrix = self.calculate_distance_matrix()
    
    def calculate_distance_matrix(self):
        n = len(self.coordinates)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.coordinates[i]
                    x2, y2 = self.coordinates[j]
                    dist_matrix[i][j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return dist_matrix

class SimpleSolution:
    def __init__(self, routes, problem):
        self.routes = routes
        self.problem = problem
        self.objectives = self.evaluate()
    
    def evaluate(self):
        total_distance = 0
        max_route_time = 0
        num_vehicles = len([r for r in self.routes if len(r) > 2])
        
        for route in self.routes:
            if len(route) <= 2:
                continue
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += self.problem.distance_matrix[route[i]][route[i+1]]
            total_distance += route_distance
            max_route_time = max(max_route_time, route_distance)
        
        return [total_distance, max_route_time, num_vehicles]

class DebugGMOEA:
    def __init__(self, problem, config):
        self.problem = problem
        self.config = config
        self.population = self.initialize_population()
        self.error_log = []
    
    def initialize_population(self):
        """初始化种群 - 简化版本"""
        population = []
        for _ in range(self.config['population_size']):
            routes = self.generate_valid_solution()
            solution = SimpleSolution(routes, self.problem)
            population.append(solution)
        return population
    
    def generate_valid_solution(self):
        """生成有效的解决方案"""
        customers = list(range(1, self.problem.num_customers + 1))
        random.shuffle(customers)
        
        routes = []
        current_route = [0]
        current_load = 0
        
        for customer in customers:
            demand = self.problem.demands[customer]
            
            if current_load + demand <= self.problem.vehicle_capacity:
                current_route.append(customer)
                current_load += demand
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, customer]
                current_load = demand
                
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
            
        return routes
    
    def dominates(self, sol1, sol2):
        """判断支配关系"""
        try:
            obj1 = sol1.objectives
            obj2 = sol2.objectives
            
            # 检查有效性
            if any(not np.isfinite(obj) for obj in obj1) or any(not np.isfinite(obj) for obj in obj2):
                return False
            
            not_worse = all(obj1[i] <= obj2[i] for i in range(len(obj1)))
            better = any(obj1[i] < obj2[i] for i in range(len(obj1)))
            
            return not_worse and better
        except Exception as e:
            self.error_log.append(f"支配关系判断错误: {e}")
            return False
    
    def fast_non_dominated_sort(self, population):
        """快速非支配排序 - 带完整错误处理"""
        print(f"开始非支配排序，种群大小: {len(population)}")
        
        if not population:
            print("种群为空")
            return []
        
        # 检查种群有效性
        valid_population = []
        for i, ind in enumerate(population):
            try:
                if hasattr(ind, 'objectives') and ind.objectives and all(np.isfinite(obj) for obj in ind.objectives):
                    valid_population.append(ind)
                else:
                    print(f"个体 {i} 无效: {getattr(ind, 'objectives', '无目标值')}")
            except Exception as e:
                print(f"检查个体 {i} 时出错: {e}")
        
        print(f"有效个体数量: {len(valid_population)}")
        
        if len(valid_population) < 2:
            print("有效个体不足，返回空前沿")
            return []
        
        fronts = [[]]
        domination_count = [0] * len(valid_population)
        dominated_solutions = [[] for _ in range(len(valid_population))]
        
        # 计算支配关系
        for i, p in enumerate(valid_population):
            for j, q in enumerate(valid_population):
                if i == j:
                    continue
                try:
                    if self.dominates(p, q):
                        dominated_solutions[i].append(j)
                    elif self.dominates(q, p):
                        domination_count[i] += 1
                except Exception as e:
                    print(f"计算支配关系时出错 (i={i}, j={j}): {e}")
                    continue
            
            if domination_count[i] == 0:
                p.rank = 0
                fronts[0].append(p)
        
        print(f"第一前沿大小: {len(fronts[0])}")
        
        if not fronts[0]:
            print("没有找到非支配解")
            return []
        
        # 构建后续前沿
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                try:
                    p_idx = valid_population.index(p)
                    for q_idx in dominated_solutions[p_idx]:
                        domination_count[q_idx] -= 1
                        if domination_count[q_idx] == 0:
                            valid_population[q_idx].rank = i + 1
                            next_front.append(valid_population[q_idx])
                except Exception as e:
                    print(f"构建前沿时出错: {e}")
                    continue
            
            i += 1
            if next_front:
                fronts.append(next_front)
                print(f"第 {i} 前沿大小: {len(next_front)}")
            else:
                break
        
        print(f"总共找到 {len(fronts)} 个前沿")
        return fronts
    
    def run_debug_training(self):
        """调试训练过程"""
        print("开始调试训练...")
        
        for generation in range(min(10, self.config['max_generations'])):  # 只运行10代进行调试
            print(f"\n=== 第 {generation} 代 ===")
            
            try:
                # 评估种群
                print("评估种群...")
                for i, individual in enumerate(self.population):
                    try:
                        individual.objectives = individual.evaluate()
                        print(f"个体 {i}: 目标值 = {individual.objectives}")
                    except Exception as e:
                        print(f"评估个体 {i} 时出错: {e}")
                        individual.objectives = [float('inf'), float('inf'), float('inf')]
                
                # 非支配排序
                print("进行非支配排序...")
                fronts = self.fast_non_dominated_sort(self.population)
                
                if not fronts:
                    print("没有前沿，跳过该代")
                    continue
                
                print(f"找到 {len(fronts)} 个前沿")
                for i, front in enumerate(fronts):
                    print(f"前沿 {i}: {len(front)} 个解")
                    if front:
                        objectives = np.array([sol.objectives for sol in front])
                        print(f"  目标范围: 距离[{objectives[:,0].min():.1f}-{objectives[:,0].max():.1f}], "
                              f"时间[{objectives[:,1].min():.1f}-{objectives[:,1].max():.1f}], "
                              f"车辆[{objectives[:,2].min()}-{objectives[:,2].max()}]")
                
                # 简单选择：只保留第一前沿
                if fronts and fronts[0]:
                    self.population = fronts[0][:self.config['population_size']]
                
                # 补充种群
                while len(self.population) < self.config['population_size']:
                    new_solution = SimpleSolution(self.generate_valid_solution(), self.problem)
                    self.population.append(new_solution)
                    
            except Exception as e:
                print(f"第 {generation} 代发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 返回结果
        final_population = [ind for ind in self.population if hasattr(ind, 'objectives') and all(np.isfinite(obj) for obj in ind.objectives)]
        
        if final_population:
            fronts = self.fast_non_dominated_sort(final_population)
            pareto_front = fronts[0] if fronts else []
            print(f"\n最终结果: {len(pareto_front)} 个Pareto解")
        else:
            pareto_front = []
            print("\n最终结果: 没有找到有效解")
        
        return pareto_front, self.population

def main():
    print("GMOEA调试版本")
    print("=" * 50)
    
    # 使用简化配置
    config = {
        'population_size': 20,  # 小种群
        'max_generations': 10,  # 少代数
    }
    
    # 创建简单问题
    problem = SimpleVRPProblem(num_customers=10)  # 小问题
    
    print(f"问题规模: {problem.num_customers}个客户点")
    print(f"车辆容量: {problem.vehicle_capacity}")
    
    # 运行调试
    algorithm = DebugGMOEA(problem, config)
    pareto_front, population = algorithm.run_debug_training()
    
    # 输出错误日志
    if algorithm.error_log:
        print("\n错误日志:")
        for error in algorithm.error_log:
            print(f"  - {error}")

if __name__ == "__main__":
    main()