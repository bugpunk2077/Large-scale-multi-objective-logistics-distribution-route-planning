import numpy as np
import torch
import torch.optim as optim
import random
import time
import copy
from .neural_network import NeuralVRPSolver, save_model

class Solution:
    """解决方案类"""
    def __init__(self, routes, problem):
        # 存储 routes 的深拷贝，防止外部对原列表的就地修改影响本实例
        try:
            self.routes = [list(r) for r in routes]
        except Exception:
            # 如果 routes 不是可迭代的，尝试使用 deepcopy 作为回退
            self.routes = copy.deepcopy(routes)
        self.problem = problem
        self.objectives = self.evaluate()
        self.rank = None
        self.crowding_distance = 0.0
        self.features = self.extract_features()
        
    def extract_features(self):
        """提取解决方案的特征用于神经网络"""
        # 简化的特征提取，实际可以根据需要扩展
        features = []
        
        # 路径长度特征
        route_lengths = [len(route) for route in self.routes]
        features.extend([np.mean(route_lengths), np.std(route_lengths), max(route_lengths)])
        
        # 距离特征
        total_dist = self.objectives[0]
        features.append(total_dist)
        
        # 负载特征
        route_loads = []
        for route in self.routes:
            load = sum(self.problem.demands[node] for node in route if node != 0)
            route_loads.append(load)
        features.extend([np.mean(route_loads), np.std(route_loads), max(route_loads)])
        
        return np.array(features)
        
    def evaluate(self):
        """评估目标函数"""
        total_distance = 0
        max_route_time = 0
        num_vehicles = len([r for r in self.routes if len(r) > 2])  # 有效车辆数
        
        for route in self.routes:
            if len(route) <= 2:  # 只有仓库
                continue
                
            route_distance = 0
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i+1]
                route_distance += self.problem.distance_matrix[from_node][to_node]
            
            total_distance += route_distance
            max_route_time = max(max_route_time, route_distance)
            
        return [total_distance, max_route_time, num_vehicles]
    
    def to_tensor(self, max_route_length=50):
        """将解决方案转换为张量格式"""
        # 将路径填充到固定长度
        padded_routes = []
        for route in self.routes:
            padded_route = route + [-1] * (max_route_length - len(route))
            padded_routes.append(padded_route[:max_route_length])
        
        # 如果路径数量不足，用-1填充
        while len(padded_routes) < self.problem.num_vehicles:
            padded_routes.append([-1] * max_route_length)
            
        return torch.tensor(padded_routes[:self.problem.num_vehicles], dtype=torch.long)

class GMOEA:
    """基于图神经网络的多目标进化算法"""
    
    def __init__(self, problem, config):
        self.problem = problem
        self.config = config
        # 明确保存常用配置为属性，避免后续方法直接引用未定义的属性
        self.population_size = int(config.get('population_size', 30))
        self.max_generations = int(config.get('max_generations', 50))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化神经网络
        self.model = NeuralVRPSolver(
            node_dim=5,  # x, y, demand, ready_time, due_date
            hidden_dim=config['hidden_dim'],
            n_actions=problem.num_customers + 1,  # 所有节点（包括仓库）
            n_objectives=3
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get('lr_step_size', 50),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        # 初始化种群
        self.population = self.initialize_population()
        
        # 训练历史
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'avg_objectives': [],
            'hv_history': []
        }
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        valid_count = 0
        
        for i in range(self.population_size):
            try:
                # 创建初始解
                individual = self.create_individual()
                
                # 评估解
                if self.evaluate_individual(individual):
                    population.append(individual)
                    valid_count += 1
                else:
                    print(f"个体 {i} 评估失败")
                    
            except Exception as e:
                print(f"初始化个体 {i} 时出错: {e}")
                continue
        
        print(f"成功初始化 {valid_count}/{self.population_size} 个有效个体")
        
        if valid_count == 0:
            print("警告: 没有有效个体，尝试创建简单可行解")
            # 创建简单的可行解作为后备
            simple_individual = self.create_simple_feasible_solution()
            if simple_individual and self.evaluate_individual(simple_individual):
                population.append(simple_individual)
                print("已创建简单可行解作为后备")
        
        return population
    
    def generate_random_solution(self):
        """基于聚类和载重平衡生成初始解"""
        customers = list(range(1, self.problem.num_customers + 1))
        
        # 将客户按位置聚类（简单的基于距离的分组）
        clusters = []
        remaining = customers.copy()
        while remaining:
            # 随机选择一个中心点
            center = random.choice(remaining)
            cluster = [center]
            remaining.remove(center)
            
            # 找到距离最近的K个点加入簇（限制更小以产生更多路径）
            K = min(8, len(remaining))  # 每簇最多8个点
            if remaining:
                distances = [(c, self.problem.distance_matrix[center][c]) 
                           for c in remaining]
                distances.sort(key=lambda x: x[1])
                
                # 添加最近的点，同时确保pickup/delivery相对平衡
                current_load = self.problem.demands[center]
                pickups_sum = max(0, self.problem.demands[center])
                deliveries_sum = abs(min(0, self.problem.demands[center]))
                
                for cust, _ in distances[:K]:
                    demand = self.problem.demands[cust]
                    new_pickups = pickups_sum + max(0, demand)
                    new_deliveries = deliveries_sum + abs(min(0, demand))
                    
                    # 检查添加后是否仍然平衡（pickup和delivery的差不超过容量的40%）
                    if (abs(new_pickups - new_deliveries) <= self.problem.vehicle_capacity * 0.4 and
                        max(new_pickups, new_deliveries) <= self.problem.vehicle_capacity * 0.8):
                        cluster.append(cust)
                        current_load += demand
                        pickups_sum = new_pickups
                        deliveries_sum = new_deliveries
                        remaining.remove(cust)
            
            clusters.append(cluster)
        
        # 将每个簇转换为可行路径
        routes = []
        for cluster in clusters:
            if not cluster:
                continue
                
            # 对簇内客户重排序：混合 pickup/delivery 以保持可行性
            pickups = [(c, self.problem.demands[c]) for c in cluster 
                      if self.problem.demands[c] > 0]
            deliveries = [(c, self.problem.demands[c]) for c in cluster 
                         if self.problem.demands[c] <= 0]
            
            # 初始路径从较大的 pickup 开始
            route = [0]  # 从仓库出发
            
            # 交替添加 pickup 和 delivery 以平衡载重
            current_load = 0
            while pickups or deliveries:
                # 如果当前负载较低且有pickup，优先pickup
                if current_load < self.problem.vehicle_capacity/2 and pickups:
                    cust, demand = pickups.pop(
                        max(range(len(pickups)), 
                            key=lambda i: pickups[i][1]))
                    route.append(cust)
                    current_load += demand
                # 否则尝试delivery
                elif deliveries:
                    cust, demand = deliveries.pop(0)
                    route.append(cust)
                    current_load += demand
                # 如果只剩pickup但负载已高，开新路径
                elif pickups:
                    route.append(0)
                    routes.append(route)
                    route = [0]
                    current_load = 0
                    continue
                
                # 更严格的路径切分条件
                need_new_route = False
                if current_load > self.problem.vehicle_capacity * 0.8:  # 降低阈值，更早切分
                    need_new_route = True
                elif len(route) > 8:  # 限制每条路径最多服务8个客户
                    need_new_route = True
                
                if need_new_route:
                    route.append(0)
                    routes.append(route)
                    route = [0]
                    current_load = 0
            
            if len(route) > 1:
                route.append(0)
                routes.append(route)
                
            # 如果这个簇生成的路径太少，尝试进一步拆分最长的路径
            cluster_routes = [r for r in routes if len(r) > 2]
            while len(cluster_routes) < 3 and any(len(r) > 6 for r in cluster_routes):
                longest_route = max(cluster_routes, key=len)
                if len(longest_route) <= 4:  # 太短的路径不拆分
                    break
                    
                # 找一个相对平衡的拆分点
                mid = len(longest_route) // 2
                r1 = [0] + longest_route[1:mid] + [0]
                r2 = [0] + longest_route[mid:-1] + [0]
                
                # 检查拆分后的路径是否可行
                r1_stats = route_stats(r1)
                r2_stats = route_stats(r2)
                if (abs(r1_stats[0] - r1_stats[1]) <= self.problem.vehicle_capacity * 0.4 and
                    abs(r2_stats[0] - r2_stats[1]) <= self.problem.vehicle_capacity * 0.4):
                    cluster_routes.remove(longest_route)
                    cluster_routes.extend([r1, r2])
                else:
                    break  # 如果拆分后不平衡，就停止拆分
                    
            routes = [r for r in routes if len(r) <= 2] + cluster_routes
        
        return routes

    # 兼容旧代码接口：生成一个被认为“有效”的解（优先使用最近邻）
    def generate_valid_solution(self):
        """兼容层：返回一个有效解供种群补充使用。"""
        # 为了增加初始种群多样性，随机选择生成策略
        try:
            if random.random() < 0.5:
                # 使用随机构造以提高多样性
                routes = self.generate_random_solution()
                # 应用车辆数约束（可选）
                routes = self.enforce_vehicle_count(routes,
                                                    min_v=self.config.get('min_vehicles', 1),
                                                    max_v=self.config.get('max_vehicles', self.problem.num_vehicles))
                return routes

            routes = self.nearest_neighbor_solution()
            if routes:
                return routes
        except Exception:
            pass

        # 回退到随机生成并应用车辆数约束
        routes = self.generate_random_solution()
        routes = self.enforce_vehicle_count(routes,
                                            min_v=self.config.get('min_vehicles', 1),
                                            max_v=self.config.get('max_vehicles', self.problem.num_vehicles))
        return routes

    def enforce_vehicle_count(self, routes, min_v=1, max_v=None):
        """调整 routes 使得车辆数位于 [min_v, max_v] 范围内。
        同时确保：
        1. pickup/delivery 在路径内相对平衡
        2. 路径长度相对均匀
        3. 避免过度合并
        """
        if max_v is None:
            max_v = self.problem.num_vehicles

        # 清理空路径
        routes = [r for r in routes if len(r) > 2]
        
        def route_stats(route):
            """计算路径的关键统计信息：
            - pickup_sum: 取货总量
            - delivery_sum: 送货总量
            - length: 路径包含的客户数
            - total_distance: 路径总距离
            """
            if len(route) <= 2:
                return 0, 0, 0, 0
            
            pickup_sum = 0
            delivery_sum = 0
            length = len(route) - 2  # 减去首尾的仓库
            total_distance = 0
            
            for i in range(len(route)-1):
                from_node = route[i]
                to_node = route[i+1]
                if from_node != 0:
                    demand = self.problem.demands[from_node]
                    if demand > 0:
                        pickup_sum += demand
                    else:
                        delivery_sum += abs(demand)
                total_distance += self.problem.distance_matrix[from_node][to_node]
                
            return pickup_sum, delivery_sum, length, total_distance

        # helper: compute load of a route
        def route_load(route):
            # 返回该路径的净需求（pickup 为正，delivery 为负）的和
            return sum(self.problem.demands[c] for c in route if c != 0)

        def route_feasibility_stats(route):
            """计算路径的前缀和统计，返回 (initial_load_needed, required_capacity, final_net)

            解释：对于沿序列的 signed demands d_i，计算前缀和 s_k = sum_{i<=k} d_i。
            若 min_prefix < 0，则车辆需要在出发时携带 initial_load = -min_prefix 才能保证途中负载不为负。
            同时，途中最大负载为 initial_load + max_prefix，因此所需车辆容量为 initial_load + max_prefix。
            """
            seq = [self.problem.demands[c] for c in route if c != 0]
            prefix = 0.0
            min_pref = 0.0
            max_pref = 0.0
            for d in seq:
                prefix += float(d)
                if prefix < min_pref:
                    min_pref = prefix
                if prefix > max_pref:
                    max_pref = prefix
            initial_needed = -min_pref if min_pref < 0 else 0.0
            required_capacity = initial_needed + max_pref
            return initial_needed, required_capacity, prefix

        # 合并超出上限时：在保持平衡的前提下谨慎合并
        if len(routes) > max_v:
            while len(routes) > max_v:
                merged = False
                # 计算所有路径的统计信息
                stats = [(i, route_stats(route)) for i, route in enumerate(routes)]
                # 按路径长度和负载排序，优先合并短路径
                stats.sort(key=lambda x: (x[1][2], x[1][0] + x[1][1]))
                
                # 尝试合并相邻的短路径
                for i in range(len(stats)-1):
                    if merged:
                        break
                    i_idx = stats[i][0]
                    i_pickup, i_delivery, i_len, i_dist = stats[i][1]
                    
                    for j in range(i+1, min(i+3, len(stats))):
                        j_idx = stats[j][0]
                        j_pickup, j_delivery, j_len, j_dist = stats[j][1]
                        
                        # 检查合并是否会导致严重不平衡
                        if ((i_len + j_len > 10) or  # 限制合并后长度
                            (abs((i_pickup + j_pickup) - (i_delivery + j_delivery)) > 
                             self.problem.vehicle_capacity * 0.4)):  # 限制pickup/delivery差异
                            continue
                            
                        # 构造合并路径并检查可行性
                        r1, r2 = routes[i_idx], routes[j_idx]
                        new_route = r1[:-1] + r2[1:]
                        _, required_capacity, _ = route_feasibility_stats(new_route)
                        
                        if required_capacity <= self.problem.vehicle_capacity * 0.9:  # 留出10%余量
                            # 合并成功
                            a, b = sorted([i_idx, j_idx], reverse=True)
                            routes.pop(a)
                            routes.pop(b)
                            routes.append(new_route)
                            merged = True
                            break
                            
                if not merged:  # 如果无法找到好的合并方案，强制合并最小的两条
                    if len(routes) >= 2:
                        r1, r2 = routes[:2]
                        new_route = r1[:-1] + r2[1:]
                        routes = routes[2:] + [new_route]
                    else:
                        break  # 防止意外情况
                    if merged:
                        break
                # 若未能合并任何对，退出循环
                if not merged:
                    break

        # 拆分不足时：尝试从最长/最重的路径拆分
        if len(routes) < min_v:
            attempts = 0
            while len(routes) < min_v and attempts < 10:
                attempts += 1
                # 选择负载或长度最大的路径进行拆分
                idx = max(range(len(routes)), key=lambda k: (route_load(routes[k]), len(routes[k])))
                route = routes[idx]
                customers = [c for c in route if c != 0]
                if len(customers) <= 1:
                    # 无法进一步拆分
                    break

                # 尝试找到一个拆分点，使得左右两部分都不超载
                split_found = False
                for cut in range(1, len(customers)):
                    left = [0] + customers[:cut] + [0]
                    right = [0] + customers[cut:] + [0]
                    if route_load(left) <= self.problem.vehicle_capacity and route_load(right) <= self.problem.vehicle_capacity:
                        # 替换并添加
                        routes.pop(idx)
                        routes.append(left)
                        routes.append(right)
                        split_found = True
                        break

                if not split_found:
                    # 无法按容量拆分，尝试按数量拆分（忽略容量，作为最后手段）
                    cut = len(customers) // 2
                    left = [0] + customers[:cut] + [0]
                    right = [0] + customers[cut:] + [0]
                    routes.pop(idx)
                    routes.append(left)
                    routes.append(right)

        # 最终清理并返回
        routes = [r for r in routes if len(r) > 2]
        return routes

    def create_individual(self):
        """创建一个 Solution 实例（用于初始化）。"""
        routes = self.generate_valid_solution()
        routes = self.enforce_vehicle_count(routes,
            min_v=self.config.get('min_vehicles', 1),
            max_v=self.config.get('max_vehicles', self.problem.num_vehicles))
        return Solution(routes, self.problem)

    def evaluate_individual(self, individual):
        """评估并验证个体的目标值是否有效。"""
        try:
            individual.objectives = individual.evaluate()
            # 要求目标值均为有限数，且总距离大于0
            if (hasattr(individual, 'objectives') and
                individual.objectives and
                all(np.isfinite(obj) for obj in individual.objectives) and
                individual.objectives[0] > 0):
                return True
            return False
        except Exception:
            return False

    def create_simple_feasible_solution(self):
        """创建一个简单的可行解（被 initialize_population 作为后备使用）。

        优先使用节约算法或最近邻法生成更稳健的可行解。
        """
        try:
            routes = self.savings_algorithm_solution()
            if routes and all(len(r) > 2 for r in routes):
                return Solution(routes, self.problem)
        except Exception:
            pass

        # 回退到最近邻
        routes = self.nearest_neighbor_solution()
        return Solution(routes, self.problem)
    
    def nearest_neighbor_solution(self):
        """最近邻算法"""
        unvisited = set(range(1, self.problem.num_customers + 1))
        routes = []
        
        while unvisited:
            current_route = [0]
            current_load = 0
            current_pos = 0
            
            while unvisited:
                # 找到最近的未访问客户
                nearest = None
                min_dist = float('inf')
                
                for customer in unvisited:
                    if current_load + self.problem.demands[customer] <= self.problem.vehicle_capacity:
                        dist = self.problem.distance_matrix[current_pos][customer]
                        if dist < min_dist:
                            min_dist = dist
                            nearest = customer
                            
                if nearest is None:
                    break
                    
                current_route.append(nearest)
                current_load += self.problem.demands[nearest]
                current_pos = nearest
                unvisited.remove(nearest)
                
            current_route.append(0)
            routes.append(current_route)
            
        return routes
    
    def savings_algorithm_solution(self):
        """节约算法"""
        # 简化的节约算法实现
        num_customers = self.problem.num_customers
        savings = []
        
        # 计算节约值
        for i in range(1, num_customers + 1):
            for j in range(i + 1, num_customers + 1):
                saving = (self.problem.distance_matrix[0][i] + 
                         self.problem.distance_matrix[0][j] - 
                         self.problem.distance_matrix[i][j])
                savings.append((saving, i, j))
                
        savings.sort(reverse=True)
        
        # 初始路径
        routes = [[0, i, 0] for i in range(1, num_customers + 1)]
        
        for saving, i, j in savings:
            route_i = self.find_route_containing(routes, i)
            route_j = self.find_route_containing(routes, j)
            
            if route_i != route_j and self.can_merge(routes[route_i], routes[route_j]):
                self.merge_routes(routes, route_i, route_j, i, j)
                
        return [route for route in routes if len(route) > 2]
    
    def find_route_containing(self, routes, customer):
        for idx, route in enumerate(routes):
            if customer in route:
                return idx
        return -1
    
    def can_merge(self, route1, route2):
        # 使用前缀和可行性检查来判断两条路径合并后是否仍然满足车辆容量约束
        new_route = route1[:-1] + route2[1:]
        try:
            # 如果 required_capacity <= vehicle_capacity，则合并可行
            seq = [self.problem.demands[c] for c in new_route if c != 0]
        except Exception:
            return False

        prefix = 0.0
        min_pref = 0.0
        max_pref = 0.0
        for d in seq:
            prefix += float(d)
            if prefix < min_pref:
                min_pref = prefix
            if prefix > max_pref:
                max_pref = prefix
        initial_needed = -min_pref if min_pref < 0 else 0.0
        required_capacity = initial_needed + max_pref
        return required_capacity <= self.problem.vehicle_capacity
    
    def merge_routes(self, routes, idx1, idx2, i, j):
        route1 = routes[idx1]
        route2 = routes[idx2]
        
        # 简单的合并策略
        new_route = route1[:-1] + route2[1:]
        routes[idx1] = new_route
        routes.pop(idx2)
    
    def prepare_node_features(self):
        """准备节点特征用于神经网络"""
        n_nodes = len(self.problem.coordinates)
        node_features = np.zeros((n_nodes, 5))  # x, y, demand, ready_time, due_date
        
        for i in range(n_nodes):
            node_features[i, 0] = self.problem.coordinates[i][0]  # x
            node_features[i, 1] = self.problem.coordinates[i][1]  # y
            node_features[i, 2] = self.problem.demands[i]         # demand
            if hasattr(self.problem, 'time_windows'):
                node_features[i, 3] = self.problem.time_windows[i][0]  # ready_time
                node_features[i, 4] = self.problem.time_windows[i][1]  # due_date
        
        # 归一化特征
        node_features = (node_features - node_features.mean(axis=0)) / (node_features.std(axis=0) + 1e-8)
        
        return torch.tensor(node_features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def neural_crossover(self, parent1, parent2):
        """使用神经网络指导的交叉操作"""
        with torch.no_grad():
            # 准备节点特征
            node_features = self.prepare_node_features()
            # 编码父代解决方案
            parent1_tensor = parent1.to_tensor().unsqueeze(0).to(self.device)
            parent2_tensor = parent2.to_tensor().unsqueeze(0).to(self.device)
            # 使用价值网络评估父代
            value1 = self.model.value_net(node_features, parent1_tensor)
            value2 = self.model.value_net(node_features, parent2_tensor)
        # 基于价值评估选择交叉策略
        if value1.mean() > value2.mean():
            base_parent, other_parent = parent1, parent2
        else:
            base_parent, other_parent = parent2, parent1
        # 执行路径交叉
        child_routes = self.route_based_crossover(base_parent, other_parent)
        # 强制车辆数约束
        child_routes = self.enforce_vehicle_count(child_routes,
            min_v=self.config.get('min_vehicles', 1),
            max_v=self.config.get('max_vehicles', self.problem.num_vehicles))
        child = Solution(child_routes, self.problem)
        return child
    
    def neural_mutation(self, individual):
        """使用神经网络指导的变异操作"""
        mutated_routes = [route.copy() for route in individual.routes]
        mutation_type = random.choice(['swap', 'relocate', '2-opt', 'reverse'])
        if mutation_type == 'swap' and len(mutated_routes) >= 2:
            self.swap_mutation(mutated_routes)
        elif mutation_type == 'relocate' and len(mutated_routes) >= 2:
            self.relocate_mutation(mutated_routes)
        elif mutation_type == '2-opt' and mutated_routes:
            self.two_opt_mutation(mutated_routes)
        elif mutation_type == 'reverse' and mutated_routes:
            self.reverse_mutation(mutated_routes)
        # 强制车辆数约束
        mutated_routes = self.enforce_vehicle_count(mutated_routes,
            min_v=self.config.get('min_vehicles', 1),
            max_v=self.config.get('max_vehicles', self.problem.num_vehicles))
        return Solution(mutated_routes, self.problem)
    
    def swap_mutation(self, routes):
        """交换变异"""
        if len(routes) < 2:
            return
            
        route1_idx, route2_idx = random.sample(range(len(routes)), 2)
        route1, route2 = routes[route1_idx], routes[route2_idx]
        
        if len(route1) > 2 and len(route2) > 2:
            pos1 = random.randint(1, len(route1) - 2)
            pos2 = random.randint(1, len(route2) - 2)
            
            route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
    
    def relocate_mutation(self, routes):
        """重定位变异"""
        if len(routes) < 2:
            return
            
        source_route_idx = random.randint(0, len(routes) - 1)
        source_route = routes[source_route_idx]
        
        if len(source_route) <= 2:
            return
            
        customer_pos = random.randint(1, len(source_route) - 2)
        customer = source_route[customer_pos]
        
        # 从源路径移除客户
        source_route.pop(customer_pos)
        
        # 插入到目标路径
        target_route_idx = random.randint(0, len(routes) - 1)
        target_route = routes[target_route_idx]
        
        if target_route_idx == source_route_idx:
            # 同一路径内重定位
            new_pos = random.randint(1, len(target_route) - 1)
            target_route.insert(new_pos, customer)
        else:
            # 不同路径间重定位
            if (sum(self.problem.demands[c] for c in target_route if c != 0) + 
                self.problem.demands[customer] <= self.problem.vehicle_capacity):
                new_pos = random.randint(1, len(target_route) - 1)
                target_route.insert(new_pos, customer)
            else:
                # 容量不足，放回原路径
                source_route.insert(customer_pos, customer)
    
    def two_opt_mutation(self, routes):
        """2-opt变异"""
        route_idx = random.randint(0, len(routes) - 1)
        route = routes[route_idx]
        
        if len(route) > 4:
            i = random.randint(1, len(route) - 3)
            j = random.randint(i + 1, len(route) - 2)
            route[i:j+1] = reversed(route[i:j+1])
    
    def reverse_mutation(self, routes):
        """反转变异"""
        route_idx = random.randint(0, len(routes) - 1)
        route = routes[route_idx]
        
        if len(route) > 3:
            start = random.randint(1, len(route) - 3)
            end = random.randint(start + 1, len(route) - 2)
            route[start:end+1] = reversed(route[start:end+1])
    
    def route_based_crossover(self, parent1, parent2):
        """基于路径的交叉"""
        child_routes = []
        
        # 从父代1选择部分路径
        num_routes_p1 = max(1, len(parent1.routes) // 2)
        child_routes.extend(parent1.routes[:num_routes_p1])
        
        # 从父代2选择不重复的客户
        covered_customers = set()
        for route in child_routes:
            covered_customers.update(route)
            
        for route in parent2.routes:
            new_customers = [c for c in route if c not in covered_customers and c != 0]
            if new_customers:
                # 尝试将新客户插入现有路径
                inserted = False
                for i, r in enumerate(child_routes):
                    current_load = sum(self.problem.demands[c] for c in r if c != 0)
                    additional_load = sum(self.problem.demands[c] for c in new_customers)
                    
                    if current_load + additional_load <= self.problem.vehicle_capacity:
                        # 在返回仓库前插入
                        insert_pos = len(r) - 1
                        r[insert_pos:insert_pos] = new_customers
                        covered_customers.update(new_customers)
                        inserted = True
                        break
                
                if not inserted:
                    # 创建新路径
                    new_route = [0] + new_customers + [0]
                    child_routes.append(new_route)
                    covered_customers.update(new_customers)
                        
        # 处理未覆盖的客户
        all_customers = set(range(1, self.problem.num_customers + 1))
        uncovered = all_customers - covered_customers
        
        if uncovered:
            # 将未覆盖客户分配到新路径
            current_route = [0]
            current_load = 0
            
            for customer in sorted(uncovered):  # 排序以获得确定性行为
                demand = self.problem.demands[customer]
                
                if current_load + demand <= self.problem.vehicle_capacity:
                    current_route.append(customer)
                    current_load += demand
                else:
                    current_route.append(0)
                    child_routes.append(current_route)
                    current_route = [0, customer]
                    current_load = demand
                    
            if len(current_route) > 1:
                current_route.append(0)
                child_routes.append(current_route)
                
        return child_routes
    
    def fast_non_dominated_sort(self, population):
        """快速非支配排序"""
        if not population:
            return []
        
        # 过滤无效解
        valid_population = [ind for ind in population if hasattr(ind, 'objectives') and ind.objectives is not None]
        if not valid_population:
            return []
        
        fronts = [[]]
        
        # 计算支配关系
        for i, p in enumerate(valid_population):
            p.domination_count = 0
            p.dominated_solutions = []
            
            for j, q in enumerate(valid_population):
                if i == j:
                    continue
                if self.dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                fronts[0].append(p)
        
        # 检查第一层前沿是否为空
        if not fronts[0]:
            print("警告: 第一层前沿为空，使用所有有效个体作为第一前沿")
            fronts[0] = valid_population[:]  # 使用所有有效个体
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        next_front.append(q)
            
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts
    
    def dominates(self, individual1, individual2):
        """判断个体1是否支配个体2"""
        try:
            if not hasattr(individual1, 'objectives') or not hasattr(individual2, 'objectives'):
                return False
            
            obj1 = individual1.objectives
            obj2 = individual2.objectives
            
            if obj1 is None or obj2 is None:
                return False
            
            # 检查目标值是否有效
            if any(np.isnan(obj) or np.isinf(obj) for obj in obj1) or \
            any(np.isnan(obj) or np.isinf(obj) for obj in obj2):
                return False
            
            # 所有目标都不差，且至少一个目标更好
            not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
            better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
            
            return not_worse and better
        except Exception as e:
            print(f"支配关系判断错误: {e}")
            return False
    
    def calculate_crowding_distance(self, front):
        """计算拥挤距离"""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
            
        num_objectives = len(front[0].objectives)
        
        for i in range(num_objectives):
            # 按第i个目标排序
            front.sort(key=lambda x: x.objectives[i])
            
            # 边界点的拥挤距离设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # 计算中间点的拥挤距离
            min_obj = front[0].objectives[i]
            max_obj = front[-1].objectives[i]
            
            if abs(max_obj - min_obj) < 1e-10:
                continue
                
            scale = max_obj - min_obj
            for j in range(1, len(front) - 1):
                distance = (front[j+1].objectives[i] - front[j-1].objectives[i]) / scale
                front[j].crowding_distance += distance
    
    def train_neural_network(self, solutions, epoch):
        """训练神经网络"""
        # 如果解集小于一个批量大小，则跳过训练以避免无效的小批量
        if len(solutions) < max(2, int(self.config.get('batch_size', 8))):
            return 0, 0

        self.model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0

        # 准备训练数据（节点特征只需计算一次）
        node_features = self.prepare_node_features()

        # 使用混合精度在支持的 GPU 上加速（使用推荐的 torch.amp API）
        use_amp = (self.device.type == 'cuda') and bool(self.config.get('use_amp', True))
        # 使用新的 API 来创建 GradScaler —— 在非 CUDA 或未启用时显式关闭
        try:
            scaler = torch.amp.GradScaler(enabled=use_amp)
        except AttributeError:
            # 向后兼容旧版本 PyTorch（仍可使用 torch.cuda.amp.GradScaler）
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        batch_size = int(self.config.get('batch_size', 32))
        for i in range(0, len(solutions), batch_size):
            batch_solutions = solutions[i:i+batch_size]
            if len(batch_solutions) < 2:
                continue

            # 准备批量数据
            solution_tensors = torch.stack([sol.to_tensor() for sol in batch_solutions]).to(self.device)
            true_objectives = torch.tensor([sol.objectives for sol in batch_solutions], dtype=torch.float32).to(self.device)
            # 归一化目标值
            true_objectives = (true_objectives - true_objectives.mean(dim=0)) / (true_objectives.std(dim=0) + 1e-8)

            self.optimizer.zero_grad()
            if use_amp:
                # 使用 torch.amp.autocast，并指定 device_type 以匹配当前设备
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    value_estimates = self.model.value_net(node_features.repeat(len(batch_solutions), 1, 1), solution_tensors)
                    value_loss = torch.nn.functional.mse_loss(value_estimates, true_objectives)
                    policy_loss = torch.tensor(0.0).to(self.device)
                    loss = value_loss + 0.1 * policy_loss
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                value_estimates = self.model.value_net(node_features.repeat(len(batch_solutions), 1, 1), solution_tensors)
                value_loss = torch.nn.functional.mse_loss(value_estimates, true_objectives)
                policy_loss = torch.tensor(0.0).to(self.device)
                loss = value_loss + 0.1 * policy_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_policy_loss += float(policy_loss.item()) if hasattr(policy_loss, 'item') else 0.0
            total_value_loss += float(value_loss.item())
            n_batches += 1

        if n_batches > 0:
            avg_policy_loss = total_policy_loss / n_batches
            avg_value_loss = total_value_loss / n_batches
            self.training_history['policy_loss'].append(avg_policy_loss)
            self.training_history['value_loss'].append(avg_value_loss)
            if epoch % 10 == 0:
                print(f"  训练损失 - 策略: {avg_policy_loss:.4f}, 价值: {avg_value_loss:.4f}")
            return avg_policy_loss, avg_value_loss

        return 0, 0
    
    def run_training(self):
        """运行训练 - 简化稳定版本"""
        print("开始GMOEA训练...")
        for generation in range(self.config['max_generations']):
            try:
                # 1. 评估种群
                valid_count = 0
                vehicle_count_stats = {}
                for individual in self.population:
                    try:
                        individual.objectives = individual.evaluate()
                        # 视为有效个体要求目标均为有限，且总距离大于0
                        if all(np.isfinite(obj) for obj in individual.objectives) and individual.objectives[0] > 0:
                            valid_count += 1
                            vnum = int(individual.objectives[2])
                            vehicle_count_stats[vnum] = vehicle_count_stats.get(vnum, 0) + 1
                    except Exception as e:
                        individual.objectives = [float('inf'), float('inf'), float('inf')]
                # 计算平均车辆数与平均总距离以便更直观地了解种群状态
                avg_vehicles = None
                avg_distance = None
                if valid_count > 0:
                    vehicles_list = [int(ind.objectives[2]) for ind in self.population if hasattr(ind, 'objectives') and all(np.isfinite(obj) for obj in ind.objectives) and ind.objectives[0] > 0]
                    distances_list = [ind.objectives[0] for ind in self.population if hasattr(ind, 'objectives') and all(np.isfinite(obj) for obj in ind.objectives) and ind.objectives[0] > 0]
                    if vehicles_list:
                        avg_vehicles = float(np.mean(vehicles_list))
                    if distances_list:
                        avg_distance = float(np.mean(distances_list))

                pct = (valid_count / len(self.population)) * 100 if len(self.population) > 0 else 0
                avg_vehicles_str = f"{avg_vehicles:.2f}" if avg_vehicles is not None else 'N/A'
                avg_distance_str = f"{avg_distance:.1f}" if avg_distance is not None else 'N/A'
                print(f"第{generation}代: {valid_count}/{len(self.population)} ({pct:.1f}%) 有效 | 平均车辆: {avg_vehicles_str} | 平均总距离: {avg_distance_str} | 车辆数分布: {vehicle_count_stats}")

                # 2. 生成子代（交叉 + 变异），数量为 population_size//2，以减少每代开销
                offspring = []
                pop_for_selection = [ind for ind in self.population]
                # 产生较少数量的子代以加快运行（减少模型前向/反向次数）
                for _ in range(max(1, self.population_size // 2)):
                    try:
                        p1, p2 = random.sample(pop_for_selection, 2)
                    except ValueError:
                        p1 = random.choice(pop_for_selection)
                        p2 = random.choice(pop_for_selection)

                    # 交叉
                    if random.random() < float(self.config.get('crossover_rate', 0.8)):
                        child = self.neural_crossover(p1, p2)
                    else:
                        # 复制并轻微变异
                        child = self.neural_mutation(p1)

                    # 变异
                    if random.random() < float(self.config.get('mutation_rate', 0.2)):
                        child = self.neural_mutation(child)

                    # 强制车辆数约束并评估
                    try:
                        child.routes = self.enforce_vehicle_count(child.routes,
                            min_v=self.config.get('min_vehicles', 1),
                            max_v=self.config.get('max_vehicles', self.problem.num_vehicles))
                        child.objectives = child.evaluate()
                        if all(np.isfinite(obj) for obj in child.objectives):
                            offspring.append(child)
                    except Exception:
                        continue

                # 合并父代与子代并进行非支配排序选择
                combined = self.population + offspring
                fronts = self.fast_non_dominated_sort(combined)
                if not fronts:
                    print("没有找到前沿，重新初始化种群")
                    self.population = self.initialize_population()
                    continue

                # 以层次填充下一代，并在需要时按拥挤距离选择
                new_population = []
                for front in fronts:
                    if len(new_population) + len(front) <= self.population_size:
                        new_population.extend(front)
                    else:
                        # 计算并使用拥挤距离选择剩余名额
                        self.calculate_crowding_distance(front)
                        front_sorted = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                        need = self.population_size - len(new_population)
                        new_population.extend(front_sorted[:need])
                        break

                self.population = new_population

                # 5. 每若干代训练神经网络（填充 training_history）
                if generation % self.config.get('train_interval', 10) == 0:
                    try:
                        train_set = self.population
                        policy_loss, value_loss = self.train_neural_network(train_set, generation)
                        if policy_loss or value_loss:
                            print(f"  训练更新 - 策略损失: {policy_loss:.4f}, 价值损失: {value_loss:.4f}")
                    except Exception as e:
                        print(f"训练网络时发生错误: {e}")

                # 6. 输出前沿统计
                if generation % 10 == 0 and fronts[0]:
                    objectives = np.array([sol.objectives for sol in fronts[0]])
                    avg_obj = np.mean(objectives, axis=0)
                    front_vehicle_stats = {}
                    for sol in fronts[0]:
                        vnum = int(sol.objectives[2])
                        front_vehicle_stats[vnum] = front_vehicle_stats.get(vnum, 0) + 1
                    print(f"  前沿大小: {len(fronts[0])}, 平均目标: [{avg_obj[0]:.1f}, {avg_obj[1]:.1f}, {avg_obj[2]:.1f}], 前沿车辆数分布: {front_vehicle_stats}")
            except Exception as e:
                print(f"第{generation}代发生错误: {e}")
                self.population = self.initialize_population()
        # 返回最终结果
        final_population = []
        for ind in self.population:
            try:
                ind.objectives = ind.evaluate()
                if all(np.isfinite(obj) for obj in ind.objectives):
                    final_population.append(ind)
            except:
                continue
        final_fronts = self.fast_non_dominated_sort(final_population)
        pareto_front = final_fronts[0] if final_fronts else []
        print(f"训练完成! 找到 {len(pareto_front)} 个Pareto解")
        # 保存模型
        model_path = "saved_models/gmoea_model_final.pth"
        save_model(self.model, model_path)
        return pareto_front, self.population, self.training_history