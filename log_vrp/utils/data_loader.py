import numpy as np
import os


class VRPProblem:
    """轻量、鲁棒的 VRP 问题解析器（按制表符分隔）

    约定：文件每行字段以制表符（\t）分隔。数据行为：
      id\tx\ty\tdemand\tearliest\tlatest\tservice\t[pickup]\t[delivery]
    第一行可选包含：num_vehicles\tvehicle_capacity\tvehicle_speed
    """

    def __init__(self, file_path=None):
        # 默认参数
        self.vehicle_capacity = 20
        self.num_vehicles = 25
        self.vehicle_speed = 1.0

        # 数据容器（index 对应文件行的节点索引）
        self.coordinates = []
        self.demands = []
        self.time_windows = []
        self.service_times = []
        self.pickup_indices = []
        self.delivery_indices = []

        if file_path and os.path.exists(file_path):
            try:
                self._load_tab_separated(file_path)
            except Exception as e:
                print(f"加载 VRP 文件失败: {e}\n将使用示例数据替代。")
                self.generate_example_data()
        else:
            self.generate_example_data()

    def _load_tab_separated(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.rstrip('\n') for l in f if l.strip()]

        if not lines:
            raise ValueError('文件为空')

        # 尝试解析首行配置（如果为数字项）
        first_tokens = lines[0].split('\t')
        if len(first_tokens) >= 3:
            try:
                nv = int(first_tokens[0])
                cap = float(first_tokens[1])
                spd = float(first_tokens[2])
                # 仅在合理范围内才替换默认值
                if nv > 0:
                    self.num_vehicles = nv
                if cap > 0:
                    self.vehicle_capacity = cap
                if spd > 0:
                    self.vehicle_speed = spd
                # 把首行当配置行解析后，从第二行开始解析节点
                data_lines = lines[1:]
            except Exception:
                # 如果解析失败，则把所有行视为数据行
                data_lines = lines
        else:
            data_lines = lines

        # 解析数据行：id x y demand earliest latest service [pickup] [delivery]
        self.coordinates = []
        self.demands = []
        self.time_windows = []
        self.service_times = []
        self.pickup_indices = []
        self.delivery_indices = []

        for ln in data_lines:
            tokens = ln.split('\t')
            if len(tokens) < 3:
                # 忽略不完整行
                continue
            try:
                idx = int(tokens[0])  # keep but not strictly used
                x = float(tokens[1])
                y = float(tokens[2])
            except Exception:
                # 忽略无法解析的行
                continue

            # 默认字段
            demand = 0.0
            earliest, latest = 0.0, 1e9
            service = 0.0
            pickup = -1
            delivery = -1

            if len(tokens) >= 4 and tokens[3] != '':
                try:
                    demand = float(tokens[3])
                except Exception:
                    demand = 0.0
            if len(tokens) >= 6:
                try:
                    earliest = float(tokens[4])
                    latest = float(tokens[5])
                except Exception:
                    earliest, latest = 0.0, 1e9
            if len(tokens) >= 7:
                try:
                    service = float(tokens[6])
                except Exception:
                    service = 0.0
            if len(tokens) >= 8:
                try:
                    pickup = int(tokens[7])
                except Exception:
                    pickup = -1
            if len(tokens) >= 9:
                try:
                    delivery = int(tokens[8])
                except Exception:
                    delivery = -1

            self.coordinates.append((x, y))
            self.demands.append(demand)
            self.time_windows.append((earliest, latest))
            self.service_times.append(service)
            self.pickup_indices.append(pickup)
            self.delivery_indices.append(delivery)

        # 标准化索引：第一行应为仓库（如果未提供仓库则生成默认仓库）
        if len(self.coordinates) == 0:
            raise ValueError('没有有效节点')

        # 如果文件没有显式仓库（demands[0] != 0），不会强制要求，但常见仓库为 index 0
        self.num_customers = max(0, len(self.coordinates) - 1)
        self.distance_matrix = self.calculate_distance_matrix()
        print(f"已加载 VRP: nodes={len(self.coordinates)} customers={self.num_customers} vehicles_limit={self.num_vehicles} capacity={self.vehicle_capacity}")

    def generate_example_data(self):
        print('使用示例数据 (PDPTW)')
        self.num_customers = 100
        self.vehicle_capacity = 200
        self.num_vehicles = 25
        self.vehicle_speed = 1.0

        np.random.seed(42)
        self.coordinates = [(40, 50)]
        self.demands = [0]
        self.time_windows = [(0, 1236)]
        self.service_times = [0]
        self.pickup_indices = [-1]
        self.delivery_indices = [-1]

        pairs_count = self.num_customers // 2
        for i in range(pairs_count):
            x1 = np.random.uniform(35, 45)
            y1 = np.random.uniform(50, 70)
            demand = np.random.randint(5, 25)
            ready_time = np.random.randint(0, 600)
            due_date = ready_time + np.random.randint(100, 300)
            service_time = np.random.randint(10, 50)

            self.coordinates.append((x1, y1))
            self.demands.append(demand)
            self.time_windows.append((ready_time, due_date))
            self.service_times.append(service_time)
            self.pickup_indices.append(-1)
            self.delivery_indices.append(-1)

            x2 = x1 + np.random.uniform(-5, 5)
            y2 = y1 + np.random.uniform(-5, 5)
            x2 = np.clip(x2, 30, 50)
            y2 = np.clip(y2, 45, 75)

            self.coordinates.append((x2, y2))
            self.demands.append(-demand)
            self.time_windows.append((ready_time + 50, due_date + 100))
            self.service_times.append(service_time)
            self.pickup_indices.append(len(self.coordinates) - 2)
            self.delivery_indices.append(-1)

        self.distance_matrix = self.calculate_distance_matrix()
        print(f"生成示例数据: nodes={len(self.coordinates)} customers={self.num_customers}")

    def calculate_distance_matrix(self):
        n = len(self.coordinates)
        mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            xi, yi = self.coordinates[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = self.coordinates[j]
                mat[i, j] = float(((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5)
        return mat


def load_vrp_instance(file_path):
    return VRPProblem(file_path)