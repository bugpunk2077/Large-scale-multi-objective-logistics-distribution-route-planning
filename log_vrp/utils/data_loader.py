import numpy as np
import os
import re


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
            raw = [l.rstrip('\n') for l in f]
        # 移除空白行但保留原始以便格式检测
        lines = [l for l in raw if l.strip()]

        if not lines:
            raise ValueError('文件为空')

        # 若文件为经典 VRPTW 格式（含关键词 VEHICLE / CUSTOMER），则使用专门解析器
        up = '\n'.join(raw).upper()
        if 'VEHICLE' in up and 'CUSTOMER' in up:
            data_lines = self._parse_vrptw(raw)
        else:
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

    def _parse_vrptw(self, raw_lines):
        """解析经典 VRPTW 文件格式，返回仅含节点数据行的列表（用制表符分隔或空格分隔的字段）。

        支持的段落：VEHICLE 部分包含 NUMBER 和 CAPACITY；
        CUSTOMER 段包含表头，然后每行为: id x y demand ready_time due_date service_time
        返回值：节点行的字符串列表（字段用 '\t' 分隔以兼容后续逻辑）
        """
        lines = [l.strip() for l in raw_lines if l.strip()]
        num_vehicles = None
        vehicle_capacity = None
        cust_lines = []
        mode = None
        for ln in lines:
            up = ln.upper()
            if up.startswith('VEHICLE'):
                mode = 'VEHICLE'
                continue
            if up.startswith('CUSTOMER'):
                mode = 'CUSTOMER'
                continue
            # VEHICLE 段可能包含字段名或直接数字行
            if mode == 'VEHICLE':
                # 尝试从行中提取两个数字：K Q
                parts = ln.replace('\t', ' ').split()
                nums = [p for p in parts if all(ch.isdigit() or ch in '.-+' for ch in p)]
                if len(nums) >= 2:
                    try:
                        if num_vehicles is None:
                            num_vehicles = int(float(nums[0]))
                        if vehicle_capacity is None:
                            vehicle_capacity = float(nums[1])
                    except Exception:
                        pass
                continue
            if mode == 'CUSTOMER':
                # 忽略可能的 header 行（包含字母）
                if any(c.isalpha() for c in ln) and not ln[0].isdigit():
                    continue
                # 将空白/多个空格或制表符统一为单个制表符，返回给上层解析
                parts = ln.replace('\t', ' ').split()
                if len(parts) >= 7:
                    # 取前 7 字段
                    row = '\t'.join(parts[:7])
                    cust_lines.append(row)
                else:
                    # 忽略非数据行
                    continue

        if num_vehicles is not None:
            try:
                self.num_vehicles = int(num_vehicles)
            except Exception:
                pass
        if vehicle_capacity is not None:
            try:
                self.vehicle_capacity = float(vehicle_capacity)
            except Exception:
                pass

        return cust_lines

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


def load_ground_truth(path_or_dir):
    """加载 .sol 最优解文件或目录。

    返回字典：{base_name: routes_list}
    每个 routes_list 是 [[0, a, b, 0], [0, c, d, 0], ...]
    兼容单文件和目录输入；在无法解析时返回 {}。
    """
    sols = {}
    if not path_or_dir:
        return sols
    files = []
    if os.path.isdir(path_or_dir):
        for fn in os.listdir(path_or_dir):
            if fn.lower().endswith('.sol'):
                files.append(os.path.join(path_or_dir, fn))
    elif os.path.isfile(path_or_dir):
        files = [path_or_dir]
    else:
        # 如果传入的是路径前缀（如 data/base），尝试附加 .sol
        if os.path.exists(path_or_dir + '.sol'):
            files = [path_or_dir + '.sol']
        else:
            return sols

    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except Exception:
            continue

        routes = []
        # 每行尝试解析一条路线中的整数序列
        for ln in lines:
            nums = re.findall(r"-?\d+", ln)
            if not nums:
                continue
            nums = [int(x) for x in nums]
            # 保留非负索引
            nums = [n for n in nums if n >= 0]
            if not nums:
                continue
            # 确保以 0 起止
            if nums[0] != 0:
                nums = [0] + nums
            if nums[-1] != 0:
                nums = nums + [0]
            # 忽略仅 depot 的线路
            if len(nums) <= 2:
                continue
            routes.append(nums)

        # 如果逐行未能解析到多条路线，尝试把文件中所有数字作为一条序列并切分
        if not routes:
            all_nums = []
            for ln in lines:
                parts = re.findall(r"\d+", ln)
                all_nums.extend(int(p) for p in parts)
            if all_nums:
                if all_nums[0] != 0:
                    all_nums = [0] + all_nums
                if all_nums[-1] != 0:
                    all_nums = all_nums + [0]
                cur = []
                for n in all_nums:
                    if n == 0:
                        if cur:
                            routes.append([0] + cur + [0])
                            cur = []
                    else:
                        cur.append(n)

        base = os.path.splitext(os.path.basename(fp))[0]
        sols[base] = routes

    return sols


def compute_kmeans(coordinates, k=4, max_iter=100, tol=1e-4, random_state=None):
    """简单的 K-means 实现（只依赖 numpy），用于把节点坐标聚类。

    参数:
      coordinates: iterable of (x,y) tuples
      k: 聚类数
      max_iter: 最大迭代次数
      tol: 收敛容忍度（质心移动小于 tol 时停止）
      random_state: 可选 int（用于可重复初始化）

    返回: (labels, centers)
      labels: numpy array shape (n,) 每个点的簇索引
      centers: numpy array shape (k,2)
    """
    import numpy as _np

    pts = _np.array(coordinates, dtype=float)
    n = pts.shape[0]
    if n == 0:
        return _np.array([]), _np.zeros((0, 2))

    if k <= 0:
        raise ValueError('k must be > 0')

    rng = _np.random.RandomState(random_state)
    # 随机选择初始质心
    init_idx = rng.choice(n, min(k, n), replace=False)
    centers = pts[init_idx].astype(float)

    labels = _np.zeros(n, dtype=int)
    for it in range(int(max_iter)):
        # 分配步骤
        dists = _np.sqrt(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))  # (n,k)
        new_labels = _np.argmin(dists, axis=1)

        # 更新步骤
        new_centers = _np.zeros_like(centers)
        for j in range(centers.shape[0]):
            members = pts[new_labels == j]
            if len(members) == 0:
                # 若某簇为空，重新随机选一个点作为中心
                new_centers[j] = pts[rng.randint(0, n)]
            else:
                new_centers[j] = members.mean(axis=0)

        shift = _np.sqrt(((new_centers - centers) ** 2).sum(axis=1)).max()
        centers = new_centers
        labels = new_labels
        if shift <= tol:
            break

    return labels, centers