import numpy as np
import os
import re

class VRPProblem:
    """VRP问题类 - 解析lc101格式"""
    
    def __init__(self, file_path=None):
        # 设置默认值
        self.vehicle_capacity = 20  # 默认容量
        self.num_vehicles = 25       # 默认车辆数
        self.coordinates = []
        self.demands = []
        self.time_windows = []
        self.service_times = []
        
        if file_path and os.path.exists(file_path):
            self.load_lc101_file(file_path)
        else:
            self.generate_example_data()
    
    def load_lc101_file(self, file_path):
        """解析lc101格式文件"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            print(f"正在解析 Solomon/LR/LRC 文件（更强容错）: {file_path}")

            # 先尝试从文件中寻找关键字（CAPACITY/VEHICLE/VEHICLES）来提取容量或车辆数
            full_text = '\n'.join(lines)
            cap_match = re.search(r'CAPACITY\D*(\d+)', full_text, flags=re.IGNORECASE)
            if cap_match:
                try:
                    self.vehicle_capacity = int(cap_match.group(1))
                    print(f"从文件中解析到车辆容量: {self.vehicle_capacity}")
                except Exception:
                    pass

            veh_match = re.search(r'VEHICLE[S]?\D*(\d+)', full_text, flags=re.IGNORECASE)
            if veh_match:
                try:
                    self.num_vehicles = int(veh_match.group(1))
                    print(f"从文件中解析到车辆数量: {self.num_vehicles}")
                except Exception:
                    pass

            # 查找数据段：使用正则定位首个看起来像数据行的索引（行首为编号，后跟坐标）
            data_start = None
            data_line_pattern = re.compile(r'^\s*(\d+)\s+[-+]?\d*\.?\d+')
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                # 支持逗号或空白分隔
                tokens = re.split(r'[\s,]+', line.strip())
                # 如果第一列是序号并且后面至少有两个浮点数（x,y），认为是数据行
                if len(tokens) >= 3:
                    try:
                        int(tokens[0])
                        float(tokens[1])
                        float(tokens[2])
                        data_start = i
                        break
                    except Exception:
                        continue

            if data_start is None:
                raise ValueError("无法定位数据段，请检查文件格式 (支持 lc/lr/lrc 变体)")

            # 解析数据行，兼容不同字段数量 (至少包含 id, x, y, demand, ready, due, service 可选)
            self.coordinates = []
            self.demands = []
            self.time_windows = []
            self.service_times = []

            nodes_parsed = 0
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                tokens = re.split(r'[\s,]+', line)
                # 跳过非数据行
                if len(tokens) < 3:
                    continue

                # 尝试解析常见列
                try:
                    cust_no = int(tokens[0])
                    x = float(tokens[1])
                    y = float(tokens[2])
                except Exception:
                    # 一旦遇到不符合格式的行，认为数据段结束
                    break

                # 默认值
                demand = 0.0
                ready_time = 0.0
                due_date = 0.0
                service_time = 0.0

                # 根据列数尝试提取更多字段
                if len(tokens) >= 4:
                    try:
                        demand = float(tokens[3])
                    except Exception:
                        demand = 0.0
                if len(tokens) >= 6:
                    try:
                        ready_time = float(tokens[4])
                        due_date = float(tokens[5])
                    except Exception:
                        ready_time = 0.0
                        due_date = 0.0
                if len(tokens) >= 7:
                    try:
                        service_time = float(tokens[6])
                    except Exception:
                        service_time = 0.0

                self.coordinates.append((x, y))
                self.demands.append(demand)
                self.time_windows.append((ready_time, due_date))
                self.service_times.append(service_time)
                nodes_parsed += 1

            # 根据解析结果设置 num_customers（减去仓库）
            self.num_customers = max(0, nodes_parsed - 1)

            # 计算距离矩阵
            self.distance_matrix = self.calculate_distance_matrix()

            print(f"成功加载: {self.num_customers} 个客户点 (共 {nodes_parsed} 行包含仓库)")
            print(f"车辆容量: {self.vehicle_capacity}")
            print(f"车辆数量(上限): {self.num_vehicles}")
            if self.time_windows:
                print(f"仓库时间窗(若有): {self.time_windows[0]}")
            
        except Exception as e:
            print(f"加载文件失败: {e}")
            print("使用示例数据...")
            self.generate_example_data()
    
    def generate_example_data(self):
        """生成示例数据"""
        print("使用示例数据...")
        self.num_customers = 100
        self.vehicle_capacity = 200
        self.num_vehicles = 25
        
        np.random.seed(42)
        self.coordinates = [(40, 50)]  # 仓库 (lc101中的仓库坐标)
        self.demands = [0]
        self.time_windows = [(0, 1236)]  # 仓库时间窗
        self.service_times = [0]
        
        for i in range(self.num_customers):
            x = np.random.uniform(35, 45)
            y = np.random.uniform(50, 70)
            demand = np.random.randint(1, 30)
            ready_time = np.random.randint(0, 800)
            due_date = ready_time + np.random.randint(100, 300)
            service_time = np.random.randint(10, 100)
            
            self.coordinates.append((x, y))
            self.demands.append(demand)
            self.time_windows.append((ready_time, due_date))
            self.service_times.append(service_time)
        
        self.distance_matrix = self.calculate_distance_matrix()
        print(f"生成示例数据: {self.num_customers}个客户点")
    
    def calculate_distance_matrix(self):
        """计算距离矩阵"""
        n = len(self.coordinates)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.coordinates[i]
                    x2, y2 = self.coordinates[j]
                    dist_matrix[i][j] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return dist_matrix

def load_vrp_instance(file_path):
    """加载VRP实例"""
    return VRPProblem(file_path)