import os
from utils.data_loader import VRPProblem

def debug_lc101_file():
    """调试lc101文件解析"""
    file_path = "data/lc101.txt"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    print("=== 调试 lc101.txt 文件解析 ===")
    
    # 首先查看文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print(f"文件总行数: {len(lines)}")
    print("\n文件前20行内容:")
    for i, line in enumerate(lines[:20]):
        print(f"{i+1:3d}: {line.rstrip()}")
    
    print("\n=== 解析结果 ===")
    try:
        problem = VRPProblem(file_path)
        
        print(f"客户点数量: {problem.num_customers}")
        print(f"车辆容量: {problem.vehicle_capacity}")
        print(f"车辆数量: {problem.num_vehicles}")
        print(f"坐标数量: {len(problem.coordinates)}")
        print(f"需求数量: {len(problem.demands)}")
        print(f"时间窗数量: {len(problem.time_windows)}")
        print(f"服务时间数量: {len(problem.service_times)}")
        
        if problem.coordinates:
            print(f"\n仓库坐标: {problem.coordinates[0]}")
            print(f"仓库需求: {problem.demands[0]}")
            if problem.time_windows:
                print(f"仓库时间窗: {problem.time_windows[0]}")
        
        if len(problem.coordinates) > 1:
            print(f"\n第一个客户点:")
            print(f"  坐标: {problem.coordinates[1]}")
            print(f"  需求: {problem.demands[1]}")
            if problem.time_windows:
                print(f"  时间窗: {problem.time_windows[1]}")
            print(f"  服务时间: {problem.service_times[1]}")
            
    except Exception as e:
        print(f"解析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lc101_file()