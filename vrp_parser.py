from data_structures import Depot, PickupPoint

def parse_vrp_file(filename: str):
    """解析VRP文件，返回仓库和取货点信息"""
    depot = None
    pickup_points = {}
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # 解析坐标部分
    coord_section = False
    demand_section = False
    depot_section = False
    
    coords = {}
    demands = {}
    depot_id = None
    
    for line in lines:
        line = line.strip()
        
        if line == "NODE_COORD_SECTION":
            coord_section = True
            demand_section = False
            depot_section = False
            continue
        elif line == "DEMAND_SECTION":
            coord_section = False
            demand_section = True
            depot_section = False
            continue
        elif line == "DEPOT_SECTION":
            coord_section = False
            demand_section = False
            depot_section = True
            continue
        elif line == "EOF" or line == "-1":
            break
        
        if coord_section and line:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[node_id] = (x, y)
        
        elif demand_section and line:
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = int(parts[1])
                demands[node_id] = demand
        
        elif depot_section and line and line != "-1":
            depot_id = int(line)
    
    # 创建仓库对象
    if depot_id and depot_id in coords:
        depot_x, depot_y = coords[depot_id]
        depot = Depot(depot_x, depot_y)
    
    # 创建取货点对象
    for node_id in coords:
        if node_id != depot_id:  # 不是仓库的节点都是取货点
            x, y = coords[node_id]
            demand = demands.get(node_id, 0)
            if demand > 0:  # 只有有需求的点才创建取货点
                pickup_points[node_id] = PickupPoint(node_id, x, y, demand)
    
    return depot, pickup_points

def print_problem_info(depot: Depot, pickup_points: dict):
    """打印问题基本信息"""
    print(f"仓库位置: ({depot.x_coord}, {depot.y_coord})")
    print(f"取货点数量: {len(pickup_points)}")
    print(f"总需求量: {sum(p.initial_demand for p in pickup_points.values())}")
    print("\n取货点信息:")
    for point_id, point in pickup_points.items():
        print(f"  点 {point_id}: 位置({point.x_coord}, {point.y_coord}), 需求={point.initial_demand}")

if __name__ == "__main__":
    depot, pickup_points = parse_vrp_file("real.vrp")
    print_problem_info(depot, pickup_points) 