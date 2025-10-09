import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def manhattan_distance(x1, y1, x2, y2):
    """计算曼哈顿距离"""
    return abs(x1 - x2) + abs(y1 - y2)

def read_vrp_data(filename):
    """读取VRP数据文件"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # 解析NODE_COORD_SECTION
    lines = content.split('\n')
    node_coords = {}
    demand_section = {}
    depot = None
    
    in_coord_section = False
    in_demand_section = False
    in_depot_section = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('NODE_COORD_SECTION'):
            in_coord_section = True
            continue
        elif line.startswith('DEMAND_SECTION'):
            in_coord_section = False
            in_demand_section = True
            continue
        elif line.startswith('DEPOT_SECTION'):
            in_demand_section = False
            in_depot_section = True
            continue
        elif line.startswith('EOF'):
            break
        
        if in_coord_section and line and not line.startswith('NAME') and not line.startswith('COMMENT'):
            parts = line.split()
            if len(parts) == 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords[node_id] = (x, y)
        
        elif in_demand_section and line and not line.startswith('NAME') and not line.startswith('COMMENT'):
            parts = line.split()
            if len(parts) == 2:
                node_id = int(parts[0])
                demand = int(parts[1])
                demand_section[node_id] = demand
        
        elif in_depot_section and line and not line.startswith('-1'):
            depot = int(line)
    
    return node_coords, demand_section, depot

def visualize_solution(vrp_file, solution_file):
    """Visualize the solution"""
    # Read VRP data
    node_coords, demand_section, depot = read_vrp_data(vrp_file)
    
    # Read solution data
    with open(solution_file, 'r') as f:
        solution_data = json.load(f)
    
    # Create figure - using 1:1 ratio and academic style
    plt.figure(figsize=(12, 12), dpi=300)  # 1:1 ratio
    plt.style.use('default')
    
    # Analyze transportation mode for each node
    truck_nodes = set()
    drone_nodes = set()
    
    # Collect nodes served by trucks
    for truck_route in solution_data['truck_routes']:
        sequence = truck_route['sequence']
        for node in sequence:
            if node != 'depot' and node != depot:
                truck_nodes.add(node)
    
    # Collect nodes served by drones
    for drone_schedule in solution_data['drone_schedules']:
        for trip in drone_schedule['trips']:
            sequence = trip['sequence']
            for node in sequence:
                if node != 'depot' and node != depot:
                    drone_nodes.add(node)
    
    # Find nodes served by both
    common_nodes = truck_nodes.intersection(drone_nodes)
    truck_only_nodes = truck_nodes - drone_nodes
    drone_only_nodes = drone_nodes - truck_nodes
    
    # Calculate node size (based on demand)
    min_demand = min(demand_section.values())
    max_demand = max(demand_section.values())
    min_size = 150
    max_size = 600
    
    # Draw all nodes
    for node_id, (x, y) in node_coords.items():
        if node_id == depot:
            # Depot node in red
            plt.scatter(x, y, c='red', s=400, zorder=6, label='Depot' if node_id == depot else "")
            plt.text(x, y + 1, f'Depot', fontsize=10, ha='center', va='bottom', fontweight='bold')
        else:
            # Determine node properties based on transportation mode and demand
            demand = demand_section.get(node_id, 0)
            # Calculate node size (linear scaling)
            node_size = min_size + (demand - min_demand) / (max_demand - min_demand) * (max_size - min_size)
            
            # Determine color and label
            if node_id in common_nodes:
                color = '#CCEFFC'  # Coordinated delivery - using truck color
                label = 'Truck & Drone' if node_id == list(common_nodes)[0] else ""
                zorder = 5
                # Coordinated delivery nodes with dashed border
                plt.scatter(x, y, c=color, s=node_size, zorder=zorder, 
                           edgecolors='#6666FF', linewidth=2, linestyle='--')
            elif node_id in truck_only_nodes:
                color = '#CCEFFC'   # Truck only - light blue
                label = 'Truck Only' if node_id == list(truck_only_nodes)[0] else ""
                zorder = 4
                plt.scatter(x, y, c=color, s=node_size, zorder=zorder, 
                           edgecolors='black', linewidth=0.5)
            elif node_id in drone_only_nodes:
                color = '#BFFFBF'  # Drone only - light green
                label = 'Drone Only' if node_id == list(drone_only_nodes)[0] else ""
                zorder = 4
                plt.scatter(x, y, c=color, s=node_size, zorder=zorder, 
                           edgecolors='black', linewidth=0.5)
            else:
                color = 'gray'    # Not served - gray
                label = 'Not Served' if 'Not Served' not in [str(l) for l in locals().values()] else ""
                zorder = 3
                plt.scatter(x, y, c=color, s=node_size, zorder=zorder, 
                           edgecolors='black', linewidth=0.5)
            plt.text(x, y, f'\n\n{node_id}\n({demand})', fontsize=8, ha='center', va='top', fontweight='bold')
    
    # Define colors for different route types
    truck_colors = ['green', 'orange', 'purple', 'brown']
    drone_colors = ['magenta', 'magenta']
    
    # Draw truck routes (using Manhattan distance paths)
    for i, truck_route in enumerate(solution_data['truck_routes']):
        color = truck_colors[i % len(truck_colors)]
        sequence = truck_route['sequence']
        
        # Draw routes
        for j in range(len(sequence) - 1):
            current = sequence[j]
            next_node = sequence[j + 1]
            
            if current == 'depot':
                current = depot
            if next_node == 'depot':
                next_node = depot
                
            x1, y1 = node_coords[current]
            x2, y2 = node_coords[next_node]
            
            # Use Manhattan distance path (horizontal line + vertical line)
            plt.plot([x1, x2], [y1, y1], color=color, linestyle='-', linewidth=2, alpha=0.7)  # Horizontal segment
            plt.plot([x2, x2], [y1, y2], color=color, linestyle='-', linewidth=2, alpha=0.7)  # Vertical segment
    
    # Draw drone routes
    for i, drone_schedule in enumerate(solution_data['drone_schedules']):
        color = drone_colors[i % len(drone_colors)]
        drone_id = drone_schedule['drone_id']
        
        for trip in drone_schedule['trips']:
            sequence = trip['sequence']
            
            # Draw drone routes (from depot to customer point and back to depot)
            for j in range(len(sequence) - 1):
                current = sequence[j]
                next_node = sequence[j + 1]
                
                if current == 'depot':
                    current = depot
                if next_node == 'depot':
                    next_node = depot
                    
                x1, y1 = node_coords[current]
                x2, y2 = node_coords[next_node]
                
                plt.plot([x1, x2], [y1, y2], color=color, linestyle='--', linewidth=1, alpha=0.6)
    
    # Set equal aspect ratio to ensure circles display correctly
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Draw dashed circle centered at depot, representing drone range
    depot_x, depot_y = node_coords[depot]
    circle = plt.Circle((depot_x, depot_y), 15, fill=False, linestyle='--', 
                       linewidth=2, color='gray', alpha=0.8)
    plt.gca().add_patch(circle)
    
    # Add drone range annotation
    plt.text(depot_x + 15, depot_y - 2, 'Drone Range', fontsize=10, 
             ha='left', va='center', color='gray', fontweight='bold')
    
    # Academic title and labels - add node size description
    plt.xlabel('X Coordinate (km)', fontsize=12, fontweight='bold')
    plt.ylabel('Y Coordinate (km)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    
    # 设置坐标轴范围和刻度
    x_coords = [coord[0] for coord in node_coords.values()]
    y_coords = [coord[1] for coord in node_coords.values()]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 添加适当的边距
    margin = 5
    plt.xlim(x_min - margin, x_max + margin)
    plt.ylim(y_min - margin, y_max + margin)
    
    # 设置刻度字体
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    # 保存图像到文件
    plt.savefig('solution_visualization.png', dpi=300, bbox_inches='tight')
    print("路线图已保存为 'solution_visualization.png'")
    
    plt.show()

if __name__ == "__main__":
    vrp_file = "real_1.vrp"
    solution_file = "solution_data_20250928_105444.json"
    visualize_solution(vrp_file, solution_file)