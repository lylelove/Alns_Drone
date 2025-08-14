import math
import random
from typing import List, Tuple
from data_structures import *
import config

class InitialSolutionBuilder:
    """初始解构建器"""
    
    def __init__(self, depot: Depot, pickup_points: dict):
        self.depot = depot
        self.pickup_points = pickup_points
        
    def build_initial_solution(self) -> Solution:
        """构建初始解决方案"""
        # 重置所有取货点的分配
        for point in self.pickup_points.values():
            point.reset_assignments()
            
        
        solution = Solution()
        
        # 1. 按需求量降序排列取货点
        sorted_points = sorted(self.pickup_points.values(), 
                              key=lambda p: p.initial_demand, reverse=True)
        
        # 2. 卡车主导分配
        truck_id = 1
        for point in sorted_points:
            remaining = point.initial_demand - point.assigned_truck_load
            if remaining > 0:
                # 检查是否可以插入现有路线
                best_route = None
                best_insertion_cost = float('inf')
                best_position = -1
                best_load = 0
                
                for route in solution.truck_routes:
                    if not route.sequence_of_points:  # 空路线
                        continue
                    insertion_result = self._find_best_truck_insertion(route, point)
                    if insertion_result and insertion_result[1] < best_insertion_cost:
                        best_route = route
                        best_insertion_cost = insertion_result[1]
                        best_position = insertion_result[0]
                        best_load = insertion_result[2]
                
                if best_route and best_load > 0:
                    # 插入到最佳位置
                    self._insert_into_truck_route(best_route, point.id, best_position, best_load)
                    point.assigned_truck_load += best_load
                else:
                    # 创建新的卡车路线
                    new_route = TruckRoute(truck_id)
                    load_to_take = min(remaining, config.TRUCK_CAPACITY)
                    new_route.sequence_of_points = ['depot', point.id, 'depot']
                    new_route.visited_points_and_loads = [(point.id, load_to_take)]
                    new_route.calculate_metrics(self.depot, self.pickup_points)
                    
                    solution.truck_routes.append(new_route)
                    point.assigned_truck_load += load_to_take
                    truck_id += 1
        
        # 3. 无人机辅助分配
        self._assign_drones_to_remaining_demand(solution)
        
        # 4. 确保所有需求都被满足
        self._ensure_all_demands_satisfied(solution)
        
        # 5. 计算解决方案指标
        solution.calculate_solution_metrics()
        
        return solution
    
    def _find_best_truck_insertion(self, route: TruckRoute, point: PickupPoint) -> Tuple[int, float, int]:
        """找到插入取货点的最佳位置"""
        if not route.sequence_of_points or len(route.sequence_of_points) < 2:
            return None
        
        best_position = -1
        best_cost_increase = float('inf')
        best_load = 0
        
        # 计算当前路线已经运载的货物
        current_load = sum(load for _, load in route.visited_points_and_loads)
        remaining_capacity = config.TRUCK_CAPACITY - current_load
        
        if remaining_capacity <= 0:
            return None
        
        # 可以运载的货物量
        remaining_demand = point.initial_demand - point.assigned_truck_load
        load_to_take = min(remaining_demand, remaining_capacity)
        if load_to_take <= 0:
            return None
        
        # 尝试每个可能的插入位置
        for i in range(1, len(route.sequence_of_points)):
            if route.sequence_of_points[i] == 'depot':
                # 在返回仓库前插入
                prev_point = route.sequence_of_points[i-1]
                
                # 计算原始成本
                if prev_point == 'depot':
                    prev_x, prev_y = self.depot.x_coord, self.depot.y_coord
                else:
                    prev_x, prev_y = self.pickup_points[prev_point].x_coord, self.pickup_points[prev_point].y_coord
                
                depot_x, depot_y = self.depot.x_coord, self.depot.y_coord
                original_dist = abs(prev_x - depot_x) + abs(prev_y - depot_y)
                
                # 计算新成本
                point_x, point_y = point.x_coord, point.y_coord
                new_dist1 = abs(prev_x - point_x) + abs(prev_y - point_y)
                new_dist2 = abs(point_x - depot_x) + abs(point_y - depot_y)
                
                # 成本增加计算
                # 从前一点到新点：运载现有货物的成本
                cost1 = new_dist1 *  config.TRUCK_COST_PER_KM
                # 从新点到depot：运载现有货物+新货物的成本，减去原来的成本
                cost2 = new_dist2 *  config.TRUCK_COST_PER_KM
                original_cost = original_dist *  config.TRUCK_COST_PER_KM
                
                cost_increase = cost1 + cost2 - original_cost
                
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_position = i
                    best_load = load_to_take
        
        if best_position >= 0:
            return (best_position, best_cost_increase, best_load)
        return None
    
    def _insert_into_truck_route(self, route: TruckRoute, point_id: int, position: int, load: int):
        """将取货点插入到卡车路线中"""
        route.sequence_of_points.insert(position, point_id)

        # ``visited_points_and_loads`` 需要与 ``sequence_of_points`` 保持顺序一致，
        # 否则后续基于索引的操作（例如交换、移除等）会出现错位。
        # ``sequence_of_points`` 包含了起点和终点的仓库节点，而 ``visited_points_and_loads``
        # 仅记录实际的取货点，因此其插入位置比 ``sequence_of_points`` 对应位置少 1。
        insert_idx = max(0, position - 1)
        route.visited_points_and_loads.insert(insert_idx, (point_id, load))

        route.calculate_metrics(self.depot, self.pickup_points)
    
    def _assign_drones_to_remaining_demand(self, solution: Solution):
        """为剩余需求分配无人机"""
        # 初始化无人机调度表
        for drone_id in range(1, config.MAX_DRONES + 1):
            solution.drone_fleet_schedules.append(DroneFleetSchedule(drone_id))
        
        # 找到所有有剩余需求且在无人机半径内的点
        remaining_points = []
        for point in self.pickup_points.values():
            remaining_demand = point.initial_demand - point.assigned_truck_load
            if remaining_demand > 0:
                distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 + 
                                            (point.y_coord - self.depot.y_coord)**2)
                if (distance_to_depot <= config.DRONE_RADIUS and 
                    distance_to_depot * 2 <= config.DRONE_MAX_DISTANCE):
                    remaining_points.append((point, remaining_demand))
        
        # 为每个剩余点创建无人机任务
        drone_idx = 0
        for point, remaining_demand in remaining_points:
            demand_left = remaining_demand
            while demand_left > 0:
                # 选择下一个可用的无人机
                drone_schedule = solution.drone_fleet_schedules[drone_idx % config.MAX_DRONES]
                
                # 创建新的无人机飞行任务
                trip = DroneTrip(drone_schedule.drone_id)
                load_to_take = min(demand_left, config.DRONE_CAPACITY)
                
                trip.sequence_of_points = ['depot', point.id, 'depot']
                trip.visited_points_and_loads = [(point.id, load_to_take)]
                trip.calculate_metrics(self.depot, self.pickup_points)
                
                # 检查可行性
                if trip.is_feasible(self.depot, self.pickup_points):
                    drone_schedule.list_of_drone_trips.append(trip)
                    point.assigned_drone_loads_list.append(load_to_take)
                    demand_left -= load_to_take
                    drone_idx += 1
                else:
                    # 如果单点飞行都不可行，说明该点不能由无人机服务
                    break
        
        # 计算无人机调度指标
        for schedule in solution.drone_fleet_schedules:
            schedule.calculate_metrics()
    
    def _ensure_all_demands_satisfied(self, solution: Solution):
        """确保所有需求都被满足，优先插入现有路线"""
        # 更新取货点分配状态以反映当前解决方案
        solution.update_pickup_point_assignments(self.pickup_points)

        for point in self.pickup_points.values():
            remaining_demand = point.remaining_demand
            
            if remaining_demand > 0:
                # 优先尝试将剩余需求插入现有卡车路线
                inserted = False
                # 按成本升序尝试插入
                sorted_routes = sorted(solution.truck_routes, key=lambda r: r.total_cost)

                for route in sorted_routes:
                    if not route.sequence_of_points:
                        continue
                    
                    current_load = sum(load for _, load in route.visited_points_and_loads)
                    remaining_capacity = config.TRUCK_CAPACITY - current_load
                    
                    if remaining_capacity > 0:
                        load_to_take = min(remaining_demand, remaining_capacity)
                        insertion_result = self._find_best_truck_insertion(route, point)
                        
                        if insertion_result:
                            best_position, _, _ = insertion_result
                            self._insert_into_truck_route(route, point.id, best_position, load_to_take)
                            point.assigned_truck_load += load_to_take
                            remaining_demand -= load_to_take
                            if remaining_demand <= 0:
                                inserted = True
                                break
                
                # 如果无法插入现有路线，则创建新卡车路线
                if not inserted and remaining_demand > 0:
                    while remaining_demand > 0:
                        load_to_take = min(remaining_demand, config.TRUCK_CAPACITY)
                        
                        new_route = TruckRoute(len(solution.truck_routes) + 1)
                        new_route.sequence_of_points = ['depot', point.id, 'depot']
                        new_route.visited_points_and_loads = [(point.id, load_to_take)]
                        new_route.calculate_metrics(self.depot, self.pickup_points)
                        
                        solution.truck_routes.append(new_route)
                        point.assigned_truck_load += load_to_take
                        remaining_demand -= load_to_take

def create_initial_solution(depot: Depot, pickup_points: dict) -> Solution:
    """创建初始解决方案的便捷函数"""
    builder = InitialSolutionBuilder(depot, pickup_points)
    return builder.build_initial_solution()

if __name__ == "__main__":
    from vrp_parser import parse_vrp_file
    
    # 测试初始解构建
    depot, pickup_points = parse_vrp_file("real.vrp")
    solution = create_initial_solution(depot, pickup_points)
    
    print(f"初始解:")
    print(f"卡车路线数: {len([r for r in solution.truck_routes if r.sequence_of_points])}")
    print(f"使用的无人机数: {sum(1 for s in solution.drone_fleet_schedules if s.is_used)}")
    print(f"总完成时间: {solution.total_makespan:.2f} 小时")
    print(f"总运营成本: {solution.total_operating_cost:.2f} 元")
    print(f"解决方案评估值: {solution.evaluate():.6f}")
    
    # 检查可行性
    print(f"解决方案可行性: {solution.is_feasible(pickup_points)}")
    
    # 显示未满足需求
    unassigned = [(p.id, p.remaining_demand) for p in pickup_points.values() if p.remaining_demand > 0]
    if unassigned:
        print(f"未满足需求: {unassigned}")
    else:
        print("所有需求已满足") 