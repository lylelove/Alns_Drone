import math
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import config

class Depot:
    """仓库类"""
    def __init__(self, x: float, y: float):
        self.x_coord = x
        self.y_coord = y

class PickupPoint:
    """取货点类"""
    def __init__(self, point_id: int, x: float, y: float, initial_demand: int):
        self.id = point_id
        self.x_coord = x
        self.y_coord = y
        self.initial_demand = initial_demand
        self.assigned_truck_load = 0
        self.assigned_drone_loads_list = []  # 记录每次无人机取货量
        
    @property
    def remaining_demand(self) -> int:
        """计算剩余需求量"""
        total_drone_load = sum(self.assigned_drone_loads_list)
        return self.initial_demand - self.assigned_truck_load - total_drone_load
    
    def reset_assignments(self):
        """重置所有分配"""
        self.assigned_truck_load = 0
        self.assigned_drone_loads_list = []

class Vehicle(ABC):
    """车辆抽象基类"""
    def __init__(self, vehicle_id: int, vehicle_type: str, capacity: int, 
                 speed: float, cost_per_km: float, startup_cost: float):
        self.id = vehicle_id
        self.type = vehicle_type
        self.capacity = capacity
        self.speed = speed
        self.cost_per_km = cost_per_km
        self.startup_cost = startup_cost

class Truck(Vehicle):
    """卡车类"""
    def __init__(self, truck_id: int):
        super().__init__(truck_id, 'Truck', config.TRUCK_CAPACITY, 
                        config.TRUCK_SPEED, config.TRUCK_COST_PER_KM, 
                        config.TRUCK_STARTUP_COST)

class Drone(Vehicle):
    """无人机类"""
    def __init__(self, drone_id: int):
        super().__init__(drone_id, 'Drone', config.DRONE_CAPACITY, 
                        config.DRONE_SPEED, config.DRONE_COST_PER_KM, 
                        config.DRONE_STARTUP_COST)

class TruckRoute:
    """卡车路线类"""
    def __init__(self, truck_id: int):
        self.truck_id = truck_id
        self.sequence_of_points = []  # 包含Depot和PickupPoint的访问序列
        self.visited_points_and_loads = []  # [(point_id, load_taken), ...]
        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_cost = 0.0
        self.total_load = 0.0
    
    def update_metrics_incrementally(self, distance_delta: float, cost_delta: float, time_delta: float, load_delta: float):
        """增量式更新路线指标"""
        self.total_distance += distance_delta
        self.total_cost += cost_delta
        self.total_time += time_delta
        self.total_load += load_delta

    def calculate_metrics(self, depot: Depot, pickup_points: dict):
        """计算路线的距离、时间和成本"""
        if not self.sequence_of_points or len(self.sequence_of_points) <= 1:
            self.total_distance = 0.0
            self.total_time = 0.0
            self.total_cost = 0.0
            self.total_load = 0.0
            return
        
        # 只有在路线真实存在时才计算启动费用
        total_distance = 0.0
        total_cost = config.TRUCK_STARTUP_COST
        
        # 计算路径距离和成本
        for i in range(len(self.sequence_of_points) - 1):
            current_point = self.sequence_of_points[i]
            next_point = self.sequence_of_points[i + 1]
            
            # 获取坐标
            if current_point == 'depot':
                curr_x, curr_y = depot.x_coord, depot.y_coord
            else:
                curr_x, curr_y = pickup_points[current_point].x_coord, pickup_points[current_point].y_coord
                
            if next_point == 'depot':
                next_x, next_y = depot.x_coord, depot.y_coord
            else:
                next_x, next_y = pickup_points[next_point].x_coord, pickup_points[next_point].y_coord
            
            # 卡车使用曼哈顿距离
            segment_distance = abs(curr_x - next_x) + abs(curr_y - next_y)
            total_distance += segment_distance
            
            # 成本只与距离有关
            total_cost += segment_distance * config.TRUCK_COST_PER_KM
        
        self.total_distance = total_distance
        self.total_time = total_distance / max(config.TRUCK_SPEED, config.NUMERICAL_STABILITY_EPSILON)  # 避免除零
        # 装载时间：每个取货点都需要装载时间
        num_pickup_points = len(self.visited_points_and_loads)
        if num_pickup_points > 0:
            self.total_time += num_pickup_points * config.TRUCK_LOADING_TIME
        self.total_cost = total_cost
        self.total_load = sum(load for _, load in self.visited_points_and_loads)

class DroneTrip:
    """无人机单次飞行任务类"""
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.sequence_of_points = []  # 包含depot和pickup points的访问序列
        self.visited_points_and_loads = []  # [(point_id, load_taken), ...]
        self.total_distance = 0.0
        self.trip_duration = 0.0
        self.trip_cost = 0.0  # 不含启动费
    
    def calculate_metrics(self, depot: Depot, pickup_points: dict):
        """计算飞行的距离、时间和成本"""
        if len(self.sequence_of_points) < 2:
            self.total_distance = 0.0
            self.trip_duration = 0.0
            self.trip_cost = 0.0
            return
        
        total_distance = 0.0
        total_cost = 0.0
        
        # 计算路径距离和成本
        for i in range(len(self.sequence_of_points) - 1):
            current_point = self.sequence_of_points[i]
            next_point = self.sequence_of_points[i + 1]
            
            # 获取坐标
            if current_point == 'depot':
                curr_x, curr_y = depot.x_coord, depot.y_coord
            else:
                curr_x, curr_y = pickup_points[current_point].x_coord, pickup_points[current_point].y_coord
                
            if next_point == 'depot':
                next_x, next_y = depot.x_coord, depot.y_coord
            else:
                next_x, next_y = pickup_points[next_point].x_coord, pickup_points[next_point].y_coord
            
            # 无人机使用直线距离
            segment_distance = math.sqrt((curr_x - next_x)**2 + (curr_y - next_y)**2)
            total_distance += segment_distance
            
            # 成本只与距离有关
            total_cost += segment_distance * config.DRONE_COST_PER_KM
        
        self.total_distance = total_distance
        self.trip_duration = total_distance / max(config.DRONE_SPEED, config.NUMERICAL_STABILITY_EPSILON)  # 避免除零
        self.trip_cost = total_cost
    
    def is_feasible(self, depot: Depot, pickup_points: dict) -> bool:
        """检查飞行任务是否可行"""
        # 检查容量约束
        total_load = sum(load for _, load in self.visited_points_and_loads)
        if total_load > config.DRONE_CAPACITY:
            return False
        
        # 检查距离约束
        if self.total_distance > config.DRONE_MAX_DISTANCE:
            return False
        
        # 检查半径约束
        for point_id in self.sequence_of_points:
            if point_id != 'depot':
                point = pickup_points[point_id]
                distance_to_depot = math.sqrt((point.x_coord - depot.x_coord)**2 + 
                                            (point.y_coord - depot.y_coord)**2)
                if distance_to_depot > config.DRONE_RADIUS:
                    return False
        
        return True

class DroneFleetSchedule:
    """无人机机队调度表类"""
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.list_of_drone_trips = []  # DroneTrip列表
        self.total_drone_time = 0.0
        self.total_drone_cost = 0.0
        self.is_used = False  # 标记该无人机是否被使用
    
    def calculate_metrics(self):
        """计算无人机总时间和成本"""
        self.total_drone_time = sum(trip.trip_duration for trip in self.list_of_drone_trips)
        trip_costs = sum(trip.trip_cost for trip in self.list_of_drone_trips)
        
        # 如果该无人机被使用，加上启动费用
        if self.list_of_drone_trips:
            self.is_used = True
            self.total_drone_cost = trip_costs + config.DRONE_STARTUP_COST
        else:
            self.is_used = False
            self.total_drone_cost = 0.0

class Solution:
    """解决方案类"""
    def __init__(self):
        self.truck_routes = []  # TruckRoute列表
        self.drone_fleet_schedules = []  # DroneFleetSchedule列表
        self.total_makespan = 0.0
        self.total_operating_cost = 0.0
        self.unassigned_demand = []  # [(point_id, remaining_load), ...]
    
    def update_pickup_point_assignments(self, pickup_points: dict):
        """更新取货点的分配信息以反映当前解决方案"""
        # 首先重置所有取货点的分配
        for point in pickup_points.values():
            point.reset_assignments()
        
        # 收集所有分配信息
        truck_assignments = {}
        drone_assignments = {}
        
        # 收集卡车分配
        for route in self.truck_routes:
            for point_id, load in route.visited_points_and_loads:
                if point_id in pickup_points:
                    truck_assignments[point_id] = truck_assignments.get(point_id, 0) + load
        
        # 收集无人机分配
        for schedule in self.drone_fleet_schedules:
            for trip in schedule.list_of_drone_trips:
                for point_id, load in trip.visited_points_and_loads:
                    if point_id in pickup_points:
                        if point_id not in drone_assignments:
                            drone_assignments[point_id] = []
                        drone_assignments[point_id].append(load)
        
        # 批量更新分配
        for point_id, truck_load in truck_assignments.items():
            pickup_points[point_id].assigned_truck_load = truck_load
        
        for point_id, drone_loads in drone_assignments.items():
            pickup_points[point_id].assigned_drone_loads_list = drone_loads
    
    def calculate_solution_metrics(self):
        """计算解决方案的总体指标"""
        # 计算总时间（Makespan）
        # 只考虑真正有效的路线：至少包含一个取货点的路线
        truck_times = [route.total_time for route in self.truck_routes
                      if route.sequence_of_points and len(route.sequence_of_points) > 2]
        drone_times = [schedule.total_drone_time for schedule in self.drone_fleet_schedules if schedule.is_used]
        
        all_times = truck_times + drone_times
        self.total_makespan = max(all_times) if all_times else 0.0
        
        # 计算总成本
        # 只考虑真正有效的路线：至少包含一个取货点的路线
        truck_costs = sum(route.total_cost for route in self.truck_routes
                         if route.sequence_of_points and len(route.sequence_of_points) > 2)
        drone_costs = sum(schedule.total_drone_cost for schedule in self.drone_fleet_schedules)
        
        self.total_operating_cost = truck_costs + drone_costs
    
    def evaluate(self) -> float:
        """评估解决方案 - 改进的多目标均衡函数"""
        if not config.USE_IMPROVED_OBJECTIVE:
            return self.evaluate_legacy()
        
        # 检查参考值是否已设置
        if config.MAKESPAN_REFERENCE is None or config.COST_REFERENCE is None:
            # 如果参考值未设置，使用legacy方法
            print("警告: 参考值未设置，使用传统评估方法")
            return self.evaluate_legacy()
            
        # 使用规范化的加权和方法
        # 基于实际数值范围进行规范化，避免数量级差异导致的偏向
        
        # 计算规范化的目标值，添加数值稳定性保护
        normalized_makespan = self.total_makespan / max(config.MAKESPAN_REFERENCE, config.NUMERICAL_STABILITY_EPSILON)
        normalized_cost = self.total_operating_cost / max(config.COST_REFERENCE, config.NUMERICAL_STABILITY_EPSILON)
        
        # 自适应权重：根据当前解的特征动态调整权重
        makespan_weight = config.BASE_MAKESPAN_WEIGHT
        cost_weight = config.BASE_COST_WEIGHT
        
        # 动态调整权重（基于相对偏差）
        if normalized_makespan > config.IMBALANCE_THRESHOLD:  # Makespan明显偏高
            makespan_weight = 0.7
            cost_weight = 0.3
        elif normalized_cost > config.IMBALANCE_THRESHOLD:   # 成本明显偏高  
            makespan_weight = 0.5
            cost_weight = 0.5
        
        # 添加平衡性奖励：当两个目标都较好时给予奖励
        if (normalized_makespan < config.BALANCE_BONUS_THRESHOLD and
            normalized_cost < config.BALANCE_BONUS_THRESHOLD):
            balance_bonus = config.BALANCE_BONUS_FACTOR
        else:
            balance_bonus = 1.0
        
        # 使用改进的目标函数
        # 注意根据算法设计文档 (docs/算法设计.md §3.3.3.3)：
        # 仅在成本项上乘以 B_bonus，而不是对加权和整体乘以
        weighted_score = (makespan_weight * normalized_makespan +
                         cost_weight * normalized_cost * balance_bonus)
        
        return weighted_score
        
    def evaluate_legacy(self) -> float:  
        """原始的评估方法（保留作为对比）"""
        # 为legacy方法定义一个固定的小权重值
        EPSILON = 0.005
        return self.total_makespan + EPSILON * self.total_operating_cost
        
    def get_evaluation_details(self) -> dict:
        """获取详细的评估信息，用于分析和调试"""
        if config.MAKESPAN_REFERENCE is None or config.COST_REFERENCE is None:
            # 如果参考值未设置，返回基础信息
            return {
                'total_score': self.evaluate(),
                'legacy_score': self.evaluate_legacy(),
                'makespan_raw': self.total_makespan,
                'cost_raw': self.total_operating_cost,
                'reference_values_set': False
            }
            
        normalized_makespan = self.total_makespan / max(config.MAKESPAN_REFERENCE, config.NUMERICAL_STABILITY_EPSILON)
        normalized_cost = self.total_operating_cost / max(config.COST_REFERENCE, config.NUMERICAL_STABILITY_EPSILON)
        
        return {
            'total_score': self.evaluate(),
            'legacy_score': self.evaluate_legacy(),
            'makespan_raw': self.total_makespan,
            'cost_raw': self.total_operating_cost,
            'makespan_normalized': normalized_makespan,
            'cost_normalized': normalized_cost,
            'makespan_reference': config.MAKESPAN_REFERENCE,
            'cost_reference': config.COST_REFERENCE,
            'makespan_relative': normalized_makespan / (normalized_makespan + normalized_cost) if (normalized_makespan + normalized_cost) > 0 else 0,
            'cost_relative': normalized_cost / (normalized_makespan + normalized_cost) if (normalized_makespan + normalized_cost) > 0 else 0,
            'reference_values_set': True
        }
    
    def is_feasible(self, pickup_points: dict, depot: 'Depot' = None) -> bool:
        """检查解决方案是否可行（所有需求都被满足且满足约束条件）"""
        # 先更新取货点分配信息
        self.update_pickup_point_assignments(pickup_points)
        
        # 检查是否所有需求都被满足
        for point in pickup_points.values():
            if point.remaining_demand > 0:
                return False
        
        # 检查卡车容量约束
        for route in self.truck_routes:
            if route.visited_points_and_loads:
                total_load = sum(load for _, load in route.visited_points_and_loads)
                if total_load > config.TRUCK_CAPACITY:
                    return False
        
        # 检查无人机约束（如果提供了depot信息）
        if depot is not None:
            for schedule in self.drone_fleet_schedules:
                for trip in schedule.list_of_drone_trips:
                    # 检查容量约束
                    total_load = sum(load for _, load in trip.visited_points_and_loads)
                    if total_load > config.DRONE_CAPACITY:
                        return False
                    
                    # 检查距离和半径约束
                    if not trip.is_feasible(depot, pickup_points):
                        return False
        
        return True