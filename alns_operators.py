import random
import math
import copy
from typing import List, Tuple, Set, Optional
from data_structures import *
import config

class ALNSOperators:
    """ALNS操作符集合"""
    
    def __init__(self, depot: Depot, pickup_points: dict):
        self.depot = depot
        self.pickup_points = pickup_points
    
    # ============= 破坏操作符 =============
    
    def random_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[int]]:
        """随机移除操作符 - 随机选择k个取货点并移除它们的所有分配"""
        assigned_points = set()
        for route in solution.truck_routes:
            for point_id, _ in route.visited_points_and_loads:
                assigned_points.add(point_id)
        
        for schedule in solution.drone_fleet_schedules:
            for trip in schedule.list_of_drone_trips:
                for point_id, _ in trip.visited_points_and_loads:
                    assigned_points.add(point_id)
        
        if not assigned_points:
            return solution, []
        
        k = min(k, len(assigned_points))
        points_to_remove = random.sample(list(assigned_points), k)
        
        self._remove_points_from_solution(solution, points_to_remove)
        
        return solution, points_to_remove
    
    def worst_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[int]]:
        """最差移除操作符 - 移除对目标函数贡献最大的k个取货点"""
        assigned_points = set()
        for route in solution.truck_routes:
            for point_id, _ in route.visited_points_and_loads:
                assigned_points.add(point_id)
        
        for schedule in solution.drone_fleet_schedules:
            for trip in schedule.list_of_drone_trips:
                for point_id, _ in trip.visited_points_and_loads:
                    assigned_points.add(point_id)
        
        if not assigned_points:
            return solution, []
        
        point_costs = []
        for point_id in assigned_points:
            cost = self._calculate_point_removal_benefit(solution, point_id)
            point_costs.append((cost, point_id))
        
        point_costs.sort(reverse=True)
        k = min(k, len(point_costs))
        points_to_remove = [point_id for _, point_id in point_costs[:k]]
        
        self._remove_points_from_solution(solution, points_to_remove)
        
        return solution, points_to_remove
    
    def similarity_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[int]]:
        """相似性移除操作符 - 移除相似的k个取货点"""
        assigned_points = []
        for route in solution.truck_routes:
            for point_id, _ in route.visited_points_and_loads:
                assigned_points.append(point_id)
        
        for schedule in solution.drone_fleet_schedules:
            for trip in schedule.list_of_drone_trips:
                for point_id, _ in trip.visited_points_and_loads:
                    assigned_points.append(point_id)
        
        assigned_points = list(set(assigned_points))
        
        if not assigned_points:
            return solution, []
        
        seed_point = random.choice(assigned_points)
        points_to_remove = [seed_point]
        
        similarities = []
        for point_id in assigned_points:
            if point_id != seed_point:
                similarity = self._calculate_point_similarity(seed_point, point_id)
                similarities.append((similarity, point_id))
        
        similarities.sort(reverse=True)
        k = min(k, len(assigned_points))
        for i in range(min(k-1, len(similarities))):
            points_to_remove.append(similarities[i][1])
        
        self._remove_points_from_solution(solution, points_to_remove)
        
        return solution, points_to_remove
    
    def route_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[int]]:
        """路线移除操作符 - 移除k条完整的路线"""
        points_to_remove = []
        
        all_routes = []
        
        for route in solution.truck_routes:
            if route.visited_points_and_loads:
                all_routes.append(('truck', route))
        
        for schedule in solution.drone_fleet_schedules:
            if schedule.is_used:
                all_routes.append(('drone', schedule))
        
        if not all_routes:
            return solution, []
        
        k = min(k, len(all_routes))
        routes_to_remove = random.sample(all_routes, k)
        
        for route_type, route_obj in routes_to_remove:
            if route_type == 'truck':
                for point_id, _ in route_obj.visited_points_and_loads:
                    points_to_remove.append(point_id)
                route_obj.sequence_of_points = []
                route_obj.visited_points_and_loads = []
                route_obj.calculate_metrics(self.depot, self.pickup_points)
            
            elif route_type == 'drone':
                for trip in route_obj.list_of_drone_trips:
                    for point_id, _ in trip.visited_points_and_loads:
                        points_to_remove.append(point_id)
                route_obj.list_of_drone_trips = []
                route_obj.calculate_metrics()
        
        for point_id in set(points_to_remove):
            if point_id in self.pickup_points:
                self.pickup_points[point_id].reset_assignments()
        
        return solution, list(set(points_to_remove))
    
    def makespan_critical_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[int]]:
        """Makespan关键移除操作符 - 移除导致最长时间的路线中的k个点"""
        max_time = 0
        critical_route = None
        critical_route_type = None
        
        for route in solution.truck_routes:
            if route.total_time > max_time:
                max_time = route.total_time
                critical_route = route
                critical_route_type = 'truck'
        
        for schedule in solution.drone_fleet_schedules:
            if schedule.total_drone_time > max_time:
                max_time = schedule.total_drone_time
                critical_route = schedule
                critical_route_type = 'drone'
        
        if not critical_route:
            return solution, []
        
        points_in_critical_route = []
        if critical_route_type == 'truck':
            points_in_critical_route = [point_id for point_id, _ in critical_route.visited_points_and_loads]
        elif critical_route_type == 'drone':
            for trip in critical_route.list_of_drone_trips:
                points_in_critical_route.extend([point_id for point_id, _ in trip.visited_points_and_loads])
        
        if not points_in_critical_route:
            return solution, []
        
        k = min(k, len(points_in_critical_route))
        points_to_remove = random.sample(points_in_critical_route, k)
        
        self._remove_points_from_solution(solution, points_to_remove)
        
        return solution, points_to_remove
    
    def coordinated_demand_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[int]]:
        """协调需求移除操作符 - 移除由卡车和无人机协同服务的k个点"""
        truck_served_points = set()
        drone_served_points = set()
        
        for route in solution.truck_routes:
            for point_id, _ in route.visited_points_and_loads:
                truck_served_points.add(point_id)
        
        for schedule in solution.drone_fleet_schedules:
            for trip in schedule.list_of_drone_trips:
                for point_id, _ in trip.visited_points_and_loads:
                    drone_served_points.add(point_id)
        
        coordinated_points = list(truck_served_points & drone_served_points)
        
        if not coordinated_points:
            return self.random_removal(solution, k)
        
        k = min(k, len(coordinated_points))
        points_to_remove = random.sample(coordinated_points, k)
        
        self._remove_points_from_solution(solution, points_to_remove)
        
        return solution, points_to_remove
    
    def drone_route_consolidation_removal(self, solution: Solution, k: int) -> Tuple[Solution, List[int]]:
        """无人机路线整合移除操作符 - 移除k条无人机飞行任务"""
        all_drone_trips = []
        for schedule in solution.drone_fleet_schedules:
            for trip in schedule.list_of_drone_trips:
                all_drone_trips.append((schedule, trip))
        
        if not all_drone_trips:
            return solution, []
        
        k = min(k, len(all_drone_trips))
        trips_to_remove = random.sample(all_drone_trips, k)
        
        points_to_remove = []
        for schedule, trip in trips_to_remove:
            for point_id, _ in trip.visited_points_and_loads:
                points_to_remove.append(point_id)
            schedule.list_of_drone_trips.remove(trip)
            schedule.calculate_metrics()
        
        for point_id in set(points_to_remove):
            if point_id in self.pickup_points:
                self.pickup_points[point_id].reset_assignments()
        
        return solution, list(set(points_to_remove))
    
    def _calculate_point_removal_benefit(self, solution: Solution, point_id: int) -> float:
        """计算移除某个点对目标函数的改善程度"""
        point = self.pickup_points[point_id]
        distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 + 
                                    (point.y_coord - self.depot.y_coord)**2)
        
        return point.initial_demand * distance_to_depot
    
    def _calculate_point_similarity(self, point1_id: int, point2_id: int) -> float:
        """计算两个点的相似度"""
        point1 = self.pickup_points[point1_id]
        point2 = self.pickup_points[point2_id]
        
        geo_distance = math.sqrt((point1.x_coord - point2.x_coord)**2 + 
                               (point1.y_coord - point2.y_coord)**2)
        geo_similarity = 1 / (1 + geo_distance)
        
        demand_diff = abs(point1.initial_demand - point2.initial_demand)
        demand_similarity = 1 / (1 + demand_diff)
        
        return config.GEO_SIMILARITY_WEIGHT * geo_similarity + config.DEMAND_SIMILARITY_WEIGHT * demand_similarity
    
    def _remove_points_from_solution(self, solution: Solution, points_to_remove: List[int]):
        """从解决方案中移除指定的取货点"""
        for route in solution.truck_routes:
            original_sequence = route.sequence_of_points.copy()
            route.sequence_of_points = [p for p in route.sequence_of_points 
                                      if p == 'depot' or p not in points_to_remove]
            
            if route.sequence_of_points and route.sequence_of_points[0] != 'depot':
                route.sequence_of_points.insert(0, 'depot')
            if route.sequence_of_points and route.sequence_of_points[-1] != 'depot':
                route.sequence_of_points.append('depot')
            
            route.visited_points_and_loads = [(p, load) for p, load in route.visited_points_and_loads 
                                            if p not in points_to_remove]
            
            if len(route.sequence_of_points) <= 2 and all(p == 'depot' for p in route.sequence_of_points):
                route.sequence_of_points = []
                route.visited_points_and_loads = []
            
            route.calculate_metrics(self.depot, self.pickup_points)
        
        for schedule in solution.drone_fleet_schedules:
            trips_to_remove = []
            for trip in schedule.list_of_drone_trips:
                original_point_count = len(trip.visited_points_and_loads)
                
                trip.sequence_of_points = [p for p in trip.sequence_of_points if p == 'depot' or p not in points_to_remove]
                trip.visited_points_and_loads = [(p, load) for p, load in trip.visited_points_and_loads if p not in points_to_remove]
                
                if not trip.visited_points_and_loads:
                    trips_to_remove.append(trip)
                elif len(trip.visited_points_and_loads) < original_point_count:
                    trip.calculate_metrics(self.depot, self.pickup_points)

            if trips_to_remove:
                schedule.list_of_drone_trips = [t for t in schedule.list_of_drone_trips if t not in trips_to_remove]
            
            schedule.calculate_metrics()
        
        for point_id in points_to_remove:
            if point_id in self.pickup_points:
                self.pickup_points[point_id].reset_assignments()
    
    # ============= 修复操作符 =============
    
    def greedy_insertion(self, partial_solution: Solution, removed_points: List[int]) -> Solution:
        """贪婪插入修复操作符 - 将移除的点贪婪地重新插入到解决方案中"""
        solution = partial_solution
        
        # 确保分配状态同步
        solution.update_pickup_point_assignments(self.pickup_points)
        
        # 只处理有剩余需求的点
        points_with_demand = []
        for point_id in removed_points:
            point = self.pickup_points[point_id]
            if point.remaining_demand > 0:
                points_with_demand.append(point_id)
        
        points_by_demand = sorted(points_with_demand,
                                key=lambda p: self.pickup_points[p].remaining_demand,
                                reverse=True)
        
        for point_id in points_by_demand:
            point = self.pickup_points[point_id]
            self._insert_point_greedily(solution, point)
        
        solution.calculate_solution_metrics()
        
        return solution
    
    def regret_insertion(self, partial_solution: Solution, removed_points: List[int], k: int = 3) -> Solution:
        """后悔插入修复操作符 - 基于后悔值优先插入点"""
        solution = partial_solution
        
        # 确保分配状态同步
        solution.update_pickup_point_assignments(self.pickup_points)
        
        # 只处理有剩余需求的点
        remaining_points = []
        for point_id in removed_points:
            point = self.pickup_points[point_id]
            if point.remaining_demand > 0:
                remaining_points.append(point_id)
        
        while remaining_points:
            regret_values = []
            
            for point_id in remaining_points:
                point = self.pickup_points[point_id]
                # 使用剩余需求计算插入成本
                insertion_costs = self._calculate_top_k_insertion_costs_with_remaining_demand(solution, point, k)
                
                if len(insertion_costs) >= 2:
                    regret_value = insertion_costs[1] - insertion_costs[0]
                else:
                    regret_value = 0
                
                regret_values.append((regret_value, point_id))
            
            regret_values.sort(reverse=True)
            selected_point_id = regret_values[0][1]
            selected_point = self.pickup_points[selected_point_id]
            
            self._insert_point_greedily(solution, selected_point)
            remaining_points.remove(selected_point_id)
        
        solution.calculate_solution_metrics()
        return solution
    
    def hybrid_demand_insertion(self, partial_solution: Solution, removed_points: List[int]) -> Solution:
        """混合需求插入修复操作符 - 智能协同分配策略"""
        solution = partial_solution
        
        # 确保分配状态同步
        solution.update_pickup_point_assignments(self.pickup_points)
        
        # 只处理有剩余需求的点
        points_with_demand = []
        for point_id in removed_points:
            point = self.pickup_points[point_id]
            if point.remaining_demand > 0:
                points_with_demand.append(point_id)
        
        points_by_priority = sorted(points_with_demand,
                                  key=lambda p: self._calculate_insertion_priority_with_remaining_demand(p),
                                  reverse=True)
        
        for point_id in points_by_priority:
            point = self.pickup_points[point_id]
            self._hybrid_insert_point(solution, point)
        
        solution.calculate_solution_metrics()
        return solution
    
    def drone_route_consolidation(self, partial_solution: Solution, removed_points: List[int]) -> Solution:
        """无人机路线整合修复操作符 - 优化无人机多点飞行"""
        solution = partial_solution
        
        # 确保分配状态同步
        solution.update_pickup_point_assignments(self.pickup_points)
        
        drone_reachable_points = []
        truck_only_points = []
        
        for point_id in removed_points:
            point = self.pickup_points[point_id]
            # 只处理有剩余需求的点
            if point.remaining_demand <= 0:
                continue
                
            distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 +
                                        (point.y_coord - self.depot.y_coord)**2)
            
            if (distance_to_depot <= config.DRONE_RADIUS and
                distance_to_depot * 2 <= config.DRONE_MAX_DISTANCE):
                drone_reachable_points.append(point_id)
            else:
                truck_only_points.append(point_id)
        
        for point_id in truck_only_points:
            point = self.pickup_points[point_id]
            self._insert_point_greedily(solution, point)
        
        if drone_reachable_points:
            self._consolidate_drone_points_with_remaining_demand(solution, drone_reachable_points)
        
        solution.calculate_solution_metrics()
        return solution
    
    def inter_route_exchange(self, solution: Solution) -> Solution:
        """路线间交换局部优化"""
        truck_routes_with_indices = []
        for i, route in enumerate(solution.truck_routes):
            if route.visited_points_and_loads:
                truck_routes_with_indices.append(i)
        
        for i in range(len(truck_routes_with_indices)):
            for j in range(i + 1, len(truck_routes_with_indices)):
                route_i_idx = truck_routes_with_indices[i]
                route_j_idx = truck_routes_with_indices[j]
                
                route_i_points = solution.truck_routes[route_i_idx].visited_points_and_loads
                route_j_points = solution.truck_routes[route_j_idx].visited_points_and_loads

                if route_i_points and route_j_points:
                    for pi in range(len(route_i_points)):
                        for pj in range(len(route_j_points)):
                            if self._try_swap_points(solution, route_i_idx, pi, route_j_idx, pj):
                                pass
        
        return solution
    
    def demand_reallocation(self, solution: Solution) -> Solution:
        """需求重新分配局部优化 - 调整卡车和无人机的协同分配"""
        coordinated_points = self._find_coordinated_points(solution)
        
        for point_id in coordinated_points:
            self._try_reallocate_demand(solution, point_id)
        
        solution.calculate_solution_metrics()
        return solution
    
    # ============= 辅助方法 =============
    
    def _calculate_top_k_insertion_costs(self, solution: Solution, point: PickupPoint, k: int) -> List[float]:
        """计算插入某点的前k个最佳成本"""
        costs = []
        
        for route in solution.truck_routes:
            insertion = self._evaluate_truck_insertion(route, point, point.initial_demand)
            if insertion:
                costs.append(insertion['cost_increase'])
        
        new_truck_cost = self._evaluate_new_truck_route(point, point.initial_demand)['cost_increase']
        costs.append(new_truck_cost)
        
        distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 + 
                                    (point.y_coord - self.depot.y_coord)**2)
        if (distance_to_depot <= config.DRONE_RADIUS and 
            distance_to_depot * 2 <= config.DRONE_MAX_DISTANCE):
            drone_insertion = self._evaluate_drone_assignment(solution, point, point.initial_demand)
            if drone_insertion:
                costs.append(drone_insertion['cost_increase'])
        
        costs.sort()
        return costs[:k]
    
    def _calculate_top_k_insertion_costs_with_remaining_demand(self, solution: Solution, point: PickupPoint, k: int) -> List[float]:
        """计算插入某点的前k个最佳成本（基于剩余需求）"""
        costs = []
        remaining_demand = point.remaining_demand
        
        if remaining_demand <= 0:
            return []
        
        for route in solution.truck_routes:
            insertion = self._evaluate_truck_insertion(route, point, remaining_demand)
            if insertion:
                costs.append(insertion['cost_increase'])
        
        new_truck_cost = self._evaluate_new_truck_route(point, remaining_demand)['cost_increase']
        costs.append(new_truck_cost)
        
        distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 +
                                    (point.y_coord - self.depot.y_coord)**2)
        if (distance_to_depot <= config.DRONE_RADIUS and
            distance_to_depot * 2 <= config.DRONE_MAX_DISTANCE):
            drone_insertion = self._evaluate_drone_assignment(solution, point, remaining_demand)
            if drone_insertion:
                costs.append(drone_insertion['cost_increase'])
        
        costs.sort()
        return costs[:k]
    
    def _calculate_insertion_priority(self, point_id: int) -> float:
        """计算点的插入优先级"""
        point = self.pickup_points[point_id]
        distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 +
                                    (point.y_coord - self.depot.y_coord)**2)
        
        return point.initial_demand / (1 + distance_to_depot)
    
    def _calculate_insertion_priority_with_remaining_demand(self, point_id: int) -> float:
        """计算点的插入优先级（基于剩余需求）"""
        point = self.pickup_points[point_id]
        distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 +
                                    (point.y_coord - self.depot.y_coord)**2)
        
        return point.remaining_demand / (1 + distance_to_depot)
    
    def _hybrid_insert_point(self, solution: Solution, point: PickupPoint):
        """混合插入单个点 - 智能选择卡车、无人机或协同方式"""
        # 确保使用当前剩余需求而非初始需求
        solution.update_pickup_point_assignments(self.pickup_points)
        remaining_demand = point.remaining_demand
        
        # 如果没有剩余需求，直接返回
        if remaining_demand <= 0:
            return
        
        distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 + 
                                    (point.y_coord - self.depot.y_coord)**2)
        drone_reachable = (distance_to_depot <= config.DRONE_RADIUS and 
                          distance_to_depot * 2 <= config.DRONE_MAX_DISTANCE)
        
        truck_full_cost = self._evaluate_full_truck_assignment_with_remaining_demand(solution, point)
        
        drone_full_cost = float('inf')
        if drone_reachable:
            drone_full_cost = self._evaluate_full_drone_assignment_with_remaining_demand(solution, point)
        
        best_split_cost = float('inf')
        best_split_plan = None
        if drone_reachable:
            best_split_cost, best_split_plan = self._evaluate_optimal_splitting_with_remaining_demand(solution, point)
        
        if truck_full_cost <= min(drone_full_cost, best_split_cost):
            self._execute_truck_full_assignment(solution, point)
        elif drone_full_cost <= best_split_cost:
            self._execute_drone_full_assignment(solution, point)
        else:
            self._execute_split_assignment(solution, point, best_split_plan)
    
    def _evaluate_full_truck_assignment(self, solution: Solution, point: PickupPoint) -> float:
        """评估卡车全量分配的成本"""
        best_cost = float('inf')
        
        for route in solution.truck_routes:
            insertion = self._evaluate_truck_insertion(route, point, point.initial_demand)
            if insertion and insertion['cost_increase'] < best_cost:
                best_cost = insertion['cost_increase']
        
        new_truck_cost = self._evaluate_new_truck_route(point, point.initial_demand)['cost_increase']
        return min(best_cost, new_truck_cost)
    
    def _evaluate_full_truck_assignment_with_remaining_demand(self, solution: Solution, point: PickupPoint) -> float:
        """评估卡车全量分配的成本（基于剩余需求）"""
        remaining_demand = point.remaining_demand
        if remaining_demand <= 0:
            return 0
            
        best_cost = float('inf')
        
        for route in solution.truck_routes:
            insertion = self._evaluate_truck_insertion(route, point, remaining_demand)
            if insertion and insertion['cost_increase'] < best_cost:
                best_cost = insertion['cost_increase']
        
        new_truck_cost = self._evaluate_new_truck_route(point, remaining_demand)['cost_increase']
        return min(best_cost, new_truck_cost)
    
    def _evaluate_full_drone_assignment(self, solution: Solution, point: PickupPoint) -> float:
        """评估无人机全量分配的成本"""
        total_cost = 0
        remaining_demand = point.initial_demand
        
        while remaining_demand > 0:
            load_per_trip = min(remaining_demand, config.DRONE_CAPACITY)
            
            distance_to_point = math.sqrt((point.x_coord - self.depot.x_coord)**2 +
                                        (point.y_coord - self.depot.y_coord)**2)
            total_distance = distance_to_point * 2
            
            if total_distance > config.DRONE_MAX_DISTANCE:
                return float('inf')
            
            trip_cost = total_distance * config.DRONE_COST_PER_KM
            total_cost += trip_cost
            remaining_demand -= load_per_trip
        
        # 如果此次任务需要激活一架此前未使用的无人机，则计入启动费用
        if any(not s.is_used for s in solution.drone_fleet_schedules):
            total_cost += config.DRONE_STARTUP_COST
        
        return total_cost
    
    def _evaluate_full_drone_assignment_with_remaining_demand(self, solution: Solution, point: PickupPoint) -> float:
        """评估无人机全量分配的成本（基于剩余需求）"""
        remaining_demand = point.remaining_demand
        if remaining_demand <= 0:
            return 0
            
        total_cost = 0
        
        while remaining_demand > 0:
            load_per_trip = min(remaining_demand, config.DRONE_CAPACITY)
            
            distance_to_point = math.sqrt((point.x_coord - self.depot.x_coord)**2 +
                                        (point.y_coord - self.depot.y_coord)**2)
            total_distance = distance_to_point * 2
            
            if total_distance > config.DRONE_MAX_DISTANCE:
                return float('inf')
            
            trip_cost = total_distance * config.DRONE_COST_PER_KM
            total_cost += trip_cost
            remaining_demand -= load_per_trip
        
        # 如果此次任务需要激活一架此前未使用的无人机，则计入启动费用
        if any(not s.is_used for s in solution.drone_fleet_schedules):
            total_cost += config.DRONE_STARTUP_COST
        
        return total_cost
    
    def _evaluate_optimal_splitting(self, solution: Solution, point: PickupPoint) -> Tuple[float, dict]:
        """评估最优的卡车-无人机拆分方案"""
        best_cost = float('inf')
        best_plan = None
        
        for truck_ratio in [0.2, 0.3, 0.5, 0.7, 0.8]:
            truck_load = int(point.initial_demand * truck_ratio)
            drone_load = point.initial_demand - truck_load
            
            if truck_load > 0 and drone_load > 0:
                truck_cost = self._evaluate_partial_truck_assignment(solution, point, truck_load)
                
                drone_cost = self._evaluate_partial_drone_assignment(solution, point, drone_load)
                
                total_cost = truck_cost + drone_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_plan = {'truck_load': truck_load, 'drone_load': drone_load}
        
        return best_cost, best_plan
    
    def _evaluate_optimal_splitting_with_remaining_demand(self, solution: Solution, point: PickupPoint) -> Tuple[float, dict]:
        """评估最优的卡车-无人机拆分方案（基于剩余需求）"""
        remaining_demand = point.remaining_demand
        if remaining_demand <= 0:
            return 0, None
            
        best_cost = float('inf')
        best_plan = None
        
        for truck_ratio in [0.2, 0.3, 0.5, 0.7, 0.8]:
            truck_load = int(remaining_demand * truck_ratio)
            drone_load = remaining_demand - truck_load
            
            if truck_load > 0 and drone_load > 0:
                truck_cost = self._evaluate_partial_truck_assignment(solution, point, truck_load)
                
                drone_cost = self._evaluate_partial_drone_assignment(solution, point, drone_load)
                
                total_cost = truck_cost + drone_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_plan = {'truck_load': truck_load, 'drone_load': drone_load}
        
        return best_cost, best_plan
    
    def _evaluate_partial_truck_assignment(self, solution: Solution, point: PickupPoint, load: int) -> float:
        """评估卡车部分分配的成本"""
        best_cost = float('inf')
        
        for route in solution.truck_routes:
            insertion = self._evaluate_truck_insertion(route, point, load)
            if insertion and insertion['cost_increase'] < best_cost:
                best_cost = insertion['cost_increase']
        
        return best_cost
    
    def _evaluate_partial_drone_assignment(self, solution: Solution, point: PickupPoint, load: int) -> float:
        """评估无人机部分分配的成本"""
        trips_needed = math.ceil(load / config.DRONE_CAPACITY)
        
        total_cost = 0
        for _ in range(trips_needed):
            trip_load = min(load, config.DRONE_CAPACITY)
            
            distance_to_point = math.sqrt((point.x_coord - self.depot.x_coord)**2 + 
                                        (point.y_coord - self.depot.y_coord)**2)
            total_distance = distance_to_point * 2
            
            if total_distance > config.DRONE_MAX_DISTANCE:
                return float('inf')
            
            trip_cost = total_distance * config.DRONE_COST_PER_KM
            total_cost += trip_cost
            load -= trip_load
        
        return total_cost
    
    def _execute_truck_full_assignment(self, solution: Solution, point: PickupPoint):
        """执行卡车全量分配"""
        self._insert_point_greedily(solution, point)
    
    def _execute_drone_full_assignment(self, solution: Solution, point: PickupPoint):
        """执行无人机全量分配"""
        # 使用剩余需求而非初始需求
        remaining_demand = point.remaining_demand
        
        if remaining_demand <= 0:
            return
        
        best_drone_schedule = min(solution.drone_fleet_schedules,
                                key=lambda s: s.total_drone_time)
        
        while remaining_demand > 0:
            load_per_trip = min(remaining_demand, config.DRONE_CAPACITY)
            
            trip = DroneTrip(best_drone_schedule.drone_id)
            trip.sequence_of_points = ['depot', point.id, 'depot']
            trip.visited_points_and_loads = [(point.id, load_per_trip)]
            trip.calculate_metrics(self.depot, self.pickup_points)
            
            best_drone_schedule.list_of_drone_trips.append(trip)
            remaining_demand -= load_per_trip
        
        best_drone_schedule.calculate_metrics()
    
    def _execute_split_assignment(self, solution: Solution, point: PickupPoint, split_plan: dict):
        """执行协同拆分分配"""
        truck_load = split_plan['truck_load']
        drone_load = split_plan['drone_load']
        
        if truck_load > 0:
            self._insert_partial_truck_load(solution, point, truck_load)
        
        if drone_load > 0:
            self._insert_partial_drone_load(solution, point, drone_load)
    
    def _insert_partial_truck_load(self, solution: Solution, point: PickupPoint, load: int):
        """插入部分卡车载货"""
        best_insertion = None
        best_cost = float('inf')
        
        for route in solution.truck_routes:
            insertion = self._evaluate_truck_insertion(route, point, load)
            if insertion and insertion['cost_increase'] < best_cost:
                best_insertion = insertion
                best_cost = insertion['cost_increase']
        
        if best_insertion:
            self._execute_insertion(solution, best_insertion)
    
    def _insert_partial_drone_load(self, solution: Solution, point: PickupPoint, load: int):
        """插入部分无人机载货"""
        remaining_load = load
        best_drone_schedule = min(solution.drone_fleet_schedules, 
                                key=lambda s: s.total_drone_time)
        
        while remaining_load > 0:
            trip_load = min(remaining_load, config.DRONE_CAPACITY)
            
            trip = DroneTrip(best_drone_schedule.drone_id)
            trip.sequence_of_points = ['depot', point.id, 'depot']
            trip.visited_points_and_loads = [(point.id, trip_load)]
            trip.calculate_metrics(self.depot, self.pickup_points)
            
            best_drone_schedule.list_of_drone_trips.append(trip)
            remaining_load -= trip_load
        
        best_drone_schedule.calculate_metrics()
    
    def _consolidate_drone_points(self, solution: Solution, drone_points: List[int]):
        """整合无人机可达点，创建多点飞行任务"""
        remaining_demand = {point_id: self.pickup_points[point_id].initial_demand for point_id in drone_points}
    
    def _consolidate_drone_points_with_remaining_demand(self, solution: Solution, drone_points: List[int]):
        """整合无人机可达点，创建多点飞行任务（基于剩余需求）"""
        remaining_demand = {point_id: self.pickup_points[point_id].remaining_demand for point_id in drone_points}
        
        while remaining_demand:
            current_trip_points = []
            current_trip_load = 0
            
            points_with_demand = [pid for pid, demand in remaining_demand.items() if demand > 0]
            
            if not points_with_demand:
                break
            
            while points_with_demand and current_trip_load < config.DRONE_CAPACITY:
                best_point = self._select_next_drone_point(points_with_demand, current_trip_points)
                
                if best_point is not None:
                    load_to_take = min(remaining_demand[best_point],
                                     config.DRONE_CAPACITY - current_trip_load)
                    
                    if load_to_take > 0:
                        current_trip_points.append((best_point, load_to_take))
                        current_trip_load += load_to_take
                        
                        remaining_demand[best_point] -= load_to_take
                        
                        if remaining_demand[best_point] <= 0:
                            del remaining_demand[best_point]
                            points_with_demand.remove(best_point)
                    else:
                        break
                else:
                    break
            
            if current_trip_points:
                self._create_optimized_drone_trip(solution, current_trip_points)
        
    
    def _select_next_drone_point(self, remaining_points: List[int], current_trip_points: List[Tuple[int, int]]) -> Optional[int]:
        """为无人机飞行选择下一个点"""
        if not remaining_points:
            return None
        
        if not current_trip_points:
            return max(remaining_points, key=lambda p: self.pickup_points[p].initial_demand)
        
        last_point_id = current_trip_points[-1][0]
        last_point = self.pickup_points[last_point_id]
        
        best_point = None
        min_distance = float('inf')
        
        for point_id in remaining_points:
            point = self.pickup_points[point_id]
            distance = math.sqrt((point.x_coord - last_point.x_coord)**2 + 
                               (point.y_coord - last_point.y_coord)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_point = point_id
        
        return best_point
    
    def _create_optimized_drone_trip(self, solution: Solution, trip_points: List[Tuple[int, int]]):
        """创建优化的无人机飞行任务"""
        if not trip_points:
            return
        
        best_drone_schedule = min(solution.drone_fleet_schedules, 
                                key=lambda s: s.total_drone_time)
        
        optimized_sequence = self._optimize_drone_trip_sequence(trip_points)
        
        trip = DroneTrip(best_drone_schedule.drone_id)
        trip.sequence_of_points = ['depot'] + [point_id for point_id, _ in optimized_sequence] + ['depot']
        trip.visited_points_and_loads = optimized_sequence
        trip.calculate_metrics(self.depot, self.pickup_points)
        
        if trip.is_feasible(self.depot, self.pickup_points):
            best_drone_schedule.list_of_drone_trips.append(trip)
            best_drone_schedule.calculate_metrics()
        else:
            for point_id, load in trip_points:
                single_trip = DroneTrip(best_drone_schedule.drone_id)
                single_trip.sequence_of_points = ['depot', point_id, 'depot']
                single_trip.visited_points_and_loads = [(point_id, load)]
                single_trip.calculate_metrics(self.depot, self.pickup_points)
                
                best_drone_schedule.list_of_drone_trips.append(single_trip)
            
            best_drone_schedule.calculate_metrics()
    
    def _optimize_drone_trip_sequence(self, trip_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """优化无人机飞行访问顺序"""
        if len(trip_points) <= 1:
            return trip_points
        
        optimized = [trip_points[0]]
        remaining = trip_points[1:]
        
        while remaining:
            last_point_id = optimized[-1][0]
            last_point = self.pickup_points[last_point_id]
            
            next_point = min(remaining, 
                           key=lambda p: math.sqrt(
                               (self.pickup_points[p[0]].x_coord - last_point.x_coord)**2 + 
                               (self.pickup_points[p[0]].y_coord - last_point.y_coord)**2))
            
            optimized.append(next_point)
            remaining.remove(next_point)
        
        return optimized
    
    def _find_coordinated_points(self, solution: Solution) -> Set[int]:
        """找到被卡车和无人机协同服务的点"""
        truck_points = set()
        drone_points = set()
        
        for route in solution.truck_routes:
            for point_id, _ in route.visited_points_and_loads:
                truck_points.add(point_id)
        
        for schedule in solution.drone_fleet_schedules:
            for trip in schedule.list_of_drone_trips:
                for point_id, _ in trip.visited_points_and_loads:
                    drone_points.add(point_id)
        
        return truck_points & drone_points
    
    def _try_reallocate_demand(self, solution: Solution, point_id: int):
        """尝试重新分配某个点的需求"""
        truck_load = 0
        drone_loads = []
        truck_route_idx = -1
        
        for i, route in enumerate(solution.truck_routes):
            for j, (pid, load) in enumerate(route.visited_points_and_loads):
                if pid == point_id:
                    truck_load = load
                    truck_route_idx = i
                    break
        
        for schedule in solution.drone_fleet_schedules:
            for trip in schedule.list_of_drone_trips:
                for pid, load in trip.visited_points_and_loads:
                    if pid == point_id:
                        drone_loads.append(load)
        
        total_drone_load = sum(drone_loads)
        
        if truck_load > 0 and total_drone_load > 0:
            if truck_load >= 1:
                new_truck_load = truck_load - 1
                
                if self._evaluate_reallocation(solution, point_id, truck_route_idx, new_truck_load):
                    self._execute_reallocation(solution, point_id, truck_route_idx, new_truck_load, total_drone_load + 1)
                    return
            
            if total_drone_load >= 1:
                new_truck_load = truck_load + 1
                
                if self._evaluate_reallocation(solution, point_id, truck_route_idx, new_truck_load):
                    self._execute_reallocation(solution, point_id, truck_route_idx, new_truck_load, total_drone_load - 1)
                    return
    
    def _evaluate_reallocation(self, solution: Solution, point_id: int, truck_route_idx: int, new_truck_load_for_point: int) -> bool:
        """评估重新分配是否有益"""
        if (self.pickup_points[point_id].initial_demand - new_truck_load_for_point) > 0:
            point = self.pickup_points[point_id]
            distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 + 
                                        (point.y_coord - self.depot.y_coord)**2)
            if distance_to_depot > config.DRONE_RADIUS:
                return False

        if truck_route_idx != -1:
            route = solution.truck_routes[truck_route_idx]
            original_load_for_point = 0
            for pid, load in route.visited_points_and_loads:
                if pid == point_id:
                    original_load_for_point = load
                    break
            
            new_total_route_load = route.total_load - original_load_for_point + new_truck_load_for_point
            if new_total_route_load > config.TRUCK_CAPACITY:
                return False
        
        return True
    
    def _execute_reallocation(self, solution: Solution, point_id: int, truck_route_idx: int, new_truck_load: int, new_drone_load: int):
        """执行重新分配"""
        if truck_route_idx >= 0:
            route = solution.truck_routes[truck_route_idx]
            if new_truck_load > 0:
                for i, (pid, load) in enumerate(route.visited_points_and_loads):
                    if pid == point_id:
                        route.visited_points_and_loads[i] = (pid, new_truck_load)
                        break
            else: # new truck load is 0, remove it from truck route
                route.visited_points_and_loads = [item for item in route.visited_points_and_loads if item[0] != point_id]
                route.sequence_of_points = [p for p in route.sequence_of_points if p != point_id]

            route.calculate_metrics(self.depot, self.pickup_points)
        
        for schedule in solution.drone_fleet_schedules:
            schedule.list_of_drone_trips = [
                trip for trip in schedule.list_of_drone_trips 
                if not any(pid == point_id for pid, _ in trip.visited_points_and_loads)
            ]
            schedule.calculate_metrics()
        
        if new_drone_load > 0:
            remaining_load = new_drone_load
            best_drone_schedule = min(solution.drone_fleet_schedules, 
                                    key=lambda s: s.total_drone_time)
            
            while remaining_load > 0:
                trip_load = min(remaining_load, config.DRONE_CAPACITY)
                
                trip = DroneTrip(best_drone_schedule.drone_id)
                trip.sequence_of_points = ['depot', point_id, 'depot']
                trip.visited_points_and_loads = [(point_id, trip_load)]
                trip.calculate_metrics(self.depot, self.pickup_points)
                
                best_drone_schedule.list_of_drone_trips.append(trip)
                remaining_load -= trip_load
            
            best_drone_schedule.calculate_metrics()
    
    def _try_swap_points(self, solution: Solution, route_i: int, pos_i: int, 
                        route_j: int, pos_j: int) -> bool:
        """尝试交换两个路线中的点"""
        route_a = solution.truck_routes[route_i]
        route_b = solution.truck_routes[route_j]
        
        if (pos_i >= len(route_a.visited_points_and_loads) or 
            pos_j >= len(route_b.visited_points_and_loads)):
            return False
        
        point_a_id, load_a = route_a.visited_points_and_loads[pos_i]
        point_b_id, load_b = route_b.visited_points_and_loads[pos_j]
        
        new_load_a = route_a.total_load - load_a + load_b
        new_load_b = route_b.total_load - load_b + load_a
        if new_load_a > config.TRUCK_CAPACITY or new_load_b > config.TRUCK_CAPACITY:
            return False
        
        # 添加负载检查
        if new_load_a < 0 or new_load_b < 0:
            return False

        original_cost = route_a.total_cost + route_b.total_cost
        
        temp_route_a = copy.deepcopy(route_a)
        temp_route_b = copy.deepcopy(route_b)
        
        temp_route_a.visited_points_and_loads[pos_i] = (point_b_id, load_b)
        temp_route_b.visited_points_and_loads[pos_j] = (point_a_id, load_a)
        
        temp_route_a.sequence_of_points = self._rebuild_sequence_from_loads(temp_route_a.visited_points_and_loads)
        temp_route_b.sequence_of_points = self._rebuild_sequence_from_loads(temp_route_b.visited_points_and_loads)
        
        self._ensure_route_integrity(temp_route_a)
        self._ensure_route_integrity(temp_route_b)
        
        temp_route_a.calculate_metrics(self.depot, self.pickup_points)
        temp_route_b.calculate_metrics(self.depot, self.pickup_points)
        
        new_cost = temp_route_a.total_cost + temp_route_b.total_cost
        
        if new_cost < original_cost:
            solution.truck_routes[route_i] = temp_route_a
            solution.truck_routes[route_j] = temp_route_b
            return True
        
        return False
    
    def _ensure_route_integrity(self, route: TruckRoute):
        """确保路线的完整性（以depot开始和结束）"""
        if route.sequence_of_points:
            if route.sequence_of_points[0] != 'depot':
                route.sequence_of_points.insert(0, 'depot')
            if route.sequence_of_points[-1] != 'depot':
                route.sequence_of_points.append('depot')
    
    def _rebuild_sequence_from_loads(self, visited_points_and_loads: List[Tuple[int, int]]) -> List:
        """从载货记录重建访问序列"""
        sequence = ['depot']
        for point_id, _ in visited_points_and_loads:
            sequence.append(point_id)
        sequence.append('depot')
        return sequence

    def _insert_point_greedily(self, solution: Solution, point: PickupPoint):
        """贪婪地插入一个取货点"""
        # 确保使用当前剩余需求而非初始需求
        solution.update_pickup_point_assignments(self.pickup_points)
        remaining_demand = point.remaining_demand
        
        # 如果没有剩余需求，直接返回
        if remaining_demand <= 0:
            return
        max_iterations = 100  # 防止无限循环
        iteration_count = 0
        
        while remaining_demand > 0 and iteration_count < max_iterations:
            best_insertion = None
            best_cost_increase = float('inf')
            
            for route in solution.truck_routes:
                insertion = self._evaluate_truck_insertion(route, point, remaining_demand)
                if insertion and insertion['cost_increase'] < best_cost_increase:
                    best_insertion = insertion
                    best_cost_increase = insertion['cost_increase']
            
            new_truck_insertion = self._evaluate_new_truck_route(point, remaining_demand)
            if new_truck_insertion['cost_increase'] < best_cost_increase:
                best_insertion = new_truck_insertion
                best_cost_increase = new_truck_insertion['cost_increase']
            
            # 检查无人机约束：半径约束和最大距离约束
            distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 + 
                                        (point.y_coord - self.depot.y_coord)**2)
            if (distance_to_depot <= config.DRONE_RADIUS and 
                distance_to_depot * 2 <= config.DRONE_MAX_DISTANCE):
                drone_insertion = self._evaluate_drone_assignment(solution, point, remaining_demand)
                if drone_insertion and drone_insertion['cost_increase'] < best_cost_increase:
                    best_insertion = drone_insertion
            
            if best_insertion:
                assigned_load = self._execute_insertion(solution, best_insertion)
                if assigned_load > 0:
                    remaining_demand -= assigned_load
                else:
                    # 如果分配失败，强制退出防止无限循环
                    print(f"警告: 分配失败，强制使用卡车处理剩余需求 {remaining_demand}")
                    break
            else:
                print(f"警告: 无法为点 {point.id} 找到可行插入，强制使用卡车处理剩余需求 {remaining_demand}")
                break
            
            iteration_count += 1
        
        # 处理剩余需求
        if remaining_demand > 0:
            new_route = TruckRoute(len(solution.truck_routes) + 1)
            new_route.sequence_of_points = ['depot', point.id, 'depot']
            new_route.visited_points_and_loads = [(point.id, remaining_demand)]
            new_route.calculate_metrics(self.depot, self.pickup_points)
            solution.truck_routes.append(new_route)

    def _execute_insertion(self, solution: Solution, insertion: dict) -> int:
        """执行插入操作，返回实际分配的载货量"""
        if insertion['type'] == 'truck_insertion':
            route = insertion['route']
            position = insertion['position']
            load = insertion['load']
            point_id = insertion['point_id']
            
            distance_delta = insertion['distance_delta']
            time_delta = insertion['time_delta']
            cost_increase = insertion['cost_increase']
            
            point_already_in_route = False
            for i, (existing_point_id, existing_load) in enumerate(route.visited_points_and_loads):
                if existing_point_id == point_id:
                    route.visited_points_and_loads[i] = (point_id, existing_load + load)
                    point_already_in_route = True
                    break
            
            if not point_already_in_route:
                if position < len(route.sequence_of_points):
                    route.sequence_of_points.insert(position, point_id)
                else:
                    route.sequence_of_points.insert(-1, point_id)
                route.visited_points_and_loads.append((point_id, load))
            
            # 修复：无论点是否已在路线中，都要更新增量指标
            route.update_metrics_incrementally(distance_delta, cost_increase, time_delta, load)

            return load
            
        elif insertion['type'] == 'new_truck_route':
            point = insertion['point']
            load = insertion['load']
            
            new_route = TruckRoute(len(solution.truck_routes) + 1)
            new_route.sequence_of_points = ['depot', point.id, 'depot']
            new_route.visited_points_and_loads = [(point.id, load)]
            new_route.calculate_metrics(self.depot, self.pickup_points)
            solution.truck_routes.append(new_route)
            return load
            
        elif insertion['type'] == 'drone_assignment':
            schedule = insertion['schedule']
            point = insertion['point']
            load = insertion['load']
            
            trip = DroneTrip(schedule.drone_id)
            trip.sequence_of_points = ['depot', point.id, 'depot']
            trip.visited_points_and_loads = [(point.id, load)]
            trip.calculate_metrics(self.depot, self.pickup_points)
            
            schedule.list_of_drone_trips.append(trip)
            schedule.calculate_metrics()
            return load
        
        return 0

    def _evaluate_truck_insertion(self, route: TruckRoute, point: PickupPoint, demand: int) -> dict:
        """评估将点插入到卡车路线的成本"""
        if not route.sequence_of_points or len(route.sequence_of_points) < 2:
            return None
        
        current_load = route.total_load
        remaining_capacity = config.TRUCK_CAPACITY - current_load
        
        if remaining_capacity <= 0:
            return None
        
        load_to_take = min(demand, remaining_capacity)
        if load_to_take <= 0:
            return None
        
        best_position = -1
        best_cost_increase = float('inf')
        best_deltas = {}
        
        for i in range(1, len(route.sequence_of_points)):
            if route.sequence_of_points[i] == 'depot':
                insertion_deltas = self._calculate_truck_insertion_deltas(route, point, i)
                cost_increase = insertion_deltas['cost_increase']
                
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_position = i
                    best_deltas = insertion_deltas
        
        if best_position >= 0:
            return {
                'type': 'truck_insertion',
                'route': route,
                'position': best_position,
                'point_id': point.id,
                'load': load_to_take,
                'cost_increase': best_cost_increase,
                'distance_delta': best_deltas['distance_delta'],
                'time_delta': best_deltas['time_delta']
            }
        return None
    
    def _calculate_truck_insertion_deltas(self, route: TruckRoute, point: PickupPoint, 
                                          position: int) -> dict:
        """计算卡车插入的指标增量（距离、成本、时间）"""
        # 边界检查
        if position <= 0 or position >= len(route.sequence_of_points):
            return {'distance_delta': 0, 'cost_increase': 0, 'time_delta': 0}
        
        prev_point = route.sequence_of_points[position-1]
        
        if prev_point == 'depot':
            prev_x, prev_y = self.depot.x_coord, self.depot.y_coord
        else:
            prev_x, prev_y = self.pickup_points[prev_point].x_coord, self.pickup_points[prev_point].y_coord
        
        depot_x, depot_y = self.depot.x_coord, self.depot.y_coord
        original_dist = abs(prev_x - depot_x) + abs(prev_y - depot_y)
        
        point_x, point_y = point.x_coord, point.y_coord
        new_dist1 = abs(prev_x - point_x) + abs(prev_y - point_y)
        new_dist2 = abs(point_x - depot_x) + abs(point_y - depot_y)
        
        distance_delta = new_dist1 + new_dist2 - original_dist
        cost_increase = distance_delta * config.TRUCK_COST_PER_KM
        time_delta = distance_delta / max(config.TRUCK_SPEED, config.NUMERICAL_STABILITY_EPSILON)  # 避免除零
        # 添加装载时间：每个新插入的取货点需要0.5小时装载时间
        time_delta += config.TRUCK_LOADING_TIME
        
        return {
            'distance_delta': distance_delta,
            'cost_increase': cost_increase,
            'time_delta': time_delta
        }

    def _calculate_truck_insertion_cost(self, route: TruckRoute, point: PickupPoint, 
                                      position: int, load: int) -> float:
        """计算卡车插入的成本增加 (此方法可能被废弃，或保留用于不需要完整增量信息的场景)"""
        deltas = self._calculate_truck_insertion_deltas(route, point, position)
        return deltas['cost_increase']
    
    def _evaluate_new_truck_route(self, point: PickupPoint, demand: int) -> dict:
        """评估创建新卡车路线的成本"""
        load_to_take = min(demand, config.TRUCK_CAPACITY)
        
        distance_to_point = abs(point.x_coord - self.depot.x_coord) + abs(point.y_coord - self.depot.y_coord)
        total_distance = distance_to_point * 2
        
        cost = total_distance * config.TRUCK_COST_PER_KM + config.TRUCK_STARTUP_COST
        
        return {
            'type': 'new_truck_route',
            'point': point,
            'load': load_to_take,
            'cost_increase': cost
        }
    
    def _evaluate_drone_assignment(self, solution: Solution, point: PickupPoint, demand: int) -> dict:
        """评估无人机分配的成本"""
        best_drone_schedule = None
        min_time = float('inf')
        
        for schedule in solution.drone_fleet_schedules:
            if schedule.total_drone_time < min_time:
                min_time = schedule.total_drone_time
                best_drone_schedule = schedule
        
        if not best_drone_schedule:
            return None
        
        load_to_take = min(demand, config.DRONE_CAPACITY)
        
        distance_to_point = math.sqrt((point.x_coord - self.depot.x_coord)**2 + 
                                    (point.y_coord - self.depot.y_coord)**2)
        total_distance = distance_to_point * 2
        
        if total_distance > config.DRONE_MAX_DISTANCE:
            return None
        
        cost = total_distance * config.DRONE_COST_PER_KM
        
        if not best_drone_schedule.is_used:
            cost += config.DRONE_STARTUP_COST
        
        return {
            'type': 'drone_assignment',
            'schedule': best_drone_schedule,
            'point': point,
            'load': load_to_take,
            'cost_increase': cost
        }

# 操作符权重管理类
class OperatorWeightManager:
    """操作符权重管理器"""
    
    def __init__(self):
        # ε-greedy探索与权重平滑
        self.epsilon = getattr(config, 'OPERATOR_EPSILON', 0.0)
        self.rho = getattr(config, 'WEIGHT_SMOOTHING_RHO', 0.0)
        self.destroy_weights = {
            'random_removal': config.INITIAL_OPERATOR_WEIGHT,
            'worst_removal': config.INITIAL_OPERATOR_WEIGHT,
            'similarity_removal': config.INITIAL_OPERATOR_WEIGHT,
            'route_removal': config.INITIAL_OPERATOR_WEIGHT,
            'makespan_critical_removal': config.INITIAL_OPERATOR_WEIGHT,
            'coordinated_demand_removal': config.INITIAL_OPERATOR_WEIGHT,
            'drone_route_consolidation_removal': config.INITIAL_OPERATOR_WEIGHT
        }
        self.repair_weights = {
            'greedy_insertion': config.INITIAL_OPERATOR_WEIGHT,
            'regret_insertion': config.INITIAL_OPERATOR_WEIGHT,
            'hybrid_demand_insertion': config.INITIAL_OPERATOR_WEIGHT,
            'drone_route_consolidation': config.INITIAL_OPERATOR_WEIGHT,
            'inter_route_exchange': config.INITIAL_OPERATOR_WEIGHT,
            'demand_reallocation': config.INITIAL_OPERATOR_WEIGHT
        }
        
        # 统计信息
        self.destroy_stats = {op: {'improvements': 0, 'neutral': 0, 'worsening': 0} 
                            for op in self.destroy_weights}
        self.repair_stats = {op: {'improvements': 0, 'neutral': 0, 'worsening': 0} 
                           for op in self.repair_weights}
    
    def select_destroy_operator(self) -> str:
        """根据权重选择破坏操作符"""
        return self._weighted_selection(self.destroy_weights)
    
    def select_repair_operator(self) -> str:
        """根据权重选择修复操作符"""
        return self._weighted_selection(self.repair_weights)
    
    def _weighted_selection(self, weights: dict) -> str:
        """加权随机选择"""
        # ε-greedy：以小概率随机探索，防止早熟收敛
        if self.epsilon > 0 and random.random() < self.epsilon:
            return random.choice(list(weights.keys()))

        total_weight = sum(weights.values())
        if total_weight <= 0:
            return random.choice(list(weights.keys()))
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        for op, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return op
        return list(weights.keys())[-1]
    
    def update_operator_performance(self, destroy_op: str, repair_op: str, result: str):
        """更新操作符性能统计"""
        if destroy_op in self.destroy_stats:
            self.destroy_stats[destroy_op][result] += 1
        if repair_op in self.repair_stats:
            self.repair_stats[repair_op][result] += 1
    
    def update_weights(self):
        """根据性能统计更新权重"""
        self._update_operator_weights(self.destroy_weights, self.destroy_stats)
        self._update_operator_weights(self.repair_weights, self.repair_stats)
        
        # 重置统计
        for op in self.destroy_stats:
            self.destroy_stats[op] = {'improvements': 0, 'neutral': 0, 'worsening': 0}
        for op in self.repair_stats:
            self.repair_stats[op] = {'improvements': 0, 'neutral': 0, 'worsening': 0}
    
    def _update_operator_weights(self, weights: dict, stats: dict):
        """更新单个操作符类型的权重"""
        for op in weights:
            total_uses = sum(stats[op].values())
            if total_uses > 0:
                score = (stats[op]['improvements'] * config.IMPROVEMENT_REWARD +
                        stats[op]['neutral'] * config.NEUTRAL_REWARD +
                        stats[op]['worsening'] * config.WORSENING_PENALTY) / total_uses
                # 平滑更新，保留历史，以降低振荡与早熟
                if self.rho and self.rho > 0:
                    new_weight = (1 - self.rho) * weights[op] + self.rho * score
                else:
                    new_weight = score
                weights[op] = max(0.1, new_weight)  # 避免权重过小 
