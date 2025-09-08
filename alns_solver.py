import math
import random
import time
from typing import Tuple
from data_structures import *
from initial_solution import create_initial_solution
from alns_operators import ALNSOperators, OperatorWeightManager
import config

class ALNSSolver:
    """自适应大规模邻域搜索求解器"""
    
    def __init__(self, depot: Depot, pickup_points: dict):
        self.depot = depot
        self.pickup_points = pickup_points
        self.operators = ALNSOperators(depot, pickup_points)
        self.weight_manager = OperatorWeightManager()
        
        # 算法参数
        self.temperature = config.INITIAL_TEMPERATURE
        self.min_temperature = config.MIN_TEMPERATURE
        self.cooling_rate = config.COOLING_RATE
        self.max_iterations = config.MAX_ITERATIONS
        self.max_no_improvement = config.MAX_NO_IMPROVEMENT
        
        # 统计信息
        self.iteration_count = 0
        self.no_improvement_count = 0
        self.start_time = None
        
    def solve(self) -> Solution:
        """执行ALNS算法求解"""
        print("开始ALNS求解...")
        self.start_time = time.time()
        
        # 1. 创建初始解
        print("构建初始解...")
        current_solution = create_initial_solution(self.depot, self.pickup_points)
        
        # 2. 基于初始解设置动态参考值
        config.set_dynamic_reference_values(
            current_solution.total_makespan, 
            current_solution.total_operating_cost
        )
        
        best_solution = self._copy_solution(current_solution)
        
        print(f"初始解评估值: {current_solution.evaluate():.6f}")
        print(f"初始Makespan: {current_solution.total_makespan:.2f}小时")
        print(f"初始成本: {current_solution.total_operating_cost:.2f}元")
        
        # 2. 主循环
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            
            # 复制当前解以进行修改，避免直接修改当前最优解
            candidate_solution = self._copy_solution(current_solution)
            
            # 选择操作符
            destroy_op = self.weight_manager.select_destroy_operator()
            repair_op = self.weight_manager.select_repair_operator()
            
            # 确定破坏规模 (健壮化，兼容小规模实例)
            # 常规范围：upper = min(MAX_DESTROY_SIZE, ⌊|P|/3⌋)，下限至少为 1
            max_k = max(1, min(config.MAX_DESTROY_SIZE, max(1, len(self.pickup_points) // 3)))
            min_k = min(config.MIN_DESTROY_SIZE, max_k)
            if min_k > max_k:
                min_k = max_k

            # 若长期无改进，触发轻量shake：一次性加大破坏规模并复温
            if self.no_improvement_count >= getattr(config, 'STAGNATION_SHAKE_THRESHOLD', 10**9):
                shake_k = int(len(self.pickup_points) * getattr(config, 'SHAKE_DESTROY_FRACTION', 0.5))
                k = max(1, min(shake_k, len(self.pickup_points)))
                # 复温到初温的一定比例，以便提升接受较差解的概率
                target_T = getattr(config, 'REHEAT_FACTOR', 0.3) * config.INITIAL_TEMPERATURE
                if self.temperature < target_T:
                    self.temperature = target_T
                    # 可选日志：提示触发shake
                    # print(f"触发shake：无改进 {self.no_improvement_count} 次，加大破坏规模为 {k}，复温至 {self.temperature:.1f}")
            else:
                k = random.randint(min_k, max_k)
            
            # 应用破坏操作 (现在直接在candidate_solution上操作)
            destroyed_solution, removed_points = self._apply_destroy(candidate_solution, destroy_op, k)
            
            # 应用修复操作 (在同一个candidate_solution上继续操作)
            repaired_solution = self._apply_repair(destroyed_solution, removed_points, repair_op)
            
            # 对修复后的解应用局部搜索以进一步优化
            final_candidate_solution = self._apply_local_search(repaired_solution)
            
            # 检查候选解的可行性，如果不可行则强制修复
            if not final_candidate_solution.is_feasible(self.pickup_points, self.depot):
                # print(f"警告: 迭代 {iteration+1} 产生了不可行解，正在强制修复...")
                final_candidate_solution = self._force_repair_infeasible_solution(final_candidate_solution)
                
                # 再次检查可行性
                if not final_candidate_solution.is_feasible(self.pickup_points, self.depot):
                    print(f"错误: 强制修复后仍不可行，跳过此次迭代")
                    continue
            
            # 评估候选解
            candidate_score = final_candidate_solution.evaluate()
            current_score = current_solution.evaluate()
            
            # 接受准则
            accept, result = self._acceptance_criterion(final_candidate_solution, current_solution)
            # 前若干次迭代输出更详细的决策信息，便于诊断早熟收敛
            if iteration < 10:
                print(f"迭代 {iteration+1}: 接受={accept}, 结果={result}, 当前评估={current_score:.6f}, 候选评估={candidate_score:.6f}, 温度={self.temperature:.2f}")
            
            if accept:
                current_solution = final_candidate_solution
                if candidate_score < best_solution.evaluate():
                    best_solution = self._copy_solution(final_candidate_solution)
                    self.no_improvement_count = 0
                    print(f"迭代 {iteration+1}: 找到更好解! 评估值={candidate_score:.6f}, "
                          f"Makespan={final_candidate_solution.total_makespan:.2f}h, "
                          f"成本={final_candidate_solution.total_operating_cost:.2f}元")
                else:
                    self.no_improvement_count += 1
            else:
                self.no_improvement_count += 1
            
            # 更新操作符性能
            self.weight_manager.update_operator_performance(destroy_op, repair_op, result)
            
            # 更新温度
            self.temperature = self.temperature * self.cooling_rate
            if self.temperature < self.min_temperature:
                self.temperature = self.min_temperature
            
            # 自适应权重更新 - 根据搜索阶段调整更新频率
            update_frequency = self._get_adaptive_update_frequency(iteration)
            if (iteration + 1) % update_frequency == 0:
                self.weight_manager.update_weights()
                # 输出权重更新信息（可选）
                if (iteration + 1) % (update_frequency * 4) == 0:
                    self._log_operator_performance()
            
            # 检查终止条件 - 优化检查顺序
            if self.no_improvement_count >= self.max_no_improvement:
                print(f"达到最大无改进迭代次数 ({self.max_no_improvement})，算法终止")
                break
            
            # 只有在温度足够低且无改进时才因温度终止
            if self.temperature <= self.min_temperature and self.no_improvement_count > 50:
                print("达到最小温度且长时间无改进，算法终止")
                break
            
            # 定期输出进度
            if (iteration + 1) % 100 == 0:
                elapsed_time = time.time() - self.start_time
                print(f"迭代 {iteration+1}/{self.max_iterations}, "
                      f"当前评估值={current_solution.evaluate():.6f}, "
                      f"最佳评估值={best_solution.evaluate():.6f}, "
                      f"温度={self.temperature:.2f}, "
                      f"无改进累计={self.no_improvement_count}, "
                      f"用时={elapsed_time:.1f}s")
        
        total_time = time.time() - self.start_time
        print(f"\nALNS求解完成! 总耗时: {total_time:.2f}秒")
        print(f"总迭代次数: {self.iteration_count}")
        print(f"最终解评估值: {best_solution.evaluate():.6f}")
        print(f"最终Makespan: {best_solution.total_makespan:.2f}小时")
        print(f"最终成本: {best_solution.total_operating_cost:.2f}元")
        
        return best_solution
    
    def _apply_destroy(self, solution: Solution, destroy_op: str, k: int) -> Tuple[Solution, List[int]]:
        """应用破坏操作符"""
        if destroy_op == 'random_removal':
            return self.operators.random_removal(solution, k)
        elif destroy_op == 'worst_removal':
            return self.operators.worst_removal(solution, k)
        elif destroy_op == 'similarity_removal':
            return self.operators.similarity_removal(solution, k)
        elif destroy_op == 'route_removal':
            return self.operators.route_removal(solution, k)
        elif destroy_op == 'makespan_critical_removal':
            return self.operators.makespan_critical_removal(solution, k)
        elif destroy_op == 'coordinated_demand_removal':
            return self.operators.coordinated_demand_removal(solution, k)
        elif destroy_op == 'drone_route_consolidation_removal':
            return self.operators.drone_route_consolidation_removal(solution, k)
        else:
            raise ValueError(f"未知的破坏操作符: {destroy_op}")
    
    def _apply_repair(self, partial_solution: Solution, removed_points: List[int], repair_op: str) -> Solution:
        """应用修复操作符"""
        if repair_op == 'greedy_insertion':
            return self.operators.greedy_insertion(partial_solution, removed_points)
        elif repair_op == 'regret_insertion':
            return self.operators.regret_insertion(partial_solution, removed_points)
        elif repair_op == 'hybrid_demand_insertion':
            return self.operators.hybrid_demand_insertion(partial_solution, removed_points)
        elif repair_op == 'drone_route_consolidation':
            return self.operators.drone_route_consolidation(partial_solution, removed_points)
        elif repair_op == 'inter_route_exchange':
            # 先进行贪婪插入，然后应用路线间交换
            repaired_solution = self.operators.greedy_insertion(partial_solution, removed_points)
            return self.operators.inter_route_exchange(repaired_solution)
        elif repair_op == 'demand_reallocation':
            # 先进行贪婪插入，然后应用需求重新分配
            repaired_solution = self.operators.greedy_insertion(partial_solution, removed_points)
            return self.operators.demand_reallocation(repaired_solution)
        else:
            raise ValueError(f"未知的修复操作符: {repair_op}")
    
    def _apply_local_search(self, solution: Solution) -> Solution:
        """应用局部搜索操作符进一步优化解"""
        # 随机选择一个局部搜索操作符
        # 在未来的版本中，可以为这些操作符也引入权重管理
        local_search_operators = ['inter_route_exchange', 'demand_reallocation']
        selected_op = random.choice(local_search_operators)
        
        improved_solution = solution
        if selected_op == 'inter_route_exchange':
            improved_solution = self.operators.inter_route_exchange(solution)
        elif selected_op == 'demand_reallocation':
            improved_solution = self.operators.demand_reallocation(solution)
            
        return improved_solution
    
    def _acceptance_criterion(self, candidate: Solution, current: Solution) -> Tuple[bool, str]:
        """接受准则（模拟退火 + Makespan优先）"""
        candidate_score = candidate.evaluate()
        current_score = current.evaluate()
        delta_score = candidate_score - current_score
        
        if delta_score < 0:
            # 候选解更优，无条件接受
            return True, 'improvements'
        
        elif delta_score == 0:
            # 解质量相同，接受
            return True, 'neutral'
        
        else:
            # 候选解更差，检查是否通过模拟退火接受
            if self.temperature > 0:
                probability = math.exp(-delta_score / self.temperature)
                if random.random() < probability:
                    return True, 'neutral'
            
            # 检查 Makespan 优先强制接受：当仅因时间更优而接受时，
            # 在算子权重统计中标记为 'neutral'，避免误将其归类为全面“改进”。
            # 这样可以减少算子权重过早向单一目标（时间）倾斜导致的早熟收敛。
            if (candidate.total_makespan < current.total_makespan and
                candidate.total_operating_cost <= current.total_operating_cost *
                (1 + config.MAX_ALLOWED_COST_INCREASE_FACTOR)):
                return True, 'neutral'
            
            return False, 'worsening'
    
    def _copy_solution(self, solution: Solution) -> Solution:
        """优化的解决方案拷贝 - 使用浅拷贝提高性能"""
        import copy
        # 对于频繁拷贝的场景，使用浅拷贝并手动处理需要深拷贝的部分
        new_solution = Solution()
        
        # 拷贝卡车路线
        new_solution.truck_routes = []
        for route in solution.truck_routes:
            new_route = TruckRoute(route.truck_id)
            new_route.sequence_of_points = route.sequence_of_points.copy()
            new_route.visited_points_and_loads = route.visited_points_and_loads.copy()
            new_route.total_distance = route.total_distance
            new_route.total_time = route.total_time
            new_route.total_cost = route.total_cost
            new_route.total_load = route.total_load
            new_solution.truck_routes.append(new_route)
        
        # 拷贝无人机调度
        new_solution.drone_fleet_schedules = []
        for schedule in solution.drone_fleet_schedules:
            new_schedule = DroneFleetSchedule(schedule.drone_id)
            new_schedule.list_of_drone_trips = []
            for trip in schedule.list_of_drone_trips:
                new_trip = DroneTrip(trip.drone_id)
                new_trip.sequence_of_points = trip.sequence_of_points.copy()
                new_trip.visited_points_and_loads = trip.visited_points_and_loads.copy()
                new_trip.total_distance = trip.total_distance
                new_trip.trip_duration = trip.trip_duration
                new_trip.trip_cost = trip.trip_cost
                new_schedule.list_of_drone_trips.append(new_trip)
            new_schedule.total_drone_time = schedule.total_drone_time
            new_schedule.total_drone_cost = schedule.total_drone_cost
            new_schedule.is_used = schedule.is_used
            new_solution.drone_fleet_schedules.append(new_schedule)
        
        # 拷贝其他属性
        new_solution.unassigned_demand = solution.unassigned_demand.copy()
        
        # 重新计算指标，确保与实际路线数据一致
        new_solution.calculate_solution_metrics()
        
        return new_solution
    
    def _force_repair_infeasible_solution(self, solution: Solution) -> Solution:
        """强化的强制修复不可行解机制"""
        try:
            # 更新取货点分配信息
            solution.update_pickup_point_assignments(self.pickup_points)
            
            # 找到未满足需求的点
            unassigned_points = []
            for point_id, point in self.pickup_points.items():
                if point.remaining_demand > 0:
                    unassigned_points.append(point_id)
            
            if not unassigned_points:
                return solution
            
            # 按需求量降序排序，优先处理大需求点
            unassigned_points.sort(key=lambda pid: self.pickup_points[pid].remaining_demand, reverse=True)
            
            # 使用多种策略修复
            for point_id in unassigned_points:
                point = self.pickup_points[point_id]
                remaining_demand = point.remaining_demand
                max_repair_iterations = 100  # 增加最大迭代次数
                repair_iteration = 0
                
                while remaining_demand > 0 and repair_iteration < max_repair_iterations:
                    repair_success = False
                    
                    # 策略1: 尝试插入现有卡车路线
                    best_truck_route = None
                    min_cost_increase = float('inf')
                    
                    for route in solution.truck_routes:
                        current_load = sum(load for _, load in route.visited_points_and_loads)
                        available_capacity = config.TRUCK_CAPACITY - current_load
                        
                        if available_capacity > 0:
                            load_to_assign = min(remaining_demand, available_capacity)
                            cost_increase = self._estimate_insertion_cost(route, point, load_to_assign)
                            
                            if cost_increase < min_cost_increase:
                                min_cost_increase = cost_increase
                                best_truck_route = (route, load_to_assign)
                    
                    if best_truck_route:
                        route, load = best_truck_route
                        self._safe_insert_to_route(route, point_id, load)
                        remaining_demand -= load
                        repair_success = True
                    
                    # 策略2: 尝试无人机分配（如果在半径内）
                    if not repair_success and remaining_demand > 0:
                        distance_to_depot = math.sqrt((point.x_coord - self.depot.x_coord)**2 +
                                                    (point.y_coord - self.depot.y_coord)**2)
                        if (distance_to_depot <= config.DRONE_RADIUS and
                            distance_to_depot * 2 <= config.DRONE_MAX_DISTANCE):
                            
                            # 找到最空闲的无人机
                            best_drone = min(solution.drone_fleet_schedules,
                                           key=lambda s: s.total_drone_time)
                            
                            load_to_assign = min(remaining_demand, config.DRONE_CAPACITY)
                            trip = DroneTrip(best_drone.drone_id)
                            trip.sequence_of_points = ['depot', point_id, 'depot']
                            trip.visited_points_and_loads = [(point_id, load_to_assign)]
                            trip.calculate_metrics(self.depot, self.pickup_points)
                            
                            if trip.is_feasible(self.depot, self.pickup_points):
                                best_drone.list_of_drone_trips.append(trip)
                                best_drone.calculate_metrics()
                                remaining_demand -= load_to_assign
                                repair_success = True
                    
                    # 策略3: 创建新卡车路线
                    if not repair_success and remaining_demand > 0:
                        new_route = TruckRoute(len(solution.truck_routes) + 1)
                        load_to_assign = min(remaining_demand, config.TRUCK_CAPACITY)
                        
                        new_route.sequence_of_points = ['depot', point_id, 'depot']
                        new_route.visited_points_and_loads = [(point_id, load_to_assign)]
                        new_route.calculate_metrics(self.depot, self.pickup_points)
                        
                        solution.truck_routes.append(new_route)
                        remaining_demand -= load_to_assign
                        repair_success = True
                    
                    if not repair_success:
                        print(f"警告: 无法为点 {point_id} 分配剩余需求 {remaining_demand}")
                        break
                    
                    repair_iteration += 1
                
                # 最终检查
                if remaining_demand > 0:
                    print(f"错误: 点 {point_id} 仍有 {remaining_demand} 单位需求未能分配")
            
            # 重新计算解决方案指标
            solution.calculate_solution_metrics()
            return solution
            
        except Exception as e:
            print(f"强制修复过程中发生错误: {e}")
            # 返回原解决方案，避免崩溃
            return solution
    
    def _safe_insert_to_route(self, route: TruckRoute, point_id: int, load: int):
        """安全地将点插入到路线中"""
        try:
            # 检查点是否已经在路线中
            point_already_in_route = False
            for i, (existing_point_id, existing_load) in enumerate(route.visited_points_and_loads):
                if existing_point_id == point_id:
                    # 点已存在，只增加载货量
                    route.visited_points_and_loads[i] = (point_id, existing_load + load)
                    point_already_in_route = True
                    break
            
            if not point_already_in_route:
                # 点不存在，添加新的载货记录和访问序列
                route.visited_points_and_loads.append((point_id, load))
                
                # 重建访问序列：在返回depot前插入新点
                if route.sequence_of_points and route.sequence_of_points[-1] == 'depot':
                    route.sequence_of_points.insert(-1, point_id)
                else:
                    route.sequence_of_points.append(point_id)
                    route.sequence_of_points.append('depot')
            
            route.calculate_metrics(self.depot, self.pickup_points)
            
        except Exception as e:
            print(f"插入点到路线时发生错误: {e}")
    
    def _get_adaptive_update_frequency(self, iteration: int) -> int:
        """根据搜索阶段自适应调整权重更新频率"""
        # 搜索初期更频繁更新，后期减少更新频率
        progress = iteration / self.max_iterations
        
        if progress < 0.3:  # 前30%迭代
            return max(20, config.WEIGHT_UPDATE_FREQUENCY // 2)
        elif progress < 0.7:  # 中间40%迭代
            return config.WEIGHT_UPDATE_FREQUENCY
        else:  # 后30%迭代
            return config.WEIGHT_UPDATE_FREQUENCY * 2
    
    def _log_operator_performance(self):
        """记录算子性能信息"""
        print("\n=== 算子性能统计 ===")
        print("破坏算子权重:")
        for op, weight in self.weight_manager.destroy_weights.items():
            print(f"  {op}: {weight:.3f}")
        
        print("修复算子权重:")
        for op, weight in self.weight_manager.repair_weights.items():
            print(f"  {op}: {weight:.3f}")
        print("=" * 25)
    
    def _estimate_insertion_cost(self, route: TruckRoute, point: PickupPoint, load: int) -> float:
        """估算插入成本"""
        if not route.sequence_of_points:
            return 0.0
        
        # 简化估算：基于到仓库的往返距离
        distance_to_depot = abs(point.x_coord - self.depot.x_coord) + abs(point.y_coord - self.depot.y_coord)
        
        # 估算成本：基于往返距离
        estimated_cost = distance_to_depot * 2 * config.TRUCK_COST_PER_KM
        
        return estimated_cost

def solve_vrp_with_alns(depot: Depot, pickup_points: dict) -> Solution:
    """使用ALNS算法求解VRP问题的便捷函数"""
    solver = ALNSSolver(depot, pickup_points)
    return solver.solve()

if __name__ == "__main__":
    from vrp_parser import parse_vrp_file
    
    # 设置随机种子
    random.seed(config.RANDOM_SEED)
    
    # 解析问题
    depot, pickup_points = parse_vrp_file("real.vrp")
    
    # 求解
    solution = solve_vrp_with_alns(depot, pickup_points)
    
    # 输出结果
    print("\n=== 最终解决方案 ===")
    print(f"卡车路线数: {len([r for r in solution.truck_routes if r.sequence_of_points])}")
    print(f"使用的无人机数: {sum(1 for s in solution.drone_fleet_schedules if s.is_used)}")
    print(f"总完成时间: {solution.total_makespan:.2f}小时")
    print(f"总运营成本: {solution.total_operating_cost:.2f}元")
    print(f"解决方案可行性: {solution.is_feasible(pickup_points)}")
    
    # 显示详细路线信息
    print("\n=== 卡车路线详情 ===")
    for i, route in enumerate(solution.truck_routes):
        if route.sequence_of_points:
            print(f"卡车 {i+1}: {route.sequence_of_points}")
            print(f"  载货: {route.visited_points_and_loads}")
            print(f"  距离: {route.total_distance:.1f}km, 时间: {route.total_time:.2f}h, 成本: {route.total_cost:.2f}元")
    
    print("\n=== 无人机任务详情 ===")
    for schedule in solution.drone_fleet_schedules:
        if schedule.is_used:
            print(f"无人机 {schedule.drone_id}: {len(schedule.list_of_drone_trips)} 次飞行")
            for j, trip in enumerate(schedule.list_of_drone_trips):
                print(f"  飞行 {j+1}: {trip.sequence_of_points}, 载货: {trip.visited_points_and_loads}")
            print(f"  总时间: {schedule.total_drone_time:.2f}h, 总成本: {schedule.total_drone_cost:.2f}元") 
