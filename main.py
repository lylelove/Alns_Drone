#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整结果分析脚本：运行ALNS算法并保存所有结果到results文件夹
"""

import os
import json
import time
import random
from datetime import datetime
import config
from vrp_parser import parse_vrp_file
from initial_solution import create_initial_solution
from alns_solver import ALNSSolver
from visualization import visualize_solution, create_comprehensive_analysis

def ensure_results_dir():
    """确保results目录存在"""
    if not os.path.exists('results'):
        os.makedirs('results')
    return 'results'

def save_solution_details(solution, pickup_points, filename):
    """保存解决方案详细信息到JSON文件"""
    solution_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_makespan': solution.total_makespan,
            'total_operating_cost': solution.total_operating_cost,
            'evaluation_score': solution.evaluate(),
            'is_feasible': solution.is_feasible(pickup_points)
        },
        'truck_routes': [],
        'drone_schedules': [],
        'summary': {
            'active_trucks': 0,
            'active_drones': 0,
            'total_truck_distance': 0,
            'total_drone_distance': 0,
            'total_pickup_points_served': 0,
            'total_demand_satisfied': 0
        }
    }
    
    # 卡车路线详情
    total_truck_distance = 0
    active_trucks = 0
    for i, route in enumerate(solution.truck_routes):
        if route.sequence_of_points and len(route.sequence_of_points) > 2:
            active_trucks += 1
            route_data = {
                'truck_id': route.truck_id,
                'sequence': route.sequence_of_points,
                'visited_points_and_loads': route.visited_points_and_loads,
                'total_distance': route.total_distance,
                'total_time': route.total_time,
                'total_cost': route.total_cost
            }
            solution_data['truck_routes'].append(route_data)
            total_truck_distance += route.total_distance
    
    # 无人机调度详情
    total_drone_distance = 0
    active_drones = 0
    for schedule in solution.drone_fleet_schedules:
        if schedule.is_used:
            active_drones += 1
            schedule_data = {
                'drone_id': schedule.drone_id,
                'total_time': schedule.total_drone_time,
                'total_cost': schedule.total_drone_cost,
                'trips': []
            }
            
            for trip in schedule.list_of_drone_trips:
                trip_data = {
                    'sequence': trip.sequence_of_points,
                    'visited_points_and_loads': trip.visited_points_and_loads,
                    'distance': trip.total_distance,
                    'duration': trip.trip_duration,
                    'cost': trip.trip_cost
                }
                schedule_data['trips'].append(trip_data)
                total_drone_distance += trip.total_distance
            
            solution_data['drone_schedules'].append(schedule_data)
    
    # 统计汇总
    total_demand = sum(point.initial_demand for point in pickup_points.values())
    
    solution_data['summary'] = {
        'active_trucks': active_trucks,
        'active_drones': active_drones,
        'total_truck_distance': total_truck_distance,
        'total_drone_distance': total_drone_distance,
        'total_pickup_points_served': len(pickup_points),
        'total_demand_satisfied': total_demand,
        'average_truck_distance': total_truck_distance / max(active_trucks, 1),
        'average_drone_distance': total_drone_distance / max(active_drones, 1)
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(solution_data, f, ensure_ascii=False, indent=2)

def save_text_report(initial_solution, final_solution, pickup_points, solver_stats, filename):
    """保存文本格式的详细报告"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("卡车-无人机协同VRP问题 ALNS算法求解报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 问题规模
        f.write("问题规模:\n")
        f.write(f"  取货点数量: {len(pickup_points)}\n")
        f.write(f"  总需求量: {sum(p.initial_demand for p in pickup_points.values())} 单位\n")
        f.write(f"  最大卡车数量: 无限制\n")
        f.write(f"  可用无人机数量: {config.MAX_DRONES}\n\n")
        
        # 车辆参数
        f.write("车辆参数:\n")
        f.write("  卡车:\n")
        f.write(f"    容量: {config.TRUCK_CAPACITY} 单位\n")
        f.write(f"    速度: {config.TRUCK_SPEED} km/h\n")
        f.write(f"    成本: {config.TRUCK_COST_PER_KM} 元/km\n")
        f.write(f"    启动费用: {config.TRUCK_STARTUP_COST} 元\n")
        f.write(f"    装载时间: {config.TRUCK_LOADING_TIME} 小时\n")
        f.write("  无人机:\n")
        f.write(f"    容量: {config.DRONE_CAPACITY} 单位\n")
        f.write(f"    速度: {config.DRONE_SPEED} km/h\n")
        f.write(f"    成本: {config.DRONE_COST_PER_KM} 元/km\n")
        f.write(f"    启动费用: {config.DRONE_STARTUP_COST} 元\n")
        f.write(f"    飞行半径: {config.DRONE_RADIUS} km\n")
        f.write(f"    最大飞行距离: {config.DRONE_MAX_DISTANCE} km\n\n")
        
        # 算法参数
        f.write("算法参数:\n")
        f.write(f"  最大迭代次数: {config.MAX_ITERATIONS}\n")
        f.write(f"  初始温度: {config.INITIAL_TEMPERATURE}\n")
        f.write(f"  冷却率: {config.COOLING_RATE}\n")
        f.write(f"  最大无改进次数: {config.MAX_NO_IMPROVEMENT}\n\n")
        
        # 初始解结果
        f.write("初始解结果:\n")
        f.write(f"  总完成时间: {initial_solution.total_makespan:.2f} 小时\n")
        f.write(f"  总运营成本: {initial_solution.total_operating_cost:.2f} 元\n")
        f.write(f"  评估值: {initial_solution.evaluate():.6f}\n")
        f.write(f"  可行性: {'✓' if initial_solution.is_feasible(pickup_points) else '✗'}\n\n")
        
        # 最终解结果
        f.write("最终解结果:\n")
        f.write(f"  总完成时间: {final_solution.total_makespan:.2f} 小时\n")
        f.write(f"  总运营成本: {final_solution.total_operating_cost:.2f} 元\n")
        f.write(f"  评估值: {final_solution.evaluate():.6f}\n")
        f.write(f"  可行性: {'✓' if final_solution.is_feasible(pickup_points) else '✗'}\n\n")
        
        # 改进情况
        improvement_percentage = ((initial_solution.evaluate() - final_solution.evaluate()) / 
                                initial_solution.evaluate()) * 100
        f.write("改进情况:\n")
        f.write(f"  目标函数改进: {improvement_percentage:.1f}%\n")
        f.write(f"  时间改进: {initial_solution.total_makespan - final_solution.total_makespan:.2f} 小时\n")
        f.write(f"  成本变化: {final_solution.total_operating_cost - initial_solution.total_operating_cost:.2f} 元\n\n")
        
        # 求解统计
        f.write("求解统计:\n")
        f.write(f"  总迭代次数: {solver_stats['iterations']}\n")
        f.write(f"  求解时间: {solver_stats['time']:.2f} 秒\n")
        f.write(f"  最后改进迭代: {solver_stats.get('last_improvement', 'N/A')}\n\n")
        
        # 详细解决方案
        f.write("详细解决方案:\n")
        f.write("-" * 40 + "\n")
        
        # 卡车路线
        active_truck_count = 0
        total_truck_distance = 0
        f.write("卡车路线:\n")
        for i, route in enumerate(final_solution.truck_routes):
            if route.sequence_of_points and len(route.sequence_of_points) > 2:
                active_truck_count += 1
                total_truck_distance += route.total_distance
                f.write(f"  卡车 {active_truck_count}:\n")
                f.write(f"    路线: {' -> '.join(map(str, route.sequence_of_points))}\n")
                f.write(f"    载货: {route.visited_points_and_loads}\n")
                f.write(f"    距离: {route.total_distance:.1f}km\n")
                f.write(f"    时间: {route.total_time:.2f}小时\n")
                f.write(f"    成本: {route.total_cost:.2f}元\n\n")
        
        # 无人机任务
        active_drone_count = 0
        total_drone_distance = 0
        f.write("无人机任务:\n")
        for schedule in final_solution.drone_fleet_schedules:
            if schedule.is_used:
                active_drone_count += 1
                for trip in schedule.list_of_drone_trips:
                    total_drone_distance += trip.total_distance
                f.write(f"  无人机 {active_drone_count} (总时间: {schedule.total_drone_time:.2f}h, 总成本: {schedule.total_drone_cost:.2f}元):\n")
                for j, trip in enumerate(schedule.list_of_drone_trips):
                    f.write(f"    飞行 {j+1}: {' -> '.join(map(str, trip.sequence_of_points))}\n")
                    f.write(f"      载货: {trip.visited_points_and_loads}\n")
                    f.write(f"      距离: {trip.total_distance:.1f}km, 时间: {trip.trip_duration:.2f}h\n")
                f.write("\n")
        
        # 总结统计
        f.write("资源使用总结:\n")
        f.write(f"  使用卡车数量: {active_truck_count}\n")
        f.write(f"  使用无人机数量: {active_drone_count}\n")
        f.write(f"  卡车总里程: {total_truck_distance:.1f}km\n")
        f.write(f"  无人机总里程: {total_drone_distance:.1f}km\n")
        f.write(f"  平均卡车里程: {total_truck_distance / max(active_truck_count, 1):.1f}km\n")
        f.write(f"  平均无人机里程: {total_drone_distance / max(active_drone_count, 1):.1f}km\n")

def run_complete_analysis():
    """运行完整分析"""
    print("开始完整的ALNS算法分析...")
    
    # 创建结果目录
    results_dir = ensure_results_dir()
    
    # 设置随机种子
    random.seed(config.RANDOM_SEED)
    print(f"使用随机种子: {config.RANDOM_SEED}")
    
    # 解析问题
    print("解析VRP问题文件...")
    depot, pickup_points = parse_vrp_file('real_2.vrp')
    print(f"成功解析：{len(pickup_points)} 个取货点，总需求 {sum(p.initial_demand for p in pickup_points.values())} 单位")
    
    # 创建初始解
    print("构建初始解...")
    initial_solution = create_initial_solution(depot, pickup_points)
    print(f"初始解：Makespan={initial_solution.total_makespan:.2f}h, 成本={initial_solution.total_operating_cost:.2f}元")
    print(f"初始解可行性: {'✓' if initial_solution.is_feasible(pickup_points) else '✗'}")
    
    # 运行ALNS算法
    print("\n开始ALNS算法求解...")
    start_time = time.time()
    
    solver = ALNSSolver(depot, pickup_points)
    final_solution = solver.solve()
    
    solve_time = time.time() - start_time
    print(f"\nALNS求解完成，用时 {solve_time:.2f} 秒")
    
    # 收集求解统计信息
    solver_stats = {
        'iterations': solver.iteration_count,
        'time': solve_time,
        'last_improvement': solver.iteration_count - solver.no_improvement_count
    }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存JSON格式的详细数据
    print("保存解决方案数据...")
    json_file = os.path.join(results_dir, f'solution_data_{timestamp}.json')
    save_solution_details(final_solution, pickup_points, json_file)
    
    # 2. 保存文本报告
    print("生成详细报告...")
    report_file = os.path.join(results_dir, f'detailed_report_{timestamp}.txt')
    save_text_report(initial_solution, final_solution, pickup_points, solver_stats, report_file)
    
    # 3. 生成可视化图像
    print("生成可视化图像...")
    
    # 生成综合分析可视化
    figures = create_comprehensive_analysis(depot, pickup_points, final_solution,
                                          "ALNS-Based Truck-Drone Collaborative VRP Solution",
                                          f"solution_analysis_{timestamp}")
    
    print(f"✅ 已生成 {len(figures)} 个详细分析图片")
    
    # 4. 保存简化的结果摘要
    summary_file = os.path.join(results_dir, f'solution_summary_{timestamp}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ALNS算法求解结果摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"问题规模: {len(pickup_points)} 个取货点\n")
        f.write(f"总需求量: {sum(p.initial_demand for p in pickup_points.values())} 单位\n")
        f.write(f"求解时间: {solve_time:.2f} 秒\n")
        f.write(f"总迭代次数: {solver.iteration_count}\n\n")
        
        f.write("初始解:\n")
        f.write(f"  Makespan: {initial_solution.total_makespan:.2f} 小时\n")
        f.write(f"  总成本: {initial_solution.total_operating_cost:.2f} 元\n")
        f.write(f"  评估值: {initial_solution.evaluate():.6f}\n\n")
        
        f.write("最终解:\n")
        f.write(f"  Makespan: {final_solution.total_makespan:.2f} 小时\n")
        f.write(f"  总成本: {final_solution.total_operating_cost:.2f} 元\n")
        f.write(f"  评估值: {final_solution.evaluate():.6f}\n")
        f.write(f"  可行性: {'✓' if final_solution.is_feasible(pickup_points) else '✗'}\n\n")
        
        improvement = ((initial_solution.evaluate() - final_solution.evaluate()) / 
                      initial_solution.evaluate()) * 100
        f.write(f"改进百分比: {improvement:.1f}%\n")
        
        # 资源使用情况
        active_trucks = sum(1 for r in final_solution.truck_routes if r.sequence_of_points and len(r.sequence_of_points) > 2)
        active_drones = sum(1 for s in final_solution.drone_fleet_schedules if s.is_used)
        f.write(f"使用卡车: {active_trucks} 辆\n")
        f.write(f"使用无人机: {active_drones} 架\n")
    
    print(f"\n所有结果已保存到 '{results_dir}' 文件夹:")
    print(f"  - 解决方案数据: {os.path.basename(json_file)}")
    print(f"  - 详细报告: {os.path.basename(report_file)}")
    print(f"  - 可视化图像: solution_analysis_{timestamp}_main_solution.png")
    print(f"  - 结果摘要: {os.path.basename(summary_file)}")
    
    return final_solution, solver_stats, pickup_points

if __name__ == "__main__":
    try:
        final_solution, stats, pickup_points = run_complete_analysis()
        print("\n✅ 完整分析执行成功！")
        print(f"最终解评估值: {final_solution.evaluate():.6f}")
        print(f"解决方案可行性: {'✓' if final_solution.is_feasible(pickup_points) else '✗'}")
    except Exception as e:
        print(f"\n❌ 分析执行失败: {e}")
        import traceback
        traceback.print_exc() 