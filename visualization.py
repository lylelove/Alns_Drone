import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Tuple
from data_structures import *
import config

class VRPVisualizer:
    """VRP问题可视化工具"""
    
    def __init__(self, depot: Depot, pickup_points: dict):
        self.depot = depot
        self.pickup_points = pickup_points
        
        # 设置matplotlib参数以获得学术论文风格
        plt.rcParams.update({
            'font.size': 10,  # 减小字体大小
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'text.usetex': False,
            'figure.figsize': (config.FIGURE_WIDTH, config.FIGURE_HEIGHT),
            'figure.dpi': config.DPI,
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True,
            'legend.fontsize': 9,  # 减小图例字体
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9
        })
    
    def analyze_transportation_modes(self, solution: Solution) -> Dict:
        """分析运输方式统计"""
        transport_analysis = {
            'truck_only': [],      # 仅卡车运输的取货点
            'drone_only': [],      # 仅无人机运输的取货点
            'collaborative': [],   # 协同运输的取货点
            'unserved': []         # 未服务的取货点
        }
        
        for point_id, point in self.pickup_points.items():
            truck_load = 0
            drone_load = 0
            
            # 统计卡车运输量
            for route in solution.truck_routes:
                for visited_point, load in route.visited_points_and_loads:
                    if visited_point == point_id:
                        truck_load += load
            
            # 统计无人机运输量
            for schedule in solution.drone_fleet_schedules:
                for trip in schedule.list_of_drone_trips:
                    for visited_point, load in trip.visited_points_and_loads:
                        if visited_point == point_id:
                            drone_load += load
            
            # 分类运输方式
            total_served = truck_load + drone_load
            if total_served == 0:
                transport_analysis['unserved'].append((point_id, point))
            elif truck_load > 0 and drone_load > 0:
                transport_analysis['collaborative'].append((point_id, point, truck_load, drone_load))
            elif truck_load > 0:
                transport_analysis['truck_only'].append((point_id, point, truck_load))
            elif drone_load > 0:
                transport_analysis['drone_only'].append((point_id, point, drone_load))
        
        return transport_analysis
    
    def create_comprehensive_visualization(self, solution: Solution,
                                         title: str = "ALNS-Based Truck-Drone Collaborative VRP Solution",
                                         save_prefix: str = "vrp_analysis"):
        """创建综合可视化分析，生成多个图片文件"""
        
        # 1. 主要解决方案图（4子图）
        fig1 = self.plot_solution(solution, title)
        self.save_figure(fig1, f"{save_prefix}_main_solution.png")
        
        # 2. 运输方式分析图
        fig2 = self.plot_transportation_analysis(solution)
        self.save_figure(fig2, f"{save_prefix}_transport_modes.png")
        
        # 3. 详细统计图
        fig3 = self.plot_detailed_statistics(solution)
        self.save_figure(fig3, f"{save_prefix}_detailed_stats.png")
        
        return [fig1, fig2, fig3]
    
    def plot_individual_solution_overview(self, solution: Solution,
                                        title: str = "Complete Solution Overview",
                                        save_filename: str = "solution_overview.png"):
        """绘制独立的完整解决方案概览图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        self._plot_complete_solution(ax, solution)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 调整图例位置，避免遮挡
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_truck_routes(self, solution: Solution,
                                   title: str = "Truck Routes",
                                   save_filename: str = "truck_routes.png"):
        """绘制独立的卡车路线图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        self._plot_truck_routes_only(ax, solution)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 调整图例位置，避免遮挡
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_drone_missions(self, solution: Solution,
                                     title: str = "Drone Missions",
                                     save_filename: str = "drone_missions.png"):
        """绘制独立的无人机任务图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        self._plot_drone_missions_only(ax, solution)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 调整图例位置，避免遮挡
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_performance_metrics(self, solution: Solution,
                                          title: str = "Performance Metrics",
                                          save_filename: str = "performance_metrics.png"):
        """绘制独立的性能指标图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        self._plot_performance_metrics(ax, solution)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_transport_mode_map(self, solution: Solution,
                                         title: str = "Transportation Mode Distribution",
                                         save_filename: str = "transport_mode_map.png"):
        """绘制独立的运输方式分布地图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        transport_analysis = self.analyze_transportation_modes(solution)
        self._plot_transport_mode_map(ax, transport_analysis)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # 调整图例位置，避免遮挡
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_transport_mode_pie(self, solution: Solution,
                                         title: str = "Transportation Mode Statistics",
                                         save_filename: str = "transport_mode_pie.png"):
        """绘制独立的运输方式统计饼图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        transport_analysis = self.analyze_transportation_modes(solution)
        self._plot_transport_mode_pie(ax, transport_analysis)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_demand_allocation(self, solution: Solution,
                                        title: str = "Demand Allocation Analysis",
                                        save_filename: str = "demand_allocation.png"):
        """绘制独立的需求量分配分析图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        transport_analysis = self.analyze_transportation_modes(solution)
        self._plot_demand_allocation(ax, transport_analysis)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_collaborative_details(self, solution: Solution,
                                            title: str = "Collaborative Transportation Details",
                                            save_filename: str = "collaborative_details.png"):
        """绘制独立的协同运输详情图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        transport_analysis = self.analyze_transportation_modes(solution)
        self._plot_collaborative_details(ax, transport_analysis)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_vehicle_utilization(self, solution: Solution,
                                          title: str = "Vehicle Utilization Analysis",
                                          save_filename: str = "vehicle_utilization.png"):
        """绘制独立的车辆利用率分析图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        self._plot_vehicle_utilization(ax, solution)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_time_analysis(self, solution: Solution,
                                    title: str = "Time Distribution Analysis",
                                    save_filename: str = "time_analysis.png"):
        """绘制独立的时间分析图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        self._plot_time_analysis(ax, solution)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_cost_breakdown(self, solution: Solution,
                                     title: str = "Cost Structure Analysis",
                                     save_filename: str = "cost_breakdown.png"):
        """绘制独立的成本结构分析图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        self._plot_cost_breakdown(ax, solution)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_individual_distance_efficiency(self, solution: Solution,
                                          title: str = "Distance & Efficiency Analysis",
                                          save_filename: str = "distance_efficiency.png"):
        """绘制独立的距离和效率分析图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        self._plot_distance_efficiency(ax, solution)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, save_filename)
        return fig
    
    def plot_transportation_analysis(self, solution: Solution):
        """绘制运输方式分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 分析运输方式
        transport_analysis = self.analyze_transportation_modes(solution)
        
        # 子图1: 运输方式分布地图
        self._plot_transport_mode_map(ax1, transport_analysis)
        ax1.set_title("(a) Transportation Mode Distribution", fontsize=14, fontweight='bold')
        
        # 子图2: 运输方式统计饼图
        self._plot_transport_mode_pie(ax2, transport_analysis)
        ax2.set_title("(b) Transportation Mode Statistics", fontsize=14, fontweight='bold')
        
        # 子图3: 需求量分配分析
        self._plot_demand_allocation(ax3, transport_analysis)
        ax3.set_title("(c) Demand Allocation Analysis", fontsize=14, fontweight='bold')
        
        # 子图4: 协同运输详情
        self._plot_collaborative_details(ax4, transport_analysis)
        ax4.set_title("(d) Collaborative Transportation Details", fontsize=14, fontweight='bold')
        
        # 设置主标题
        fig.suptitle("Transportation Mode Analysis", fontsize=16, fontweight='bold', y=0.96)
        
        # 调整布局 - 增加间距
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
        
        return fig
    
    def _plot_transport_mode_map(self, ax, transport_analysis: Dict):
        """绘制运输方式分布地图"""
        # 绘制仓库
        ax.scatter(self.depot.x_coord, self.depot.y_coord, 
                  s=80, c='red', marker='s', label='Depot', 
                  edgecolor='black', linewidth=2, zorder=10)
        
        # 绘制无人机半径
        drone_circle = patches.Circle((self.depot.x_coord, self.depot.y_coord),
                                config.DRONE_RADIUS, fill=False, color='blue',
                                linestyle='--', alpha=0.3, linewidth=2)
        ax.add_patch(drone_circle)
        
        # 定义颜色和标记
        colors = {
            'truck_only': '#FF6B6B',      # 红色 - 仅卡车
            'drone_only': '#4ECDC4',      # 青色 - 仅无人机
            'collaborative': '#45B7D1',   # 蓝色 - 协同
            'unserved': '#95A5A6'         # 灰色 - 未服务
        }
        
        markers = {
            'truck_only': 's',      # 方形
            'drone_only': '^',      # 三角形
            'collaborative': 'D',   # 菱形
            'unserved': 'x'         # 叉号
        }
        
        labels = {
            'truck_only': 'Truck Only',
            'drone_only': 'Drone Only', 
            'collaborative': 'Collaborative',
            'unserved': 'Unserved'
        }
        
        # 绘制不同运输方式的取货点
        for mode, points_data in transport_analysis.items():
            if not points_data:
                continue
                
            x_coords = []
            y_coords = []
            sizes = []
            
            for data in points_data:
                point_id, point = data[0], data[1]
                x_coords.append(point.x_coord)
                y_coords.append(point.y_coord)
                
                # 根据需求量设置大小
                if mode == 'collaborative':
                    # 协同运输：大小基于总需求
                    sizes.append(max(80, point.initial_demand * 10))
                else:
                    sizes.append(max(80, point.initial_demand * 8))
            
            if x_coords:
                ax.scatter(x_coords, y_coords, s=sizes, c=colors[mode], 
                          marker=markers[mode], label=labels[mode], 
                          alpha=0.8, edgecolor='black', linewidth=1, zorder=5)
        
        self._format_axis(ax)
        ax.legend(fontsize=8)
    
    def _plot_transport_mode_pie(self, ax, transport_analysis: Dict):
        """绘制运输方式统计饼图"""
        # 统计各种运输方式的数量和需求量
        mode_counts = {}
        mode_demands = {}
        
        for mode, points_data in transport_analysis.items():
            mode_counts[mode] = len(points_data)
            mode_demands[mode] = 0
            
            for data in points_data:
                point = data[1]
                mode_demands[mode] += point.initial_demand
        
        # 过滤掉空的类别
        non_empty_modes = {k: v for k, v in mode_counts.items() if v > 0}
        
        if not non_empty_modes:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            return
        
        # 绘制饼图
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6']
        labels = []
        sizes = []
        colors_used = []
        
        color_map = {
            'truck_only': '#FF6B6B',
            'drone_only': '#4ECDC4', 
            'collaborative': '#45B7D1',
            'unserved': '#95A5A6'
        }
        
        label_map = {
            'truck_only': 'Truck Only',
            'drone_only': 'Drone Only',
            'collaborative': 'Collaborative', 
            'unserved': 'Unserved'
        }
        
        for mode, count in non_empty_modes.items():
            labels.append(f'{label_map[mode]}\n({count} points)')
            sizes.append(count)
            colors_used.append(color_map[mode])
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_used, 
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 10})
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_demand_allocation(self, ax, transport_analysis: Dict):
        """绘制需求量分配分析"""
        modes = []
        truck_demands = []
        drone_demands = []
        
        # 统计各模式的需求分配
        for mode, points_data in transport_analysis.items():
            if not points_data:
                continue
                
            total_truck = 0
            total_drone = 0
            
            if mode == 'truck_only':
                for point_id, point, load in points_data:
                    total_truck += load
                modes.append('Truck Only')
                truck_demands.append(total_truck)
                drone_demands.append(0)
                
            elif mode == 'drone_only':
                for point_id, point, load in points_data:
                    total_drone += load
                modes.append('Drone Only')
                truck_demands.append(0)
                drone_demands.append(total_drone)
                
            elif mode == 'collaborative':
                for point_id, point, truck_load, drone_load in points_data:
                    total_truck += truck_load
                    total_drone += drone_load
                modes.append('Collaborative')
                truck_demands.append(total_truck)
                drone_demands.append(total_drone)
        
        if not modes:
            ax.text(0.5, 0.5, 'No demand data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # 创建堆叠条形图
        x = np.arange(len(modes))
        width = 0.6
        
        bars1 = ax.bar(x, truck_demands, width, label='Truck Demand', 
                      color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x, drone_demands, width, bottom=truck_demands, 
                      label='Drone Demand', color='#4ECDC4', alpha=0.8)
        
        # 添加数值标签
        for i, (truck, drone) in enumerate(zip(truck_demands, drone_demands)):
            if truck > 0:
                ax.text(i, truck/2, f'{truck}', ha='center', va='center', 
                       fontweight='bold', color='white')
            if drone > 0:
                ax.text(i, truck + drone/2, f'{drone}', ha='center', va='center',
                       fontweight='bold', color='white')
        
        ax.set_xlabel('Transportation Mode', fontweight='bold')
        ax.set_ylabel('Demand Units', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_collaborative_details(self, ax, transport_analysis: Dict):
        """绘制协同运输详情"""
        collaborative_points = transport_analysis.get('collaborative', [])
        
        if not collaborative_points:
            ax.text(0.5, 0.5, 'No collaborative transportation found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # 准备数据
        point_ids = []
        truck_loads = []
        drone_loads = []
        truck_ratios = []
        
        for point_id, point, truck_load, drone_load in collaborative_points:
            point_ids.append(f'P{point_id}')
            truck_loads.append(truck_load)
            drone_loads.append(drone_load)
            total_load = truck_load + drone_load
            truck_ratios.append(truck_load / total_load * 100)
        
        # 创建双轴图
        x = np.arange(len(point_ids))
        width = 0.35
        
        # 左轴：绝对需求量
        bars1 = ax.bar(x - width/2, truck_loads, width, label='Truck Load', 
                      color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, drone_loads, width, label='Drone Load', 
                      color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Pickup Points', fontweight='bold')
        ax.set_ylabel('Load (units)', fontweight='bold', color='black')
        ax.set_xticks(x)
        ax.set_xticklabels(point_ids, rotation=45)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 右轴：卡车占比
        ax2 = ax.twinx()
        line = ax2.plot(x, truck_ratios, color='#45B7D1', marker='o', 
                       linewidth=2, markersize=6, label='Truck Ratio')
        ax2.set_ylabel('Truck Ratio (%)', fontweight='bold', color='#45B7D1')
        ax2.tick_params(axis='y', labelcolor='#45B7D1')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def plot_solution(self, solution: Solution, title: str = "ALNS-Based Truck-Drone Collaborative VRP Solution"):
        """绘制完整的解决方案图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 完整解决方案概览
        self._plot_complete_solution(ax1, solution)
        ax1.set_title("(a) Complete Solution Overview", fontsize=14, fontweight='bold')
        
        # 子图2: 卡车路线详情
        self._plot_truck_routes_only(ax2, solution)
        ax2.set_title("(b) Truck Routes", fontsize=14, fontweight='bold')
        
        # 子图3: 无人机任务详情
        self._plot_drone_missions_only(ax3, solution)
        ax3.set_title("(c) Drone Missions", fontsize=14, fontweight='bold')
        
        # 子图4: 性能统计
        self._plot_performance_metrics(ax4, solution)
        ax4.set_title("(d) Performance Metrics", fontsize=14, fontweight='bold')
        
        # 设置主标题
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
        
        # 调整布局 - 增加间距，避免重叠
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
        
        return fig
    
    def _plot_complete_solution(self, ax, solution: Solution):
        """绘制完整解决方案"""
        # 绘制仓库
        ax.scatter(self.depot.x_coord, self.depot.y_coord, 
                  s=80, c='red', marker='s', label='Depot', 
                  edgecolor='black', linewidth=2, zorder=10)
        
        # 绘制取货点
        pickup_x = [p.x_coord for p in self.pickup_points.values()]
        pickup_y = [p.y_coord for p in self.pickup_points.values()]
        demands = [p.initial_demand for p in self.pickup_points.values()]
        
        # 根据需求量设置点的大小
        sizes = [max(50, d * 8) for d in demands]
        scatter = ax.scatter(pickup_x, pickup_y, s=sizes, c=demands, 
                           cmap='viridis', alpha=0.7, label='Pickup Points',
                           edgecolor='black', linewidth=0.5)
        
        # # 添加颜色条
        # cbar = plt.colorbar(scatter, ax=ax)
        # cbar.set_label('Demand', rotation=270, labelpad=45)
        
        # 绘制卡车路线
        active_truck_routes = [route for route in solution.truck_routes if route.sequence_of_points and len(route.sequence_of_points) > 2]
        truck_colors = plt.get_cmap('Set1')(np.linspace(0, 1, max(len(active_truck_routes), 1)))
        
        for i, (route, color) in enumerate(zip(active_truck_routes, truck_colors)):
            self._draw_truck_route(ax, route, color, f'Truck {i+1}')
        
        # 绘制无人机半径
        drone_circle = patches.Circle((self.depot.x_coord, self.depot.y_coord),
                                config.DRONE_RADIUS, fill=False, color='blue',
                                linestyle='--', alpha=0.5, linewidth=2)
        ax.add_patch(drone_circle)
        
        # 优化无人机半径标注位置 - 放在右下角，避免遮挡
        all_x = [self.depot.x_coord] + [p.x_coord for p in self.pickup_points.values()]
        all_y = [self.depot.y_coord] + [p.y_coord for p in self.pickup_points.values()]
        
        # 计算合适的标注位置（图表右下角）
        text_x = max(all_x) - 5
        text_y = min(all_y) + 3
        
        ax.text(text_x, text_y, f'Drone Range: {config.DRONE_RADIUS}km', 
                fontsize=9, color='blue', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='blue'))
        
        # 绘制无人机任务
        drone_colors = plt.get_cmap('Set2')(np.linspace(0, 1, config.MAX_DRONES))
        for schedule, color in zip(solution.drone_fleet_schedules, drone_colors):
            if schedule.is_used:
                self._draw_drone_missions(ax, schedule, color, f'Drone {schedule.drone_id}')
        
        self._format_axis(ax)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    def _plot_truck_routes_only(self, ax, solution: Solution):
        """仅绘制卡车路线"""
        # 绘制仓库和取货点
        self._draw_base_map(ax)
        
        # 绘制卡车路线 - 修复编号问题
        active_truck_routes = [route for route in solution.truck_routes if route.sequence_of_points and len(route.sequence_of_points) > 2]
        truck_colors = plt.get_cmap('Set1')(np.linspace(0, 1, max(len(active_truck_routes), 1)))
        
        for i, (route, color) in enumerate(zip(active_truck_routes, truck_colors)):
            self._draw_truck_route(ax, route, color, f'Truck {i+1}')
        
        self._format_axis(ax)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    def _plot_drone_missions_only(self, ax, solution: Solution):
        """仅绘制无人机任务"""
        # 绘制仓库和取货点
        self._draw_base_map(ax)
        
        # 绘制无人机半径
        drone_circle = patches.Circle((self.depot.x_coord, self.depot.y_coord),
                                config.DRONE_RADIUS, fill=False, color='blue',
                                linestyle='--', alpha=0.5, linewidth=2)
        ax.add_patch(drone_circle)
        
        # 绘制无人机任务
        drone_colors = plt.get_cmap('Set2')(np.linspace(0, 1, config.MAX_DRONES))
        for schedule, color in zip(solution.drone_fleet_schedules, drone_colors):
            if schedule.is_used:
                self._draw_drone_missions(ax, schedule, color, f'Drone {schedule.drone_id}')
        
        self._format_axis(ax)
        ax.legend(fontsize=8)
    
    def _plot_performance_metrics(self, ax, solution: Solution):
        """绘制改进的性能指标"""
        # 计算统计数据
        active_truck_routes = [route for route in solution.truck_routes 
                              if route.sequence_of_points and len(route.sequence_of_points) > 2]
        truck_count = len(active_truck_routes)
        drone_count = sum(1 for s in solution.drone_fleet_schedules if s.is_used)
        total_points = len(self.pickup_points)
        total_demand = sum(p.initial_demand for p in self.pickup_points.values())
        
        # 计算运输方式统计
        transport_analysis = self.analyze_transportation_modes(solution)
        collaborative_count = len(transport_analysis.get('collaborative', []))
        
        # 创建更丰富的条形图
        metrics = ['Trucks\nUsed', 'Drones\nUsed', 'Makespan\n(h)', 'Cost\n(×100¥)', 'Collaborative\nPoints']
        values = [truck_count, drone_count, solution.total_makespan, 
                 solution.total_operating_cost/100, collaborative_count]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签 - 优化位置和字体
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 添加改进的表格信息
        table_data = [
            ['Total Pickup Points', f'{total_points}'],
            ['Total Demand', f'{total_demand}'],
            ['Points Served', f'{total_points - len(transport_analysis.get("unserved", []))}'],
            ['Solution Feasible', f'{solution.is_feasible(self.pickup_points)}'],
            ['Objective Value', f'{solution.evaluate():.3f}'],
            ['Avg Truck Utilization', f'{self._calculate_avg_truck_utilization(solution):.1f}%'],
            ['Avg Drone Utilization', f'{self._calculate_avg_drone_utilization(solution):.1f}%']
        ]
        
        table = ax.table(cellText=table_data, 
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='lower center',
                        bbox=[0.1, -0.6, 0.8, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        
        # 美化表格
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 标题行
                    cell.set_facecolor('#E8E8E8')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#F8F8F8' if i % 2 == 0 else 'white')
        
        ax.set_ylabel('Value', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(values) * 1.3)
    
    def _calculate_avg_truck_utilization(self, solution: Solution) -> float:
        """计算平均卡车利用率"""
        active_trucks = [route for route in solution.truck_routes 
                        if route.sequence_of_points and len(route.sequence_of_points) > 2]
        if not active_trucks:
            return 0.0
        
        total_utilization = 0
        for route in active_trucks:
            total_load = sum(load for _, load in route.visited_points_and_loads)
            utilization = (total_load / config.TRUCK_CAPACITY) * 100
            total_utilization += utilization
        
        return total_utilization / len(active_trucks)
    
    def _calculate_avg_drone_utilization(self, solution: Solution) -> float:
        """计算平均无人机利用率
        
        对于无人机，由于可以重复利用进行多次飞行，利用率应该计算为：
        每次飞行的容量利用率的平均值
        """
        active_drones = [schedule for schedule in solution.drone_fleet_schedules if schedule.is_used]
        if not active_drones:
            return 0.0
        
        total_utilization = 0
        total_trips = 0
        
        for schedule in active_drones:
            for trip in schedule.list_of_drone_trips:
                # 计算这次飞行的总载货量
                trip_load = sum(load for _, load in trip.visited_points_and_loads)
                # 计算这次飞行的容量利用率
                trip_utilization = (trip_load / config.DRONE_CAPACITY) * 100
                total_utilization += trip_utilization
                total_trips += 1
        
        return total_utilization / total_trips if total_trips > 0 else 0.0
    
    def plot_detailed_statistics(self, solution: Solution):
        """绘制详细统计分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 车辆利用率分析
        self._plot_vehicle_utilization(ax1, solution)
        ax1.set_title("(a) Vehicle Utilization Analysis", fontsize=14, fontweight='bold')
        
        # 子图2: 时间分析
        self._plot_time_analysis(ax2, solution)
        ax2.set_title("(b) Time Distribution Analysis", fontsize=14, fontweight='bold')
        
        # 子图3: 成本结构分析
        self._plot_cost_breakdown(ax3, solution)
        ax3.set_title("(c) Cost Structure Analysis", fontsize=14, fontweight='bold')
        
        # 子图4: 距离和效率分析
        self._plot_distance_efficiency(ax4, solution)
        ax4.set_title("(d) Distance & Efficiency Analysis", fontsize=14, fontweight='bold')
        
        # 设置主标题
        fig.suptitle("Detailed Performance Statistics", fontsize=16, fontweight='bold', y=0.96)
        
        # 调整布局 - 增加间距
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
        
        return fig
    
    def _plot_vehicle_utilization(self, ax, solution: Solution):
        """绘制车辆利用率分析"""
        # 统计卡车利用率
        active_trucks = [route for route in solution.truck_routes 
                        if route.sequence_of_points and len(route.sequence_of_points) > 2]
        truck_utilizations = []
        truck_labels = []
        
        for i, route in enumerate(active_trucks):
            total_load = sum(load for _, load in route.visited_points_and_loads)
            utilization = (total_load / config.TRUCK_CAPACITY) * 100
            truck_utilizations.append(utilization)
            truck_labels.append(f'T{i+1}')
        
        # 统计无人机利用率
        active_drones = [schedule for schedule in solution.drone_fleet_schedules if schedule.is_used]
        drone_utilizations = []
        drone_labels = []
        
        for schedule in active_drones:
            # 计算该无人机所有飞行的平均利用率
            if schedule.list_of_drone_trips:
                trip_utilizations = []
                for trip in schedule.list_of_drone_trips:
                    trip_load = sum(load for _, load in trip.visited_points_and_loads)
                    trip_utilization = (trip_load / config.DRONE_CAPACITY) * 100
                    trip_utilizations.append(trip_utilization)
                
                # 该无人机的平均利用率
                avg_utilization = sum(trip_utilizations) / len(trip_utilizations)
                drone_utilizations.append(avg_utilization)
            else:
                drone_utilizations.append(0.0)
            
            drone_labels.append(f'D{schedule.drone_id}')

        # 创建分组条形图
        x_trucks = np.arange(len(truck_labels))
        x_drones = np.arange(len(drone_labels)) + len(truck_labels) + 1
        
        # 绘制卡车利用率
        bars1 = ax.bar(x_trucks, truck_utilizations, color='#FF6B6B', 
                      alpha=0.8, label='Trucks', edgecolor='black', linewidth=1)
        
        # 绘制无人机利用率
        if drone_utilizations:
            bars2 = ax.bar(x_drones, drone_utilizations, color='#4ECDC4', 
                          alpha=0.8, label='Drones', edgecolor='black', linewidth=1)
        
        # 添加数值标签 - 优化位置和字体
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(max(truck_utilizations + [0]), max(drone_utilizations + [0])) * 0.01,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        if drone_utilizations:
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(max(truck_utilizations + [0]), max(drone_utilizations + [0])) * 0.01,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 添加100%参考线
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Full Capacity')
        
        # 设置标签和格式
        all_labels = truck_labels + ([''] if truck_labels and drone_labels else []) + drone_labels
        all_x = list(x_trucks) + ([len(truck_labels)] if truck_labels and drone_labels else []) + list(x_drones)
        
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_labels)
        ax.set_ylabel('Capacity Utilization (%)', fontweight='bold')
        ax.set_xlabel('Vehicles', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(truck_utilizations + [0]), max(drone_utilizations + [0])) * 1.2)
    
    def _plot_time_analysis(self, ax, solution: Solution):
        """绘制时间分析"""
        # 收集时间数据
        truck_times = []
        drone_times = []
        
        # 卡车时间
        for route in solution.truck_routes:
            if route.sequence_of_points and len(route.sequence_of_points) > 2:
                truck_times.append(route.total_time)
        
        # 无人机时间
        for schedule in solution.drone_fleet_schedules:
            if schedule.is_used:
                drone_times.append(schedule.total_drone_time)
        
        # 创建箱线图
        data_to_plot = []
        labels = []
        
        if truck_times:
            data_to_plot.append(truck_times)
            labels.append('Trucks')
        
        if drone_times:
            data_to_plot.append(drone_times)
            labels.append('Drones')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
            
            # 添加统计信息 - 优化位置，避免遮挡图表
            stats_text = []
            if truck_times:
                stats_text.append(f'Trucks: μ={np.mean(truck_times):.2f}h, σ={np.std(truck_times):.2f}h')
            if drone_times:
                stats_text.append(f'Drones: μ={np.mean(drone_times):.2f}h, σ={np.std(drone_times):.2f}h')
            
            # 将统计信息放在右上角，使用更小的字体和更透明的背景
            ax.text(0.98, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7, edgecolor='gray'))
        
        # 添加makespan线
        ax.axhline(y=solution.total_makespan, color='red', linestyle='--', 
                  alpha=0.8, label=f'Makespan: {solution.total_makespan:.2f}h')
        
        ax.set_ylabel('Time (hours)', fontweight='bold')
        ax.set_title('Vehicle Operation Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cost_breakdown(self, ax, solution: Solution):
        """绘制成本结构分析"""
        # 计算成本组成
        truck_distance_cost = 0
        truck_startup_cost = 0
        drone_distance_cost = 0
        drone_startup_cost = 0
        
        # 卡车成本
        active_truck_count = 0
        for route in solution.truck_routes:
            if route.sequence_of_points and len(route.sequence_of_points) > 2:
                active_truck_count += 1
                truck_distance_cost += route.total_cost - config.TRUCK_STARTUP_COST
        
        truck_startup_cost = active_truck_count * config.TRUCK_STARTUP_COST
        
        # 无人机成本
        active_drone_count = sum(1 for schedule in solution.drone_fleet_schedules if schedule.is_used)
        for schedule in solution.drone_fleet_schedules:
            if schedule.is_used:
                drone_distance_cost += schedule.total_drone_cost - config.DRONE_STARTUP_COST
        
        drone_startup_cost = active_drone_count * config.DRONE_STARTUP_COST
        
        # 创建堆叠条形图
        categories = ['Truck Costs', 'Drone Costs']
        startup_costs = [truck_startup_cost, drone_startup_cost]
        distance_costs = [truck_distance_cost, drone_distance_cost]
        
        x = np.arange(len(categories))
        width = 0.6
        
        bars1 = ax.bar(x, startup_costs, width, label='Startup Costs', 
                      color='#FF9999', alpha=0.8)
        bars2 = ax.bar(x, distance_costs, width, bottom=startup_costs, 
                      label='Distance Costs', color='#66B2FF', alpha=0.8)
        
        # 添加数值标签
        for i, (startup, distance) in enumerate(zip(startup_costs, distance_costs)):
            if startup > 0:
                ax.text(i, startup/2, f'{startup:.0f}¥', ha='center', va='center',
                       fontweight='bold', color='white')
            if distance > 0:
                ax.text(i, startup + distance/2, f'{distance:.0f}¥', ha='center', va='center',
                       fontweight='bold', color='white')
            
            # 总计标签
            total = startup + distance
            ax.text(i, total + max(startup_costs + distance_costs) * 0.02, 
                   f'Total: {total:.0f}¥', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Vehicle Type', fontweight='bold')
        ax.set_ylabel('Cost (CNY)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加总成本信息 - 优化位置和样式
        total_cost = sum(startup_costs) + sum(distance_costs)
        ax.text(0.02, 0.98, f'Total: {total_cost:.0f}¥', 
               transform=ax.transAxes, ha='left', va='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7, edgecolor='green'))
    
    def _plot_distance_efficiency(self, ax, solution: Solution):
        """绘制距离和效率分析"""
        # 收集距离数据
        truck_distances = []
        truck_loads = []
        drone_distances = []
        drone_loads = []
        
        # 卡车数据
        for route in solution.truck_routes:
            if route.sequence_of_points and len(route.sequence_of_points) > 2:
                truck_distances.append(route.total_distance)
                total_load = sum(load for _, load in route.visited_points_and_loads)
                truck_loads.append(total_load)
        
        # 无人机数据
        for schedule in solution.drone_fleet_schedules:
            if schedule.is_used:
                total_distance = sum(trip.total_distance for trip in schedule.list_of_drone_trips)
                total_load = sum(sum(load for _, load in trip.visited_points_and_loads) 
                               for trip in schedule.list_of_drone_trips)
                drone_distances.append(total_distance)
                drone_loads.append(total_load)
        
        # 计算效率（载货量/距离）
        truck_efficiency = [load/dist if dist > 0 else 0 for load, dist in zip(truck_loads, truck_distances)]
        drone_efficiency = [load/dist if dist > 0 else 0 for load, dist in zip(drone_loads, drone_distances)]
        
        # 创建散点图
        if truck_distances:
            ax.scatter(truck_distances, truck_efficiency, s=100, c='#FF6B6B', 
                      alpha=0.7, label='Trucks', marker='s', edgecolor='black')
        
        if drone_distances:
            ax.scatter(drone_distances, drone_efficiency, s=100, c='#4ECDC4', 
                      alpha=0.7, label='Drones', marker='^', edgecolor='black')
        
        # 添加趋势线
        if len(truck_distances) > 1 and len(truck_efficiency) > 1:
            z = np.polyfit(truck_distances, truck_efficiency, 1)
            p = np.poly1d(z)
            ax.plot(truck_distances, p(truck_distances), "r--", alpha=0.8, linewidth=2)
        
        if len(drone_distances) > 1 and len(drone_efficiency) > 1:
            z = np.polyfit(drone_distances, drone_efficiency, 1)
            p = np.poly1d(z)
            ax.plot(drone_distances, p(drone_distances), "b--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Total Distance (km)', fontweight='bold')
        ax.set_ylabel('Efficiency (units/km)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息 - 优化位置，避免与散点图重叠
        stats_text = []
        if truck_efficiency:
            stats_text.append(f'Truck Avg: {np.mean(truck_efficiency):.2f} units/km')
        if drone_efficiency:
            stats_text.append(f'Drone Avg: {np.mean(drone_efficiency):.2f} units/km')
        
        if stats_text:
            ax.text(0.02, 0.02, '\n'.join(stats_text), transform=ax.transAxes,
                   verticalalignment='bottom', horizontalalignment='left', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.7, edgecolor='cyan'))
    
    def _draw_base_map(self, ax):
        """绘制基础地图（仓库和取货点）"""
        # 绘制仓库
        ax.scatter(self.depot.x_coord, self.depot.y_coord, 
                  s=80, c='red', marker='s', label='Depot', 
                  edgecolor='black', linewidth=2, zorder=10)
        
        # 绘制取货点
        pickup_x = [p.x_coord for p in self.pickup_points.values()]
        pickup_y = [p.y_coord for p in self.pickup_points.values()]
        demands = [p.initial_demand for p in self.pickup_points.values()]
        
        sizes = [max(50, d * 8) for d in demands]
        ax.scatter(pickup_x, pickup_y, s=sizes, c='lightgray', 
                  alpha=0.6, label='Pickup Points',
                  edgecolor='black', linewidth=0.5)
        
        # 添加点标签 - 优化位置和字体，减少重叠
        for point_id, point in self.pickup_points.items():
            ax.annotate(f'{point_id}', 
                       (point.x_coord, point.y_coord),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=7, alpha=0.8, color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))
    
    def _draw_truck_route(self, ax, route: TruckRoute, color, label):
        """绘制卡车路线"""
        if len(route.sequence_of_points) < 2:
            return
        
        x_coords = []
        y_coords = []
        
        for point in route.sequence_of_points:
            if point == 'depot':
                x_coords.append(self.depot.x_coord)
                y_coords.append(self.depot.y_coord)
            else:
                pickup_point = self.pickup_points[point]
                x_coords.append(pickup_point.x_coord)
                y_coords.append(pickup_point.y_coord)
        
        # 绘制路线（使用曼哈顿距离风格的连线）
        for i in range(len(x_coords) - 1):
            # 曼哈顿路径：先水平后垂直
            x1, y1 = x_coords[i], y_coords[i]
            x2, y2 = x_coords[i + 1], y_coords[i + 1]
            
            # 水平线段
            ax.plot([x1, x2], [y1, y1], color=color, linewidth=2, alpha=0.8)
            # 垂直线段
            ax.plot([x2, x2], [y1, y2], color=color, linewidth=2, alpha=0.8)
            
            # 添加箭头指示方向
            if i == 0:
                ax.annotate('', xy=(x2, y1), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax.annotate('', xy=(x2, y2), xytext=(x2, y1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # 添加图例项（只在第一个线段添加）
        ax.plot([], [], color=color, linewidth=3, label=label, alpha=0.8)
    
    def _draw_drone_missions(self, ax, schedule: DroneFleetSchedule, color, label):
        """绘制无人机任务"""
        for i, trip in enumerate(schedule.list_of_drone_trips):
            if len(trip.sequence_of_points) < 2:
                continue
            
            x_coords = []
            y_coords = []
            
            for point in trip.sequence_of_points:
                if point == 'depot':
                    x_coords.append(self.depot.x_coord)
                    y_coords.append(self.depot.y_coord)
                else:
                    pickup_point = self.pickup_points[point]
                    x_coords.append(pickup_point.x_coord)
                    y_coords.append(pickup_point.y_coord)
            
            # 绘制直线路径
            ax.plot(x_coords, y_coords, color=color, linewidth=2, 
                   linestyle='--', alpha=0.7, marker='o', markersize=4)
            
            # 添加箭头
            for j in range(len(x_coords) - 1):
                mid_x = (x_coords[j] + x_coords[j+1]) / 2
                mid_y = (y_coords[j] + y_coords[j+1]) / 2
                dx = x_coords[j+1] - x_coords[j]
                dy = y_coords[j+1] - y_coords[j]
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                           xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        
        # 添加图例项（只添加一次）
        if schedule.list_of_drone_trips:
            ax.plot([], [], color=color, linewidth=2, linestyle='--', 
                   marker='o', markersize=4, label=label, alpha=0.7)
    
    def _format_axis(self, ax):
        """格式化坐标轴"""
        ax.set_xlabel('X Coordinate (km)', fontweight='bold')
        ax.set_ylabel('Y Coordinate (km)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # 设置坐标轴范围
        all_x = [self.depot.x_coord] + [p.x_coord for p in self.pickup_points.values()]
        all_y = [self.depot.y_coord] + [p.y_coord for p in self.pickup_points.values()]
        
        margin = 5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    def save_figure(self, fig, filename: str = "vrp_solution.png"):
        """保存图像"""
        fig.savefig(filename, dpi=config.DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图像已保存为: {filename}")

def visualize_solution(depot: Depot, pickup_points: dict, solution: Solution, 
                      title: str = "ALNS-Based Truck-Drone Collaborative VRP Solution",
                      filename: str = "vrp_solution.png"):
    """可视化解决方案的便捷函数"""
    visualizer = VRPVisualizer(depot, pickup_points)
    fig = visualizer.plot_solution(solution, title)
    visualizer.save_figure(fig, filename)
    # plt.show()
    return fig

def create_comprehensive_analysis(depot: Depot, pickup_points: dict, solution: Solution,
                                title: str = "ALNS-Based Truck-Drone Collaborative VRP Solution",
                                save_prefix: str = "vrp_comprehensive"):
    """创建综合分析可视化，生成多个详细的分析图片"""
    visualizer = VRPVisualizer(depot, pickup_points)
    
    print("🎨 生成综合可视化分析...")
    
    # 生成所有分析图
    figures = visualizer.create_comprehensive_visualization(solution, title, save_prefix)
    
    # 打印生成的文件信息
    print(f"✅ 已生成 {len(figures)} 个分析图片:")
    print(f"   📊 {save_prefix}_main_solution.png - 主要解决方案图")
    print(f"   🚛 {save_prefix}_transport_modes.png - 运输方式分析图")
    print(f"   📈 {save_prefix}_detailed_stats.png - 详细统计分析图")
    
    return figures

def create_individual_visualizations(depot: Depot, pickup_points: dict, solution: Solution,
                                   save_prefix: str = "individual_viz"):
    """创建所有独立的可视化图片"""
    visualizer = VRPVisualizer(depot, pickup_points)
    
    print("🎨 生成独立的可视化图片...")
    
    # 生成所有独立的图片
    figures = []
    
    # 主要解决方案相关图片
    figures.append(visualizer.plot_individual_solution_overview(
        solution, "Complete Solution Overview", f"{save_prefix}_solution_overview.png"))
    
    figures.append(visualizer.plot_individual_truck_routes(
        solution, "Truck Routes", f"{save_prefix}_truck_routes.png"))
    
    figures.append(visualizer.plot_individual_drone_missions(
        solution, "Drone Missions", f"{save_prefix}_drone_missions.png"))
    
    figures.append(visualizer.plot_individual_performance_metrics(
        solution, "Performance Metrics", f"{save_prefix}_performance_metrics.png"))
    
    # 运输方式分析相关图片
    figures.append(visualizer.plot_individual_transport_mode_map(
        solution, "Transportation Mode Distribution", f"{save_prefix}_transport_mode_map.png"))
    
    figures.append(visualizer.plot_individual_transport_mode_pie(
        solution, "Transportation Mode Statistics", f"{save_prefix}_transport_mode_pie.png"))
    
    figures.append(visualizer.plot_individual_demand_allocation(
        solution, "Demand Allocation Analysis", f"{save_prefix}_demand_allocation.png"))
    
    figures.append(visualizer.plot_individual_collaborative_details(
        solution, "Collaborative Transportation Details", f"{save_prefix}_collaborative_details.png"))
    
    # 详细统计相关图片
    figures.append(visualizer.plot_individual_vehicle_utilization(
        solution, "Vehicle Utilization Analysis", f"{save_prefix}_vehicle_utilization.png"))
    
    figures.append(visualizer.plot_individual_time_analysis(
        solution, "Time Distribution Analysis", f"{save_prefix}_time_analysis.png"))
    
    figures.append(visualizer.plot_individual_cost_breakdown(
        solution, "Cost Structure Analysis", f"{save_prefix}_cost_breakdown.png"))
    
    figures.append(visualizer.plot_individual_distance_efficiency(
        solution, "Distance & Efficiency Analysis", f"{save_prefix}_distance_efficiency.png"))
    
    # 打印生成的文件信息
    print(f"✅ 已生成 {len(figures)} 个独立的可视化图片")
    
    return figures

if __name__ == "__main__":
    from vrp_parser import parse_vrp_file
    from alns_solver import solve_vrp_with_alns
    import random
    
    # 设置随机种子
    random.seed(config.RANDOM_SEED)
    
    # 解析问题并求解
    parsed_result = parse_vrp_file("real.vrp")
    if parsed_result is not None:
        depot, pickup_points = parsed_result
        if depot is not None:
            solution = solve_vrp_with_alns(depot, pickup_points)
            
            # 可视化结果
            visualize_solution(depot, pickup_points, solution)
        else:
            print("无法解析到有效的仓库信息")
    else:
        print("无法解析VRP文件")