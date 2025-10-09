import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import PercentFormatter

def load_solution_data(file_path):
    """加载解决方案数据"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_performance_metrics_chart(solution_data):
    """Create performance metrics chart"""
    # Extract performance metrics
    metrics = {
        'Total Time': solution_data['metadata']['total_makespan'],
        'Total Cost': solution_data['metadata']['total_operating_cost'],
        'Evaluation Score': solution_data['metadata']['evaluation_score']
    }
    
    # Create figure with academic style
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('default')
    
    # Create bar chart
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Set title and labels with academic style
    plt.title('Solution Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Value', fontsize=12, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=10)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("Performance metrics chart saved as 'performance_metrics.png'")
    plt.close()

def create_vehicle_usage_chart(solution_data):
    """Create vehicle usage chart"""
    # Extract vehicle usage data
    truck_count = solution_data['summary']['active_trucks']
    drone_count = solution_data['summary']['active_drones']
    
    # Create figure with academic style
    plt.figure(figsize=(8, 8), dpi=300)
    plt.style.use('default')
    
    # Data and labels
    sizes = [truck_count, drone_count]
    labels = ['Trucks', 'Drones']
    colors = ['#1f77b4', '#ff7f0e']
    explode = (0.05, 0.05)  # Highlight slices
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Set title with academic style
    plt.title('Vehicle Usage Proportion', fontsize=16, fontweight='bold', pad=20)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('vehicle_usage.png', dpi=300, bbox_inches='tight')
    print("Vehicle usage chart saved as 'vehicle_usage.png'")
    plt.close()

def create_distance_comparison_chart(solution_data):
    """Create distance comparison chart"""
    # Extract distance data
    truck_distance = solution_data['summary']['total_truck_distance']
    drone_distance = solution_data['summary']['total_drone_distance']
    total_distance = truck_distance + drone_distance
    
    # Calculate percentages
    truck_percent = truck_distance / total_distance * 100
    drone_percent = drone_distance / total_distance * 100
    
    # Create figure with academic style
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('default')
    
    # Data and labels
    categories = ['Trucks', 'Drones']
    distances = [truck_distance, drone_distance]
    colors = ['#1f77b4', '#ff7f0e']
    
    # Create horizontal bar chart
    bars = plt.barh(categories, distances, color=colors)
    
    # Add value labels
    for i, (bar, distance, percent) in enumerate(zip(bars, distances, [truck_percent, drone_percent])):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{distance:.1f} km ({percent:.1f}%)',
                va='center', fontweight='bold', fontsize=10)
    
    # Set title and labels with academic style
    plt.title('Distance Comparison: Trucks vs. Drones', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Distance (km)', fontsize=12, fontweight='bold')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=11, fontweight='bold')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('distance_comparison.png', dpi=300, bbox_inches='tight')
    print("Distance comparison chart saved as 'distance_comparison.png'")
    plt.close()

def create_truck_routes_analysis(solution_data):
    """Create truck routes analysis chart"""
    # Extract truck routes data
    truck_routes = solution_data['truck_routes']
    
    # Prepare data
    truck_ids = [route['truck_id'] for route in truck_routes]
    distances = [route['total_distance'] for route in truck_routes]
    times = [route['total_time'] for route in truck_routes]
    costs = [route['total_cost'] for route in truck_routes]
    
    # Create figure with academic style
    plt.figure(figsize=(12, 10), dpi=300)
    plt.style.use('default')
    
    # Create subplots
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    
    # Distance bar chart
    bars1 = ax1.bar(truck_ids, distances, color='#1f77b4')
    ax1.set_title('Distance by Truck Route', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Distance (km)', fontsize=12, fontweight='bold')
    ax1.set_xticks(truck_ids)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Time bar chart
    bars2 = ax2.bar(truck_ids, times, color='#ff7f0e')
    ax2.set_title('Time by Truck Route', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_xticks(truck_ids)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Cost bar chart
    bars3 = ax3.bar(truck_ids, costs, color='#2ca02c')
    ax3.set_title('Cost by Truck Route', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cost', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Truck ID', fontsize=12, fontweight='bold')
    ax3.set_xticks(truck_ids)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('truck_routes_analysis.png', dpi=300, bbox_inches='tight')
    print("Truck routes analysis chart saved as 'truck_routes_analysis.png'")
    plt.close()

def create_drone_analysis_chart(solution_data):
    """Create drone analysis chart"""
    # Extract drone data
    drone_schedules = solution_data['drone_schedules']
    
    # Prepare data
    drone_ids = [schedule['drone_id'] for schedule in drone_schedules]
    times = [schedule['total_time'] for schedule in drone_schedules]
    costs = [schedule['total_cost'] for schedule in drone_schedules]
    trip_counts = [len(schedule['trips']) for schedule in drone_schedules]
    
    # Create figure with academic style
    plt.figure(figsize=(12, 10), dpi=300)
    plt.style.use('default')
    
    # Create subplots
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    
    # Time bar chart
    bars1 = ax1.bar(drone_ids, times, color='#1f77b4')
    ax1.set_title('Total Operating Time by Drone', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_xticks(drone_ids)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Cost bar chart
    bars2 = ax2.bar(drone_ids, costs, color='#ff7f0e')
    ax2.set_title('Total Cost by Drone', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cost', fontsize=12, fontweight='bold')
    ax2.set_xticks(drone_ids)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Trip count bar chart
    bars3 = ax3.bar(drone_ids, trip_counts, color='#2ca02c')
    ax3.set_title('Number of Trips by Drone', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Trips', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Drone ID', fontsize=12, fontweight='bold')
    ax3.set_xticks(drone_ids)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('drone_analysis.png', dpi=300, bbox_inches='tight')
    print("Drone analysis chart saved as 'drone_analysis.png'")
    plt.close()

def create_demand_satisfaction_chart(solution_data):
    """Create demand satisfaction chart"""
    # Extract demand data
    total_demand = solution_data['summary']['total_demand_satisfied']
    pickup_points = solution_data['summary']['total_pickup_points_served']
    
    # Create figure with academic style
    plt.figure(figsize=(14, 6), dpi=300)
    plt.style.use('default')
    
    # Create subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Demand satisfaction
    ax1.text(0.5, 0.5, f'Total Demand\nSatisfied\n{total_demand}', 
             ha='center', va='center', fontsize=20, fontweight='bold',
             transform=ax1.transAxes)
    ax1.set_title('Demand Satisfaction', fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Service points
    ax2.text(0.5, 0.5, f'Number of\nService Points\n{pickup_points}', 
             ha='center', va='center', fontsize=20, fontweight='bold',
             transform=ax2.transAxes)
    ax2.set_title('Service Points', fontsize=16, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('demand_satisfaction.png', dpi=300, bbox_inches='tight')
    print("Demand satisfaction chart saved as 'demand_satisfaction.png'")
    plt.close()

def generate_all_charts(solution_file):
    """Generate all analysis charts"""
    # Load data
    solution_data = load_solution_data(solution_file)
    
    # Generate all charts
    create_performance_metrics_chart(solution_data)
    create_vehicle_usage_chart(solution_data)
    create_distance_comparison_chart(solution_data)
    create_truck_routes_analysis(solution_data)
    create_drone_analysis_chart(solution_data)
    create_demand_satisfaction_chart(solution_data)
    
    print("\nAll analysis charts have been generated successfully!")

if __name__ == "__main__":
    solution_file = "solution_data_20250928_105444.json"
    generate_all_charts(solution_file)