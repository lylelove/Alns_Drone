import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_drone_sensitivity(csv_path):
    """
    分析无人机数量敏感性数据并生成图表。

    Args:
        csv_path (str): 存放敏感性分析数据的CSV文件路径。
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return

    # 使用pandas读取CSV数据
    data = pd.read_csv(csv_path)

    # 提取文件名用于保存图片，不包含扩展名
    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
    
    # 创建一个2x2的子图布局，用于展示不同的分析维度
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Sensitivity Analysis of Maximum Number of Drones', fontsize=16)

    # x轴数据
    max_drones = data['max_drones']

    # --- 子图1: 目标函数分析 ---
    axs[0, 0].plot(max_drones, data['final_objective'], marker='o', linestyle='-', label='Final Objective')
    axs[0, 0].plot(max_drones, data['final_makespan'], marker='s', linestyle='--', label='Final Makespan')
    axs[0, 0].plot(max_drones, data['final_cost'], marker='^', linestyle=':', label='Final Cost')
    axs[0, 0].set_title('Objective Function Analysis')
    axs[0, 0].set_xlabel('Maximum Number of Drones')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- 子图2: 性能分析 ---
    ax2_perf = axs[0, 1]
    ax2_time = ax2_perf.twinx() # 创建双Y轴
    
    line1, = ax2_perf.plot(max_drones, data['improvement_rate'], marker='o', linestyle='-', color='g', label='Improvement Rate')
    ax2_perf.set_xlabel('Maximum Number of Drones')
    ax2_perf.set_ylabel('Improvement Rate (%)', color='g')
    ax2_perf.tick_params(axis='y', labelcolor='g')

    line2, = ax2_time.plot(max_drones, data['solve_time'], marker='s', linestyle='--', color='b', label='Solve Time')
    ax2_time.set_ylabel('Solve Time (s)', color='b')
    ax2_time.tick_params(axis='y', labelcolor='b')
    
    ax2_perf.set_title('Performance Analysis')
    ax2_perf.legend(handles=[line1, line2], loc='best')
    ax2_perf.grid(True)

    # --- 子图3: 资源使用情况分析 ---
    axs[1, 0].plot(max_drones, data['active_trucks'], marker='o', linestyle='-', label='Active Trucks')
    axs[1, 0].plot(max_drones, data['active_drones'], marker='s', linestyle='--', label='Active Drones')
    axs[1, 0].set_title('Resource Usage Analysis')
    axs[1, 0].set_xlabel('Maximum Number of Drones')
    axs[1, 0].set_ylabel('Number of Vehicles')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- 子图4: 行驶距离分析 ---
    axs[1, 1].plot(max_drones, data['total_distance'], marker='o', linestyle='-', label='Total Distance')
    axs[1, 1].plot(max_drones, data['total_truck_distance'], marker='s', linestyle='--', label='Total Truck Distance')
    axs[1, 1].plot(max_drones, data['total_drone_distance'], marker='^', linestyle=':', label='Total Drone Distance')
    axs[1, 1].set_title('Distance Analysis')
    axs[1, 1].set_xlabel('Maximum Number of Drones')
    axs[1, 1].set_ylabel('Distance')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图表
    output_image_path = os.path.join(os.path.dirname(csv_path), f'{base_filename}_analysis.png')
    plt.savefig(output_image_path)
    print(f"分析图表已保存至: {output_image_path}")

    # 显示图表
    plt.show()


if __name__ == '__main__':
    # 指定要分析的数据文件路径
    # 请确保这个文件存在于脚本运行的目录下的 'results' 文件夹中
    csv_file_path = 'results/max_drones_sensitivity_data_20250625_154700.csv'
    
    # 调用分析函数
    analyze_drone_sensitivity(csv_file_path) 