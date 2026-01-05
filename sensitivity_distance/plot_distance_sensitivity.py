# -*- coding: utf-8 -*-
from pathlib import Path
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE = Path('d_distance.txt')


def parse_sensitivity(file_path: Path):
    """解析无人机路程敏感性结果文本为结构化记录。"""
    # 使用字典暂存当前条目的字段
    current = {}
    records = []
    drone_pattern = re.compile(r"使用无人机:\s*(\d+)")
    distance_pattern = re.compile(r"无人机路程：\s*(\d+)")
    cost_pattern = re.compile(r"总成本:\s*([\d.]+)")
    makespan_pattern = re.compile(r"Makespan:\s*([\d.]+)")
    
    for raw_line in file_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('使用无人机'):
            # 提取无人机数量
            match = drone_pattern.search(line)
            if match:
                current['drones'] = int(match.group(1))
        elif line.startswith('无人机路程'):
            # 提取无人机路程
            match = distance_pattern.search(line)
            if match:
                current['distance'] = int(match.group(1))
        elif line.startswith('总成本'):
            # 提取成本数值
            match = cost_pattern.search(line)
            if match:
                current['cost'] = float(match.group(1))
        elif line.startswith('Makespan'):
            # 提取时长数值
            match = makespan_pattern.search(line)
            if match:
                current['makespan'] = float(match.group(1))
        elif set(current) == {'drones', 'distance', 'cost', 'makespan'}:
            # 遇到分隔线时保存完整记录
            records.append(current)
            current = {}
    
    # 处理末尾未被分隔线捕获的记录
    if set(current) == {'drones', 'distance', 'cost', 'makespan'}:
        records.append(current)
    
    return records


def build_series(records):
    # 根据无人机路程聚合成本和时长
    cost_series = defaultdict(list)
    makespan_series = defaultdict(list)
    
    for record in records:
        distance = record['distance']
        cost_series[distance].append(record['cost'])
        makespan_series[distance].append(record['makespan'])
    
    return cost_series, makespan_series


def plot_line_charts(cost_series, makespan_series):
    """绘制折线图展示成本和makespan随无人机范围的变化趋势"""
    # 将无人机路程排序
    distances = sorted(cost_series)

    # 计算每个距离下的均值和标准差
    cost_means = [np.mean(cost_series[d]) for d in distances]
    cost_stds = [np.std(cost_series[d]) for d in distances]
    makespan_means = [np.mean(makespan_series[d]) for d in distances]
    makespan_stds = [np.std(makespan_series[d]) for d in distances]

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制成本趋势图
    axes[0].plot(distances, cost_means, 'o-', linewidth=2, markersize=8,
                 color='#2E86AB', label='Mean Cost')
    axes[0].fill_between(distances,
                         np.array(cost_means) - np.array(cost_stds),
                         np.array(cost_means) + np.array(cost_stds),
                         alpha=0.3, color='#2E86AB', label='±1 Std Dev')
    axes[0].set_title('Cost Sensitivity by Drone Range', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Drone Range (km)', fontsize=12)
    axes[0].set_ylabel('Total Cost (CNY)', fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(loc='best')

    # 标注最优点(如果15km在数据中)
    if 15 in distances:
        idx = distances.index(15)
        axes[0].axvline(x=15, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        axes[0].scatter([15], [cost_means[idx]], color='red', s=150, zorder=5,
                       marker='*', edgecolors='darkred', linewidths=1.5)

    # 绘制makespan趋势图
    axes[1].plot(distances, makespan_means, 's-', linewidth=2, markersize=8,
                 color='#A23B72', label='Mean Makespan')
    axes[1].fill_between(distances,
                         np.array(makespan_means) - np.array(makespan_stds),
                         np.array(makespan_means) + np.array(makespan_stds),
                         alpha=0.3, color='#A23B72', label='±1 Std Dev')
    axes[1].set_title('Makespan Sensitivity by Drone Range', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Drone Range (km)', fontsize=12)
    axes[1].set_ylabel('Makespan (hours)', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='best')

    # 标注最优点(如果15km在数据中)
    if 15 in distances:
        idx = distances.index(15)
        axes[1].axvline(x=15, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        axes[1].scatter([15], [makespan_means[idx]], color='red', s=150, zorder=5,
                       marker='*', edgecolors='darkred', linewidths=1.5)

    fig.tight_layout()
    plt.savefig('distance_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 解析文本并绘制折线图
    records = parse_sensitivity(DATA_FILE)
    if not records:
        raise ValueError('No records were parsed from d_distance.txt')

    cost_series, makespan_series = build_series(records)
    plot_line_charts(cost_series, makespan_series)


if __name__ == '__main__':
    main()