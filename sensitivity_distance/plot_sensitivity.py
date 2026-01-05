# -*- coding: utf-8 -*-
from pathlib import Path
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE = Path('d_num.txt')


def parse_sensitivity(file_path: Path):
    """解析敏感性结果文本为结构化记录。"""
    # 使用字典暂存当前条目的字段
    current = {}
    records = []
    drone_pattern = re.compile(r"使用无人机:\s*(\d+)")
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
        elif set(current) == {'drones', 'cost', 'makespan'}:
            # 遇到分隔线时保存完整记录
            records.append(current)
            current = {}

    # 处理末尾未被分隔线捕获的记录
    if set(current) == {'drones', 'cost', 'makespan'}:
        records.append(current)

    return records


def build_series(records):
    # 根据无人机数量聚合成本和时长
    cost_series = defaultdict(list)
    makespan_series = defaultdict(list)

    for record in records:
        drones = record['drones']
        cost_series[drones].append(record['cost'])
        makespan_series[drones].append(record['makespan'])

    return cost_series, makespan_series


def plot_boxplots(cost_series, makespan_series):
    """绘制箱型图展示成本和makespan随无人机数量的变化"""
    # 将无人机数量排序以保持箱型图顺序一致
    drone_counts = sorted(cost_series)
    cost_data = [cost_series[count] for count in drone_counts]
    makespan_data = [makespan_series[count] for count in drone_counts]

    # 计算中位数用于趋势线
    cost_medians = [np.median(data) for data in cost_data]
    makespan_medians = [np.median(data) for data in makespan_data]

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ========== 绘制成本箱型图 ==========
    bp1 = axes[0].boxplot(cost_data, positions=drone_counts, patch_artist=True, widths=0.6,
                          boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                          medianprops=dict(color='#1A5276', linewidth=2),
                          whiskerprops=dict(color='#2E86AB', linewidth=1.5),
                          capprops=dict(color='#2E86AB', linewidth=1.5),
                          flierprops=dict(marker='o', markerfacecolor='#2E86AB',
                                        markersize=6, alpha=0.5))

    # 添加中位数趋势线
    axes[0].plot(drone_counts, cost_medians, 'o-', color='#FF6B35', linewidth=2.5,
                markersize=7, label='Median Trend', zorder=10, alpha=0.8)

    # # 标注关键转折点：0→1的成本下降
    # if 0 in drone_counts and 1 in drone_counts:
    #     idx_0 = drone_counts.index(0)
    #     idx_1 = drone_counts.index(1)
    #     axes[0].annotate('', xy=(1, cost_medians[idx_1]), xytext=(0, cost_medians[idx_0]),
    #                     arrowprops=dict(arrowstyle='->', color='green', lw=2.5, alpha=0.7))
    #     axes[0].text(0.5, (cost_medians[idx_0] + cost_medians[idx_1])/2,
    #                 'Cost\nReduction', fontsize=9, ha='center',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.6))

    # 找到成本最低点（最优平衡点）
    min_cost_idx = cost_medians.index(min(cost_medians))
    min_cost_drones = drone_counts[min_cost_idx]
    axes[0].scatter([min_cost_drones], [cost_medians[min_cost_idx]],
                   color='red', s=200, zorder=15, marker='*',
                   edgecolors='darkred', linewidths=2, label='Optimal Point')

    axes[0].set_title('Cost Sensitivity by Drone Count', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Drones', fontsize=12)
    axes[0].set_ylabel('Total Cost (CNY)', fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_xticks(drone_counts)

    # ========== 绘制makespan箱型图 ==========
    bp2 = axes[1].boxplot(makespan_data, positions=drone_counts, patch_artist=True, widths=0.6,
                          boxprops=dict(facecolor='#A23B72', alpha=0.7),
                          medianprops=dict(color='#6B1E4A', linewidth=2),
                          whiskerprops=dict(color='#A23B72', linewidth=1.5),
                          capprops=dict(color='#A23B72', linewidth=1.5),
                          flierprops=dict(marker='s', markerfacecolor='#A23B72',
                                        markersize=6, alpha=0.5))

    # 添加中位数趋势线
    axes[1].plot(drone_counts, makespan_medians, 's-', color='#FF6B35', linewidth=2.5,
                markersize=7, label='Median Trend', zorder=10, alpha=0.8)

    # 标注0→1的makespan变化（可能增加）
    if 0 in drone_counts and 1 in drone_counts:
        idx_0 = drone_counts.index(0)
        idx_1 = drone_counts.index(1)
        if makespan_medians[idx_1] > makespan_medians[idx_0]:
            axes[1].annotate('', xy=(1, makespan_medians[idx_1]),
                           xytext=(0, makespan_medians[idx_0]),
                           arrowprops=dict(arrowstyle='->', color='orange', lw=2.5, alpha=0.7))
            axes[1].text(0.5, (makespan_medians[idx_0] + makespan_medians[idx_1])/2,
                       'Makespan\nIncrease', fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.6))

    # 找到makespan最低点
    min_makespan_idx = makespan_medians.index(min(makespan_medians))
    min_makespan_drones = drone_counts[min_makespan_idx]
    axes[1].scatter([min_makespan_drones], [makespan_medians[min_makespan_idx]],
                   color='red', s=200, zorder=15, marker='*',
                   edgecolors='darkred', linewidths=2, label='Optimal Point')

    axes[1].set_title('Makespan Sensitivity by Drone Count', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Drones', fontsize=12)
    axes[1].set_ylabel('Makespan (hours)', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].set_xticks(drone_counts)

    fig.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 解析文本并绘制箱型图
    records = parse_sensitivity(DATA_FILE)
    if not records:
        raise ValueError('No records were parsed from se.txt')

    cost_series, makespan_series = build_series(records)
    plot_boxplots(cost_series, makespan_series)


if __name__ == '__main__':
    main()
