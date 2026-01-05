
# -*- coding: utf-8 -*-
from pathlib import Path
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


DATA_FILE = Path('d_num.txt')

# 设置中文字体和全局样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'


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
    # 将无人机数量排序以保持箱型图顺序一致
    drone_counts = sorted(cost_series)
    cost_data = [cost_series[count] for count in drone_counts]
    makespan_data = [makespan_series[count] for count in drone_counts]

    # 创建更美观的图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    fig.suptitle('无人机数量敏感性分析', fontsize=18, fontweight='bold', y=0.98)
    
    # 定义颜色方案
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#34495e', '#f1c40f']
    
    # 绘制成本箱型图
    bp_cost = axes[0].boxplot(cost_data, tick_labels=drone_counts, patch_artist=True, 
                             boxprops=dict(linewidth=1.5), 
                             whiskerprops=dict(linewidth=1.5, linestyle='--'),
                             capprops=dict(linewidth=1.5),
                             medianprops=dict(linewidth=2, color='black'))
    
    # 为成本箱型图添加颜色
    for patch, color in zip(bp_cost['boxes'], colors[:len(drone_counts)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_title('成本敏感性分析', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('无人机数量', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('总成本 (元)', fontsize=12, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.7, axis='y')
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    # 绘制时间箱型图
    bp_makespan = axes[1].boxplot(makespan_data, tick_labels=drone_counts, patch_artist=True,
                                 boxprops=dict(linewidth=1.5),
                                 whiskerprops=dict(linewidth=1.5, linestyle='--'),
                                 capprops=dict(linewidth=1.5),
                                 medianprops=dict(linewidth=2, color='black'))
    
    # 为时间箱型图添加颜色
    for patch, color in zip(bp_makespan['boxes'], colors[:len(drone_counts)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_title('完成时间敏感性分析', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('无人机数量', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('完成时间 (小时)', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.7, axis='y')
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    # 添加数据点数量标注
    for i, (ax, data) in enumerate(zip(axes, [cost_data, makespan_data])):
        for j, count in enumerate(drone_counts):
            ax.text(j+1, ax.get_ylim()[1]*0.95, f'n={len(data[j])}',
                   ha='center', va='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 调整布局
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
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