# -*- coding: utf-8 -*-
from pathlib import Path
import re
from collections import defaultdict
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


def plot_boxplots(cost_series, makespan_series):
    # 将无人机路程排序以保持箱型图顺序一致
    distances = sorted(cost_series)
    cost_data = [cost_series[distance] for distance in distances]
    makespan_data = [makespan_series[distance] for distance in distances]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    
    axes[0].boxplot(cost_data, tick_labels=distances, patch_artist=True)
    axes[0].set_title('Cost Sensitivity by Drone Range')
    axes[0].set_xlabel('Drone Range')
    axes[0].set_ylabel('Total Cost (CNY)')
    
    axes[1].boxplot(makespan_data, tick_labels=distances, patch_artist=True)
    axes[1].set_title('Makespan Sensitivity by Drone Range')
    axes[1].set_xlabel('Drone Range')
    axes[1].set_ylabel('Makespan (hours)')
    
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('distance_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 解析文本并绘制箱型图
    records = parse_sensitivity(DATA_FILE)
    if not records:
        raise ValueError('No records were parsed from d_distance.txt')
    
    cost_series, makespan_series = build_series(records)
    plot_boxplots(cost_series, makespan_series)


if __name__ == '__main__':
    main()