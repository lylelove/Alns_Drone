# -*- coding: utf-8 -*-
from pathlib import Path
import re
from collections import defaultdict
import matplotlib.pyplot as plt


DATA_FILE = Path('se.txt')


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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    axes[0].boxplot(cost_data, tick_labels=drone_counts, patch_artist=True)
    axes[0].set_title('Cost Sensitivity by Drone Count')
    axes[0].set_xlabel('Number of Drones')
    axes[0].set_ylabel('Total Cost (CNY)')

    axes[1].boxplot(makespan_data, tick_labels=drone_counts, patch_artist=True)
    axes[1].set_title('Makespan Sensitivity by Drone Count')
    axes[1].set_xlabel('Number of Drones')
    axes[1].set_ylabel('Makespan (hours)')

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
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
