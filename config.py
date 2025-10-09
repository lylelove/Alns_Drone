# 配置参数文件
# 所有算法参数统一管理

# 车辆参数
TRUCK_CAPACITY = 100  # 卡车容量
TRUCK_SPEED = 40  # 卡车速度 (km/h)
TRUCK_COST_PER_KM = 1.2  # 卡车运输成本 (元/公里)
TRUCK_STARTUP_COST = 100  # 卡车启动费用 (元)
TRUCK_LOADING_TIME = 0.5  # 卡车装载时间 (小时)

DRONE_CAPACITY = 5  # 无人机容量
DRONE_SPEED = 40  # 无人机速度 (km/h) 
DRONE_COST_PER_KM = 0.1  # 无人机运输成本 (元/公里)
DRONE_STARTUP_COST = 20  # 无人机启动费用 (元)
DRONE_RADIUS = 15  # 无人机运输半径 (km)
DRONE_MAX_DISTANCE = 30  # 无人机单次飞行最大距离 (km)
MAX_DRONES = 2  # 最大无人机数量

# ALNS算法参数
INITIAL_TEMPERATURE = 2000.0  # 初始温度（降低用于测试）
MIN_TEMPERATURE = 1.0  # 最小温度
COOLING_RATE = 0.997  # 冷却率
MAX_ITERATIONS = 2000  # 最大迭代次数（降低用于测试）
MAX_NO_IMPROVEMENT = 2000  # 最大无改进迭代次数（降低用于测试）

# 操作符权重相关
INITIAL_OPERATOR_WEIGHT = 1.0  # 初始操作符权重
WEIGHT_UPDATE_FREQUENCY = 50  # 权重更新频率（降低用于测试）
IMPROVEMENT_REWARD = 1.2  # 改进奖励因子
NEUTRAL_REWARD = 1.0  # 中性奖励因子
WORSENING_PENALTY = 0.8  # 恶化惩罚因子
OPERATOR_EPSILON = 0.08  # ε-greedy探索概率，防止算子过早固化
WEIGHT_SMOOTHING_RHO = 0.2  # 权重平滑系数，降低抖动与早熟



# 动态参考值参数（基于初始解的比例）
MAKESPAN_REFERENCE_RATIO = 0.5  # Makespan参考值
COST_REFERENCE_RATIO = 0.5  # 成本参考值

# 预设优化场景（可选择性使用）
OPTIMIZATION_SCENARIOS = {
    'balanced': {
        'makespan_ratio': 1.0,
        'cost_ratio': 1.0,
        'description': '平衡优化：时间和成本并重'
    },
    'time_priority': {
        'makespan_ratio': 0.8,
        'cost_ratio': 1.0,
        'description': '时间优先：加大时间优化压力'
    },
    'cost_priority': {
        'makespan_ratio': 1.0,
        'cost_ratio': 0.8,
        'description': '成本优先：加大成本优化压力'
    },
    'strict_optimization': {
        'makespan_ratio': 0.7,
        'cost_ratio': 0.7,
        'description': '严格优化：对时间和成本都要求较高'
    },
    'relaxed_optimization': {
        'makespan_ratio': 1.3,
        'cost_ratio': 1.3,
        'description': '宽松优化：允许相对较大的时间和成本'
    }
}

# 用于运行时计算的参考值（将由算法初始化时动态设置）
MAKESPAN_REFERENCE = None  # 运行时动态计算
COST_REFERENCE = None      # 运行时动态计算

BASE_MAKESPAN_WEIGHT = 0.6  # 基础时间权重 (根据算法设计文档，略偏向 Makespan)
BASE_COST_WEIGHT = 0.4      # 基础成本权重
IMBALANCE_THRESHOLD = 1.2   # 不平衡阈值
BALANCE_BONUS_THRESHOLD = 0.8  # 平衡奖励阈值
BALANCE_BONUS_FACTOR = 0.95    # 平衡奖励因子（5%奖励）

# 目标函数模式选择
USE_IMPROVED_OBJECTIVE = True  # 是否使用改进的目标函数

# 接受准则参数
MAX_ALLOWED_COST_INCREASE_FACTOR = 0.1  # 最大允许成本增加因子

# 破坏操作参数
MIN_DESTROY_SIZE = 3  # 最小破坏规模
MAX_DESTROY_SIZE = 10  # 最大破坏规模

# 随机种子
RANDOM_SEED = 42

# 算法常量
GEO_SIMILARITY_WEIGHT = 0.7  # Shaw移除中地理相似度权重
DEMAND_SIMILARITY_WEIGHT = 0.3  # Shaw移除中需求相似度权重
NUMERICAL_STABILITY_EPSILON = 1e-6  # 数值稳定性保护常量

# 停滞缓解（避免过快早熟收敛）
STAGNATION_SHAKE_THRESHOLD = 200  # 超过该无改进次数后触发shake
SHAKE_DESTROY_FRACTION = 0.5      # shake时一次破坏规模占比
REHEAT_FACTOR = 0.3               # 复温比例（相对初温）

# 可视化参数
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 10
DPI = 400

def set_dynamic_reference_values(initial_makespan: float, initial_cost: float):
    """
    基于初始解设置动态参考值
    Args:
        initial_makespan: 初始解的makespan值
        initial_cost: 初始解的成本值
    """
    global MAKESPAN_REFERENCE, COST_REFERENCE
    MAKESPAN_REFERENCE = initial_makespan * MAKESPAN_REFERENCE_RATIO
    COST_REFERENCE = initial_cost * COST_REFERENCE_RATIO
    
    print(f"设置动态参考值:")
    print(f"  Makespan参考值: {MAKESPAN_REFERENCE:.2f} (初始值: {initial_makespan:.2f}, 比例: {MAKESPAN_REFERENCE_RATIO})")
    print(f"  成本参考值: {COST_REFERENCE:.2f} (初始值: {initial_cost:.2f}, 比例: {COST_REFERENCE_RATIO})")

def get_reference_values():
    """获取当前的参考值"""
    return MAKESPAN_REFERENCE, COST_REFERENCE

def apply_optimization_scenario(scenario_name: str):
    """
    应用预设的优化场景
    Args:
        scenario_name: 场景名称，必须是OPTIMIZATION_SCENARIOS中的键
    """
    global MAKESPAN_REFERENCE_RATIO, COST_REFERENCE_RATIO
    
    if scenario_name not in OPTIMIZATION_SCENARIOS:
        available_scenarios = list(OPTIMIZATION_SCENARIOS.keys())
        raise ValueError(f"未知的优化场景: {scenario_name}. 可用场景: {available_scenarios}")
    
    scenario = OPTIMIZATION_SCENARIOS[scenario_name]
    MAKESPAN_REFERENCE_RATIO = scenario['makespan_ratio']
    COST_REFERENCE_RATIO = scenario['cost_ratio']
    
    print(f"应用优化场景: {scenario_name}")
    print(f"  描述: {scenario['description']}")
    print(f"  Makespan比例: {MAKESPAN_REFERENCE_RATIO}")
    print(f"  成本比例: {COST_REFERENCE_RATIO}")

def list_optimization_scenarios():
    """列出所有可用的优化场景"""
    print("可用的优化场景:")
    for name, scenario in OPTIMIZATION_SCENARIOS.items():
        print(f"  {name}: {scenario['description']}")
        print(f"    - Makespan比例: {scenario['makespan_ratio']}")
        print(f"    - 成本比例: {scenario['cost_ratio']}")

def set_custom_reference_ratios(makespan_ratio: float, cost_ratio: float):
    """
    设置自定义的参考值比例
    Args:
        makespan_ratio: Makespan参考值比例
        cost_ratio: 成本参考值比例
    """
    global MAKESPAN_REFERENCE_RATIO, COST_REFERENCE_RATIO
    
    if makespan_ratio <= 0 or cost_ratio <= 0:
        raise ValueError("参考值比例必须为正数")
    
    MAKESPAN_REFERENCE_RATIO = makespan_ratio
    COST_REFERENCE_RATIO = cost_ratio
    
    print(f"设置自定义参考值比例:")
    print(f"  Makespan比例: {MAKESPAN_REFERENCE_RATIO}")
    print(f"  成本比例: {COST_REFERENCE_RATIO}")
