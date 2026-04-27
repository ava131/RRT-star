import numpy as np

# --- 窗口与基础设置 ---
WINDOW_SIZE = 700
CENTER = (WINDOW_SIZE // 2, WINDOW_SIZE // 2)

# --- 机械臂物理参数 ---
LINK_LENGTHS = [100, 100, 80]
START_CFG = np.array([45.0, -30.0, -15.0])
GOAL_CFG = np.array([135.0, -30.0, -15.0])

# --- 默认障碍物 (用于 GUI 演示模式) ---
DEFAULT_OBSTACLE = {'x': 350, 'y': 70, 'r': 30}

# --- 规划器核心参数 ---
STEP_SIZE = 10.0
MAX_ITERS = 2000