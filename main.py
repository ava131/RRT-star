import time
import pandas as pd
import numpy as np

from config import *
from environment import RobotArmEnv
from planner import RRTPlanner
from gui import RobotArmGUI


def run_randomized_benchmark(env, trials=50):
    print(f"\n🚀 开始 Benchmark 测试 (随机障碍物场景，共 {trials} 次实验)...")

    # 1. 构建同一套“题库” (预生成 50 个随机障碍物)
    random_obstacles = []
    np.random.seed(42)  # 固定题库种子，保证你每次跑脚本题库都一样
    for _ in range(trials):
        rand_x = np.random.randint(150, 550)
        rand_y = np.random.randint(50, 300)
        rand_r = np.random.randint(20, 50)
        random_obstacles.append({'x': rand_x, 'y': rand_y, 'r': rand_r})

    results = []

    for algo in ['RRT', 'RRT*']:
        use_star = (algo == 'RRT*')
        print(f"🔄 正在运行 {algo} 挑战 {trials} 个随机场景...")

        success_count, total_time, total_cost, total_nodes = 0, 0.0, 0.0, 0

        for i in range(trials):
            # 将环境切换为题库中的第 i 个随机障碍物
            env.update_obstacle(random_obstacles[i])

            # 【核心控制变量】固定该次实验的算法采样种子
            np.random.seed(i)

            planner = RRTPlanner(env, START_CFG, GOAL_CFG, step_size=STEP_SIZE, use_rrt_star=use_star)

            start_time = time.time()
            success = planner.plan(MAX_ITERS)
            end_time = time.time()

            if success:
                success_count += 1
                total_time += (end_time - start_time)
                total_nodes += len(planner.node_list)
                path_cost = sum(
                    [np.linalg.norm(planner.path[j] - planner.path[j - 1]) for j in range(1, len(planner.path))])
                total_cost += path_cost

        # 统计平均值
        avg_time = (total_time / success_count) * 1000 if success_count > 0 else 0
        avg_cost = total_cost / success_count if success_count > 0 else 0
        avg_nodes = total_nodes / success_count if success_count > 0 else 0

        results.append({
            'Algorithm': algo,
            'Success Rate (%)': f"{(success_count / trials) * 100:.1f}",
            'Avg Time (ms)': round(avg_time, 2),
            'Avg Path Cost': round(avg_cost, 2),
            'Avg Nodes': int(avg_nodes)
        })

    print("\n" + "=" * 60)
    print("📊 Benchmark 实验结果汇总 (基于随机障碍物场景)")
    print("=" * 60)
    print(pd.DataFrame(results).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    # 初始化全局环境 (使用 config 中配置的默认障碍物)
    env = RobotArmEnv(LINK_LENGTHS, CENTER, DEFAULT_OBSTACLE)

    # ==========================================
    # 请通过注释/取消注释下方代码，在两个模式间切换
    # ==========================================

    # 【模式A】交互式演示 (用于课堂 Presentation)
    # app = RobotArmGUI(env)
    # app.run()

    # 【模式B】Benchmark 数据统计 (用于撰写 Report)
    run_randomized_benchmark(env, trials=50)