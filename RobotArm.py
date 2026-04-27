import numpy as np
import pygame
import math
import sys
import time
import pandas as pd


# ==========================================
# 1. 基础环境与几何工具
# ==========================================
class RobotArmEnv:
    def __init__(self):
        self.link_lengths = [100, 100, 80]
        self.window_size = 700
        self.center = (self.window_size // 2, self.window_size // 2)
        # 障碍物位置 (紫色圆)
        self.obstacle = {'x': 350, 'y': 70, 'r': 30}

    def forward_kinematics(self, angles):
        rads = np.radians(angles)
        x, y = self.center
        points = [(x, y)]
        sum_theta = 0
        for i, length in enumerate(self.link_lengths):
            sum_theta += rads[i]
            x = x + length * math.cos(sum_theta)
            y = y - length * math.sin(sum_theta)
            points.append((x, y))
        return points

    def check_collision(self, angles):
        points = self.forward_kinematics(angles)
        ox, oy, r = self.obstacle['x'], self.obstacle['y'], self.obstacle['r']
        for i in range(len(points) - 1):
            p1, p2 = np.array(points[i]), np.array(points[i + 1])
            if self._dist_segment_to_circle(p1, p2, np.array([ox, oy])) < r:
                return True
        return False

    def _dist_segment_to_circle(self, p1, p2, center):
        p1_p2 = p2 - p1
        if np.linalg.norm(p1_p2) == 0: return np.linalg.norm(center - p1)
        t = np.clip(np.dot(center - p1, p1_p2) / np.dot(p1_p2, p1_p2), 0, 1)
        return np.linalg.norm(center - (p1 + t * p1_p2))


# ==========================================
# 2. 规划器 (兼容 RRT 和 RRT*)
# ==========================================
class Node:
    def __init__(self, angles):
        self.angles = np.array(angles)
        self.parent = None
        self.cost = 0.0


class InteractivePlanner:
    def __init__(self, env, start, goal, step_size=10.0, use_rrt_star=True):
        self.env = env
        self.start = Node(start)
        self.goal = Node(goal)
        self.step_size = step_size
        self.use_rrt_star = use_rrt_star  # 核心开关：决定用哪个算法
        self.node_list = [self.start]
        self.path = []
        self.done = False

    def step(self):
        if self.done: return True

        # 1. 采样 (20% 目标偏置)
        if np.random.rand() < 0.2:
            rnd = self.goal.angles
        else:
            rnd = np.random.uniform(-180, 180, 3)

        # 2. 找最近
        dists = [np.linalg.norm(node.angles - rnd) for node in self.node_list]
        nearest_ind = np.argmin(dists)
        nearest_node = self.node_list[nearest_ind]

        # 3. 延伸
        diff = rnd - nearest_node.angles
        dist = np.linalg.norm(diff)
        new_angles = nearest_node.angles + (diff / dist) * min(dist, self.step_size)

        if self.env.check_collision(new_angles):
            return False

        new_node = Node(new_angles)
        new_node.cost = nearest_node.cost + np.linalg.norm(new_angles - nearest_node.angles)
        new_node.parent = nearest_node

        # ==================================
        # 4. RRT* 核心逻辑 (条件触发)
        # ==================================
        if self.use_rrt_star:
            search_radius = 30.0
            near_inds = [i for i, n in enumerate(self.node_list)
                         if np.linalg.norm(n.angles - new_node.angles) < search_radius]

            # A. 选最好的爸爸
            for i in near_inds:
                near_node = self.node_list[i]
                d = np.linalg.norm(near_node.angles - new_node.angles)
                if near_node.cost + d < new_node.cost:
                    if not self.check_collision_line(near_node.angles, new_node.angles):
                        new_node.parent = near_node
                        new_node.cost = near_node.cost + d

            self.node_list.append(new_node)

            # B. 挖墙脚 (Rewire)
            for i in near_inds:
                near_node = self.node_list[i]
                d = np.linalg.norm(new_node.angles - near_node.angles)
                if new_node.cost + d < near_node.cost:
                    if not self.check_collision_line(new_node.angles, near_node.angles):
                        near_node.parent = new_node
                        near_node.cost = new_node.cost + d
        else:
            # 标准 RRT：不优化，直接加入
            self.node_list.append(new_node)

        # 检查是否到达终点
        if np.linalg.norm(new_node.angles - self.goal.angles) <= self.step_size:
            final_node = Node(self.goal.angles)
            final_node.parent = new_node
            self.path = self._get_path(final_node)
            self.done = True
            return True

        return False

    def check_collision_line(self, a1, a2):
        mid = (a1 + a2) / 2
        return self.env.check_collision(mid)

    def _get_path(self, node):
        path = []
        while node:
            path.append(node.angles)
            node = node.parent
        return path[::-1]


# ==========================================
# 3. 绘图与工具函数
# ==========================================
def draw_star(surface, color, center, radius):
    cx, cy = center
    points = []
    for i in range(5):
        angle_deg = -90 + i * 72
        angle_rad = math.radians(angle_deg)
        points.append((cx + radius * math.cos(angle_rad), cy + radius * math.sin(angle_rad)))
        angle_deg += 36
        angle_rad = math.radians(angle_deg)
        points.append((cx + radius * 0.4 * math.cos(angle_rad), cy + radius * 0.4 * math.sin(angle_rad)))
    pygame.draw.polygon(surface, color, points)


def interpolate_path(path, steps=10):
    if not path: return []
    smooth = []
    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, steps, endpoint=False):
            smooth.append(path[i] + (path[i + 1] - path[i]) * t)
    smooth.append(path[-1])
    return smooth


# ==========================================
# 4. 模式A：交互式可视化演示 (GUI)
# ==========================================
def run_interactive_demo():
    pygame.init()
    env = RobotArmEnv()
    screen = pygame.display.set_mode((env.window_size, env.window_size))
    pygame.display.set_caption("Interactive RRT* Robot Arm")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    start_cfg = np.array([45.0, -30.0, -15.0])
    goal_cfg = np.array([135.0, -30.0, -15.0])

    STATE_IDLE, STATE_PLANNING, STATE_READY, STATE_MOVING = 0, 1, 2, 3
    current_state = STATE_IDLE
    planner = None
    smooth_path = []
    move_idx = 0
    btn_rect = pygame.Rect(20, 20, 140, 40)

    running = True
    while running:
        click = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN and btn_rect.collidepoint(event.pos): click = True

        if current_state == STATE_IDLE and click:
            # GUI 模式默认使用 RRT* (效果好看)
            planner = InteractivePlanner(env, start_cfg, goal_cfg, use_rrt_star=True)
            current_state = STATE_PLANNING

        elif current_state == STATE_PLANNING:
            for _ in range(10):  # 加快渲染速度
                planner.step()
                if planner.done:
                    smooth_path = interpolate_path(planner.path, steps=10)
                    current_state = STATE_READY
                    break

        elif current_state == STATE_READY and click:
            move_idx = 0
            current_state = STATE_MOVING

        elif current_state == STATE_MOVING:
            if move_idx < len(smooth_path):
                move_idx += 1
            else:
                current_state, planner = STATE_IDLE, None

        # --- 渲染 ---
        screen.fill((255, 255, 255))

        # 按钮
        color = (100, 200, 100) if current_state in [STATE_IDLE, STATE_READY] else (200, 200, 200)
        pygame.draw.rect(screen, color, btn_rect, border_radius=5)
        text = ["Start Plan", "Planning...", "Start Move", "Moving..."][current_state]
        screen.blit(font.render(text, True, (0, 0, 0)), (btn_rect.x + 20, btn_rect.y + 10))

        # 树、终点、障碍物、起点
        if planner and planner.node_list:
            for node in planner.node_list:
                if node.parent:
                    pygame.draw.line(screen, (220, 220, 220), env.forward_kinematics(node.parent.angles)[-1],
                                     env.forward_kinematics(node.angles)[-1], 1)

        if current_state in [STATE_READY, STATE_MOVING] and planner.path:
            pts = [env.forward_kinematics(ang)[-1] for ang in planner.path]
            if len(pts) > 1: pygame.draw.lines(screen, (0, 200, 0), False, pts, 4)

        pygame.draw.circle(screen, (128, 0, 128), (env.obstacle['x'], env.obstacle['y']), env.obstacle['r'])

        start_pts = env.forward_kinematics(start_cfg)
        for i in range(len(start_pts) - 1): pygame.draw.line(screen, (200, 200, 200), start_pts[i], start_pts[i + 1], 2)

        goal_pts = env.forward_kinematics(goal_cfg)
        for i in range(len(goal_pts) - 1): pygame.draw.line(screen, (255, 215, 0), goal_pts[i], goal_pts[i + 1], 2)
        draw_star(screen, (255, 215, 0), goal_pts[-1], 15)

        # 机械臂实体
        arm_angles = smooth_path[move_idx - 1] if current_state == STATE_MOVING else start_cfg
        arm_pts = env.forward_kinematics(arm_angles)
        for i in range(len(arm_pts) - 1):
            pygame.draw.line(screen, (50, 50, 50), arm_pts[i], arm_pts[i + 1], 6)
            pygame.draw.circle(screen, (50, 50, 50), (int(arm_pts[i][0]), int(arm_pts[i][1])), 6)

        pygame.display.update()
        clock.tick(60)
    pygame.quit()


# ==========================================
# 5. 模式B：无头 Benchmark 测试 (用于写报告)
# ==========================================
def run_benchmark(trials=50, max_iters=2000):
    print(f"\n🚀 开始 Benchmark 测试 (各算法运行 {trials} 次)...")
    env = RobotArmEnv()
    start_cfg = np.array([45.0, -30.0, -15.0])
    goal_cfg = np.array([135.0, -30.0, -15.0])

    results = []

    for algo in ['RRT', 'RRT*']:
        use_star = (algo == 'RRT*')
        print(f"🔄 正在运行 {algo}...")

        success_count, total_time, total_cost, total_nodes = 0, 0.0, 0.0, 0

        for i in range(trials):
            np.random.seed(i)
            planner = InteractivePlanner(env, start_cfg, goal_cfg, step_size=10.0, use_rrt_star=use_star)

            start_time = time.time()
            success = False
            for _ in range(max_iters):
                if planner.step():
                    success = True
                    break
            end_time = time.time()

            if success:
                success_count += 1
                total_time += (end_time - start_time)
                total_nodes += len(planner.node_list)
                # 计算路径总代价值 (C-Space欧氏距离总和)
                path_cost = sum(
                    [np.linalg.norm(planner.path[j] - planner.path[j - 1]) for j in range(1, len(planner.path))])
                total_cost += path_cost

        # 统计平均值
        avg_time = (total_time / success_count) * 1000 if success_count > 0 else 0
        avg_cost = total_cost / success_count if success_count > 0 else 0
        avg_nodes = total_nodes / success_count if success_count > 0 else 0
        success_rate = (success_count / trials) * 100

        results.append({
            'Algorithm': algo,
            'Success Rate (%)': f"{success_rate:.1f}",
            'Avg Time (ms)': round(avg_time, 2),
            'Avg Path Cost': round(avg_cost, 2),
            'Avg Nodes': int(avg_nodes)
        })

    # 打印最终表格
    df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("📊 Benchmark 实验结果汇总 (可直接抄进 Report)")
    print("=" * 50)
    print(df.to_string(index=False))
    print("=" * 50)


# ==========================================
# ⚡ 程序入口 (在这里切换模式！)
# ==========================================
if __name__ == "__main__":
    # 【模式A】如果你想在课堂上演示动画，请取消注释下面这行：
    # run_interactive_demo()

    # 【模式B】如果你想跑数据写 Report，请取消注释下面这行：
    run_benchmark(trials=50)