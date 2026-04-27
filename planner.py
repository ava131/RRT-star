import numpy as np


class Node:
    def __init__(self, angles):
        self.angles = np.array(angles)
        self.parent = None
        self.cost = 0.0


def interpolate_path(path, steps=10):
    """路径平滑插值函数"""
    if not path: return []
    smooth = []
    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, steps, endpoint=False):
            smooth.append(path[i] + (path[i + 1] - path[i]) * t)
    smooth.append(path[-1])
    return smooth


class RRTPlanner:
    def __init__(self, env, start, goal, step_size=10.0, use_rrt_star=True):
        self.env = env
        self.start = Node(start)
        self.goal = Node(goal)
        self.step_size = step_size
        self.use_rrt_star = use_rrt_star
        self.node_list = [self.start]
        self.path = []
        self.done = False

    def plan(self, max_iters):
        """供无头模式（Benchmark）直接获取结果的阻塞调用"""
        for _ in range(max_iters):
            if self.step():
                return True
        return False

    def step(self):
        """单步执行，供 GUI 渲染动画使用"""
        if self.done: return True

        # 采样 (20% 目标偏置)
        rnd = self.goal.angles if np.random.rand() < 0.2 else np.random.uniform(-180, 180, 3)
        nearest_node = min(self.node_list, key=lambda n: np.linalg.norm(n.angles - rnd))

        diff = rnd - nearest_node.angles
        dist = np.linalg.norm(diff)
        new_angles = nearest_node.angles + (diff / dist) * min(dist, self.step_size)

        if self.env.check_collision(new_angles): return False

        new_node = Node(new_angles)
        new_node.cost = nearest_node.cost + np.linalg.norm(new_angles - nearest_node.angles)
        new_node.parent = nearest_node

        if self.use_rrt_star:
            self._rewire(new_node)
        else:
            self.node_list.append(new_node)

        # 检查是否触达终点
        if np.linalg.norm(new_node.angles - self.goal.angles) <= self.step_size:
            final_node = Node(self.goal.angles)
            final_node.parent = new_node
            self.path = self._get_path(final_node)
            self.done = True
            return True

        return False

    def _rewire(self, new_node):
        """RRT* 的核心优化与重连逻辑"""
        search_radius = 30.0
        near_inds = [i for i, n in enumerate(self.node_list)
                     if np.linalg.norm(n.angles - new_node.angles) < search_radius]

        # 1. 选最优父节点 (Choose Parent)
        for i in near_inds:
            near_node = self.node_list[i]
            d = np.linalg.norm(near_node.angles - new_node.angles)
            if near_node.cost + d < new_node.cost and not self._check_collision_line(near_node.angles, new_node.angles):
                new_node.parent = near_node
                new_node.cost = near_node.cost + d

        self.node_list.append(new_node)

        # 2. 挖墙脚重连 (Rewire)
        for i in near_inds:
            near_node = self.node_list[i]
            d = np.linalg.norm(new_node.angles - near_node.angles)
            if new_node.cost + d < near_node.cost and not self._check_collision_line(new_node.angles, near_node.angles):
                near_node.parent = new_node
                near_node.cost = new_node.cost + d

    def _check_collision_line(self, a1, a2):
        return self.env.check_collision((a1 + a2) / 2)

    def _get_path(self, node):
        path = []
        while node:
            path.append(node.angles)
            node = node.parent
        return path[::-1]