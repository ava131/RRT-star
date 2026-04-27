import numpy as np
import math

class RobotArmEnv:
    def __init__(self, link_lengths, center, obstacle):
        self.link_lengths = link_lengths
        self.center = center
        self.obstacle = obstacle

    def update_obstacle(self, new_obstacle):
        """用于在 Benchmark 中动态更新随机障碍物"""
        self.obstacle = new_obstacle

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