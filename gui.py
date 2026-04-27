import pygame
import math
from config import *
from planner import RRTPlanner, interpolate_path


def draw_star(surface, color, center, radius):
    """画五角星工具函数"""
    cx, cy = center
    points = []
    for i in range(5):
        angle_rad = math.radians(-90 + i * 72)
        points.append((cx + radius * math.cos(angle_rad), cy + radius * math.sin(angle_rad)))
        angle_rad = math.radians(-90 + i * 72 + 36)
        points.append((cx + radius * 0.4 * math.cos(angle_rad), cy + radius * 0.4 * math.sin(angle_rad)))
    pygame.draw.polygon(surface, color, points)


class RobotArmGUI:
    STATE_IDLE, STATE_PLANNING, STATE_READY, STATE_MOVING = 0, 1, 2, 3

    def __init__(self, env):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Interactive RRT* Robot Arm")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.btn_rect = pygame.Rect(20, 20, 140, 40)

        self.reset()

    def reset(self):
        self.state = self.STATE_IDLE
        self.planner = None
        self.smooth_path = []
        self.move_idx = 0

    def run(self):
        running = True
        while running:
            clicked = self._handle_events()
            if clicked is False:
                running = False
                continue

            self._update_logic(clicked)
            self._render()

            self.clock.tick(60)
        pygame.quit()

    def _handle_events(self):
        clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and self.btn_rect.collidepoint(event.pos):
                clicked = True
        return clicked

    def _update_logic(self, clicked):
        if self.state == self.STATE_IDLE and clicked:
            self.planner = RRTPlanner(self.env, START_CFG, GOAL_CFG, step_size=STEP_SIZE, use_rrt_star=True)
            self.state = self.STATE_PLANNING

        elif self.state == self.STATE_PLANNING:
            for _ in range(10):  # 加快渲染速度
                self.planner.step()
                if self.planner.done:
                    self.smooth_path = interpolate_path(self.planner.path, steps=10)
                    self.state = self.STATE_READY
                    break

        elif self.state == self.STATE_READY and clicked:
            self.move_idx = 0
            self.state = self.STATE_MOVING

        elif self.state == self.STATE_MOVING:
            if self.move_idx < len(self.smooth_path):
                self.move_idx += 1
            else:
                self.reset()  # 移动结束，复位

    def _render(self):
        self.screen.fill((255, 255, 255))
        self._draw_ui()
        self._draw_environment()
        if self.planner: self._draw_tree_and_path()
        self._draw_robot_arm()
        pygame.display.update()

    def _draw_ui(self):
        color = (100, 200, 100) if self.state in [self.STATE_IDLE, self.STATE_READY] else (200, 200, 200)
        pygame.draw.rect(self.screen, color, self.btn_rect, border_radius=5)
        texts = ["Start Plan", "Planning...", "Start Move", "Moving..."]
        text_surface = self.font.render(texts[self.state], True, (0, 0, 0))
        self.screen.blit(text_surface, (self.btn_rect.x + 20, self.btn_rect.y + 10))

    def _draw_environment(self):
        obs = self.env.obstacle
        pygame.draw.circle(self.screen, (128, 0, 128), (obs['x'], obs['y']), obs['r'])

        start_pts = self.env.forward_kinematics(START_CFG)
        for i in range(len(start_pts) - 1): pygame.draw.line(self.screen, (200, 200, 200), start_pts[i],
                                                             start_pts[i + 1], 2)

        goal_pts = self.env.forward_kinematics(GOAL_CFG)
        for i in range(len(goal_pts) - 1): pygame.draw.line(self.screen, (255, 215, 0), goal_pts[i], goal_pts[i + 1], 2)
        draw_star(self.screen, (255, 215, 0), goal_pts[-1], 15)

    def _draw_tree_and_path(self):
        if self.planner.node_list:
            for node in self.planner.node_list:
                if node.parent:
                    pygame.draw.line(self.screen, (220, 220, 220),
                                     self.env.forward_kinematics(node.parent.angles)[-1],
                                     self.env.forward_kinematics(node.angles)[-1], 1)

        if self.state in [self.STATE_READY, self.STATE_MOVING] and self.planner.path:
            pts = [self.env.forward_kinematics(ang)[-1] for ang in self.planner.path]
            if len(pts) > 1: pygame.draw.lines(self.screen, (0, 200, 0), False, pts, 4)

    def _draw_robot_arm(self):
        arm_angles = self.smooth_path[
            self.move_idx - 1] if self.state == self.STATE_MOVING and self.move_idx > 0 else START_CFG
        arm_pts = self.env.forward_kinematics(arm_angles)
        for i in range(len(arm_pts) - 1):
            pygame.draw.line(self.screen, (50, 50, 50), arm_pts[i], arm_pts[i + 1], 6)
            pygame.draw.circle(self.screen, (50, 50, 50), (int(arm_pts[i][0]), int(arm_pts[i][1])), 6)