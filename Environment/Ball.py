import pygame
import parameters
from CollusionStrategies import AbstractCollusionStrategy
import numpy as np
import random


class Ball:
    def __init__(self, collusion_strategy: AbstractCollusionStrategy, window):
        self.x = parameters.X_BALL
        self.y = parameters.Y_BALL
        self.r = parameters.R_BALL
        self.V = parameters.V_BALL
        self.alpha = parameters.ALPHA_BALL
        self.d_alpha = parameters.DELTA_ALPHA
        self.initial_v_x = parameters.V_X_BALL
        self.initial_v_y = parameters.V_Y_BALL
        self.collusion_strategy = collusion_strategy
        self.window = window
        self.collusion_strategy.set_ball(self)
        self.color = parameters.BALL_COLOR

    def draw(self):
        pygame.draw.circle(self.window, self.color, (int(np.round(self.x)), int(np.round(self.y))), self.r)

    def move(self):
        if self.alpha < 0:
            self.alpha += 2 * np.pi
        self.x += self.V * np.cos(self.alpha)
        self.y += self.V * np.sin(self.alpha)

    def reset(self):
        rand1 = random.random()
        self.alpha = parameters.ALPHA_BALL if rand1 < 0.5 else np.pi - parameters.ALPHA_BALL
        self.alpha = random.uniform(self.alpha - np.pi / 12, self.alpha + np.pi / 12)
        self.y = parameters.Y_BALL
        self.x = parameters.X_BALL
