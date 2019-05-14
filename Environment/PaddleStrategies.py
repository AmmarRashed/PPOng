from abc import ABC, abstractmethod
import pygame
from pygame.locals import *
import random
import parameters


class AbstractPaddleStrategy(ABC):
    def __init__(self):
        self.paddle = None
        self.ball = None
        self.env = None
        self.action_set = [-1, 0, 1]

    def set_ball(self, ball):
        self.ball = ball

    def set_paddle(self, paddle):
        self.paddle = paddle

    def set_env(self, env):
        self.env = env

    @abstractmethod
    def move(self, action=0):
        pass

class MousePaddleStrategy(AbstractPaddleStrategy):
    def __init__(self):
        AbstractPaddleStrategy.__init__(self)

    def move(self, action=0):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.paddle.update_position(mouse_y - int(self.paddle.h / 2))


class SimpleAIStrategy(AbstractPaddleStrategy):
    def move(self, action=0):
        if random.random() < 0.8:
            if self.ball.y > self.paddle.y + self.paddle.h / 2:
                self.paddle.update_with_velocity(1)
            elif self.ball.y < self.paddle.y + self.paddle.h / 2:
                self.paddle.update_with_velocity(-1)
            else:
                self.paddle.update_with_velocity(0)
        else:
            self.paddle.update_with_velocity(random.choice([-1, 0, 1]))

class ReinforcementLearningStrategy(AbstractPaddleStrategy):
    def move(self, action=0):
        action = action - 1
        self.paddle.update_with_velocity(action)
