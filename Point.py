import numpy as np
import pygame

class Point:
    def __init__(self, x, y, color='black', cluster=None):
        self.x = x
        self.y = y
        self.color = color
        self.cluster = cluster

    @property
    def pos(self):
        return self.x, self.y

    def dist(self, point):
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

    def __str__(self, x, y):
        return f"x: {self.x}, y: {self.y}"

    def draw(self, screen):
        pygame.draw.circle(screen, self.cluster, (self.x, self.y), 10)