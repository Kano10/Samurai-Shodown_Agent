import retro
import numpy as np
import pygame
from pygame.locals import *

pygame.init()

# Crear entorno sin render de Pyglet
env = retro.make(game='SamuraiShodown-Genesis', render_mode='rgb_array')
obs, info = env.reset()

h, w, _ = obs.shape
screen = pygame.display.set_mode((w, h))
clock = pygame.time.Clock()

running = True
action = np.zeros(env.action_space.n, dtype=np.uint8)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Acción aleatoria
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Mostrar frame
    surf = pygame.surfarray.make_surface(np.flipud(np.rot90(obs)))
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    if done:
        obs, info = env.reset()

    clock.tick(60)

env.close()
pygame.quit()
