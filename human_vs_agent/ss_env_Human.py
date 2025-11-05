import gymnasium as gym
from gymnasium import spaces
import numpy as np
import retro
import cv2
import pygame

FRAME_SKIP = 4
BOTONES_USADOS = ['A', 'B', 'C', 'UP', 'DOWN', 'LEFT', 'RIGHT']

class SamuraiShodownEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super().__init__()

        # Cargamos el juego con dos jugadores
        self.game = retro.make(
            game="SamuraiShodown-Genesis",
            state="Level1.HaohmaruVsWanFu.2P",
            players=2,
            use_restricted_actions=retro.Actions.ALL
        )

        obs_shape = self.game.get_screen().shape
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = spaces.MultiBinary(len(BOTONES_USADOS))

        # Configuración Pygame
        pygame.init()
        pygame.display.set_caption("Samurai Shodown - Humano vs Agente")

        # Mapeo de teclas → botones del jugador humano (P2)
        self.key_action_map = {
            pygame.K_UP: "UP",
            pygame.K_DOWN: "DOWN",
            pygame.K_LEFT: "LEFT",
            pygame.K_RIGHT: "RIGHT",
            pygame.K_z: "A",
            pygame.K_x: "B",
            pygame.K_c: "C",
            #pygame.K_RETURN: "START"
        }

        self.current_frame = None

    def _get_human_action(self):
        """Lee las teclas presionadas por el humano (jugador 2)."""
        action = [0] * len(BOTONES_USADOS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        keys = pygame.key.get_pressed()
        for key, button in self.key_action_map.items():
            if keys[key]:
                idx = BOTONES_USADOS.index(button)
                action[idx] = 1
        return action

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.game.reset()
        self.current_frame = obs
        return obs

    def step(self, action_agent):
        """El agente controla P1 y tú controlas P2."""
        action_human = self._get_human_action()

        # Concatenamos acciones [P1 + P2] para el entorno Retro
        combined_action = np.concatenate([action_agent, action_human])

        total_reward = 0
        terminated = False
        truncated = False
        info = {}

        for _ in range(FRAME_SKIP):
            obs, reward, terminated, truncated, info = self.game.step(combined_action)
            total_reward += reward
            if terminated or truncated:
                break

        self.current_frame = obs
        return obs, total_reward, terminated, truncated, info

    def render(self):
        """Render fallback."""
        try:
            self.game.render()
        except Exception:
            if self.current_frame is not None:
                frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Samurai Shodown (Humano vs Agente)", frame)
                cv2.waitKey(1)

    def close(self):
        self.game.close()
        cv2.destroyAllWindows()
        pygame.quit()
