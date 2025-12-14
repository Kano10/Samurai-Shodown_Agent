# pip3 install opencv-python
# pip3 install stable-baselines3
# pip3 install tensorboard
# pip3 install pygame


# Escenarios disponibles
# Level1.WanfuVsHaohmaru
# Level1.Wanfu.bonusstage
# Level1.HaohmaruVsWanFu
# Level1.HaohmaruVsWanFu.2P
# Level1.HaohmaruVsHaohmaru

# Controles
# https://www.retrogames.cz/manualy/Genesis/Samurai_Shodown_-_Genesis_-_Manual.pdf
import gymnasium as gym
import numpy as np
import retro
import cv2
from gymnasium import spaces

class SamuraiShodownEnv(gym.Env):
    def __init__(self, resize_shape=(84, 84), n=4):
        super().__init__()

        self.posibles_estados = [
            "Level1.HaohmaruVsHaohmaru",
            "Level1.HaohmaruVsWanFu",
            "Level1.WanfuVsHaohmaru"
        ]

        self.env = None
        self._crear_nuevo_env()

        self.resize_shape = resize_shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.resize_shape[1], self.resize_shape[0]),
            dtype=np.uint8
        )
        self.action_space = self.env.action_space

        # Estado interno
        self.last_step_action = None
        self.last_step_info = None
        self.no_atack_steps = 0
        self.prev_health = 120
        self.prev_enemy_hp = 120

        self.n = n

        # Estadísticas
        self.damage_to_player_steps = 0
        self.efective_attack_steps = 0
        self.total_steps = 0

        # Intro
        self.saltando_intro = True
        self.intro_steps_restantes = 175

        # Botones
        self.botones = self.env.buttons
        # Mapeo ataques
        self.attack_buttons = [
            self.botones.index('B'),
            self.botones.index('A'),
            self.botones.index('C')
        ]

    def _crear_nuevo_env(self):
        estado_random = np.random.choice(self.posibles_estados)
        print(f"[SamuraiShodownEnv] Escenario seleccionado: {estado_random}")

        self.env = retro.make(
            game='SamuraiShodown-Genesis',
            state=estado_random,
            players=1,
            scenario='scenario',
            render_mode=False
        )

    def preprocess(self, obs):
        resized = cv2.resize(obs, self.resize_shape, interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.uint8), (2, 0, 1))

    def reset(self, seed=None, options=None):
        self.saltando_intro = True
        self.intro_steps_restantes = 175

        if self.env:
            self.env.close()
        self._crear_nuevo_env()

        obs, info = self.env.reset()
        obs = self.preprocess(obs)

        self.last_step_action = None
        self.last_step_info = info
        self.no_atack_steps = 0
        self.prev_health = 120
        self.prev_enemy_hp = 120

        self.damage_to_player_steps = 0
        self.efective_attack_steps = 0
        self.total_steps = 0

        return obs, info

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0

        # Saltar intro
        if self.saltando_intro:
            accion_neutral = [0] * self.action_space.shape[0]
            obs, _, terminated, truncated, info = self.env.step(accion_neutral)
            self.intro_steps_restantes -= 1

            if terminated or truncated or self.intro_steps_restantes <= 0:
                self.saltando_intro = False

            obs = self.preprocess(obs)
            return obs, 0.0, terminated, truncated, info

        # Repetir acción n frames
        for _ in range(self.n):
            obs, _, terminated, truncated, info = self.env.step(action)
            obs = self.preprocess(obs)
            if terminated or truncated:
                break

        self.total_steps += 1

        # Leer HP
        curr_health = info.get('health', self.prev_health)
        curr_enemy_hp = info.get('enemy_health', self.prev_enemy_hp)

        damage_to_Enemy = self.prev_enemy_hp - curr_enemy_hp
        damage_to_Player = self.prev_health - curr_health

        # Recompensas
        if damage_to_Enemy > 0:
            reward += 0.6 + (damage_to_Enemy / 120)
            self.efective_attack_steps += 1

        if curr_health == self.prev_health:
            if damage_to_Enemy == 0:
                self.no_atack_steps += 1
            else:
                self.no_atack_steps = 0
                # no hacer nada por 3 segundos
            if self.no_atack_steps >= 45:
                reward -= 0.06

        if damage_to_Player > 0:
            self.damage_to_player_steps += 1
            reward -= 0.2 + (damage_to_Player / 120)

        # Fin del episodio
        if terminated or truncated:
            if curr_health <= 0:
                reward -= 5
            if curr_enemy_hp <= 0:
                reward += 10

        info["efective_attack_steps"] = self.efective_attack_steps
        info["total_steps"] = self.total_steps
        info["damage_to_player_steps"] = self.damage_to_player_steps

        self.prev_health = curr_health
        self.prev_enemy_hp = curr_enemy_hp
        self.last_step_action = action
        self.last_step_info = info

        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
