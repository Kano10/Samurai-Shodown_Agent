import gymnasium as gym
import numpy as np
import retro
import cv2
from gymnasium import spaces

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

class SamuraiShodownEnv(gym.Env):
    def __init__(self, resize_shape=(84, 84), n=4):
        super().__init__()

        # Escenario fijo
        self.estado_fijo = "Level1.HaohmaruVsWanFu"
        self.env = None
        self._crear_nuevo_env()
        
        # Inicialización del entorno y del procesamiento
        self.resize_shape = resize_shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.resize_shape[1], self.resize_shape[0]),
            dtype=np.uint8
        )
        self.action_space = self.env.action_space

        # Variables para la evaluación de recompensas
        self.last_step_action = None
        self.last_step_info = None
        self.no_atack_steps = 0
        self.prev_health = 120
        self.prev_enemy_hp = 120

        # Parámetros para frame skip
        self.n = n

        # Parámetros para estadísticas
        self.damage_to_player_steps = 0
        self.efective_attack_steps = 0
        self.efective_block_steps = 0
        self.total_steps = 0

        # Control para intro de batalla
        self.saltando_intro = False
        self.intro_steps_restantes = 175


    def _crear_nuevo_env(self):
        """Crea siempre el mismo escenario"""
        self.env = retro.make(
            game='SamuraiShodown-Genesis',
            state=self.estado_fijo,
            players=1,
            scenario='scenario',
            render_mode=True
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
        self.efective_block_steps = 0
        self.total_steps = 0

        return obs, info


    def step(self, action):
        terminated = False
        truncated = False
        reward = 0

        if self.saltando_intro:
            accion_neutral = [0] * self.action_space.shape[0]
            obs, _, terminated, truncated, info = self.env.step(accion_neutral)
            self.intro_steps_restantes -= 1

            if terminated or truncated or self.intro_steps_restantes <= 0:
                self.saltando_intro = False

            obs = self.preprocess(obs)
            return obs, 0.0, terminated, truncated, info

        for i in range(self.n):
            obs, _, terminated, truncated, info = self.env.step(action)
            obs = self.preprocess(obs)
            if terminated or truncated:
                break

        self.total_steps += 1

        curr_health = info.get('health', self.prev_health)
        curr_enemy_hp = info.get('enemy_health', self.prev_enemy_hp)
        damage_to_Enemy = self.prev_enemy_hp - curr_enemy_hp
        damage_to_Player = self.prev_health - curr_health

        attack_buttons = [0, 1, 8]
        block_buttons = [3]
        attack_pressed = [action[i] for i in attack_buttons]

        if damage_to_Enemy > 0:
            reward += 0.6 + (damage_to_Enemy / 120)
            self.efective_attack_steps += 1
        
        if curr_health == self.prev_health:
            if damage_to_Enemy == 0:
                self.no_atack_steps += 1
            else:
                self.no_atack_steps = 0

            if self.no_atack_steps >= 10:
                reward -= 0.06

        if damage_to_Player > 0:
            self.damage_to_player_steps += 1
            reward -= 0.2 + (damage_to_Player / 120)

            if any(action[i] for i in block_buttons) and sum(attack_pressed) == 0:
                reward += 0.05
                self.efective_block_steps += 1

        self.prev_health = curr_health
        self.prev_enemy_hp = curr_enemy_hp
        self.last_step_action = action
        self.last_step_info = info
        
        if terminated or truncated:
            if curr_health <= 0:
                reward -= 5
            if curr_enemy_hp <= 0:
                reward += 10

        info["efective_attack_steps"] = self.efective_attack_steps
        info["efective_block_steps"] = self.efective_block_steps
        info["total_steps"] = self.total_steps
        info["damage_to_player_steps"] = self.damage_to_player_steps

        done = terminated or truncated
        return obs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
