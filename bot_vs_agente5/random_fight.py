import numpy as np
from ss_env_Bot import SamuraiShodownEnv
import time

def combate_random(max_steps=1000):
    env = SamuraiShodownEnv(n=4)  # frame skip = 4
    obs, info = env.reset()

    print("=== COMBATE RANDOM INICIADO ===")

    for step in range(max_steps):
        # Acción aleatoria
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        # Leer vida desde info
        player_hp = info.get("health", "N/A")
        enemy_hp = info.get("enemy_health", "N/A")

        print(
            f"Paso {step:04d} | "
            f"HP Jugador: {player_hp:>3} | "
            f"HP Enemigo: {enemy_hp:>3} | "
            f"Reward: {reward:+.3f}"
        )

        # Pequeña pausa para que sea visible
        time.sleep(0.03)

        if terminated or truncated:
            print("=== COMBATE TERMINADO ===")
            break

    env.close()


if __name__ == "__main__":
    combate_random()
