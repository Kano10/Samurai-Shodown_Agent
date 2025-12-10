import time
import numpy as np
from ss_env_Bot import SamuraiShodownEnv

# Botones que quieres permitir
BOTONES_PERMITIDOS = ['B','A','UP','DOWN','LEFT','RIGHT','C','Y','X','Z']

def main():
    env = SamuraiShodownEnv()
    obs, info = env.reset()

    botones_reales = env.env.buttons
    mapeo = {b:i for i,b in enumerate(botones_reales)}

    for b in BOTONES_PERMITIDOS:
        if b not in mapeo:
            raise ValueError(f"Botón '{b}' no existe. Usa uno de: {botones_reales}")

    while True:
        # Crear acción vacía
        action = np.zeros(len(botones_reales), dtype=int)

        # Elegir un botón permitido
        boton = np.random.choice(BOTONES_PERMITIDOS)
        idx = mapeo[boton]
        action[idx] = 1

        # Step correcto
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Retro renderiza solo
        time.sleep(0.02)

        if done:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
