import sys
sys.path.append('./RL')
from stable_baselines3 import PPO
from ss_env_Bot import SamuraiShodownEnv
import time

# === Parámetros ===
model_path = "./RL/ppo_SS_model"
max_steps = 10_000

# === Carga el entorno y el modelo ===
env = SamuraiShodownEnv()
model = PPO.load(model_path)
obs, _ = env.reset()
print("Modelo cargado")
reca = 0

# === Loop de visualización ===
for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)

    # Nuevo formato Gymnasium
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(reca)

    time.sleep(0.02)

    if done:
        reca = 0
        obs, _ = env.reset()

env.close()

