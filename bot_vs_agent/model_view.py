import sys
sys.path.append('./RL')
from stable_baselines3 import PPO
from ss_env_Bot import SamuraiShodownEnv
import time

# === Parámetros ===
model_path = "./RL/ppo_SS_model1" 
max_steps = 10_000                 # Num de pasos a ver

# === Carga el entorno y el modelo ===
env = SamuraiShodownEnv()
model = PPO.load(model_path)
obs, _ = env.reset()
print("Modelo Cargado")
reca = 0

# === Loop de visualización ===
for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    #print("Inicio de step")
    obs, reward, done, _, info = env.step(action)
    reca += reward
    print(reca)
    env.render()
    #print("Step hecho")

    # Si termina el episodio, reseteamos
    if done:
        reca = 0
        obs, _ = env.reset()

env.close()
