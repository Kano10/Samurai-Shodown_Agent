from ss_env import SamuraiShodownEnv
from stable_baselines3 import PPO

env = SamuraiShodownEnv(human_vs_agent=True)
model = PPO.load("./RL/ppo_mk_model", env=env)

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render()
