from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from ss_env_Human import SamuraiShodownEnv
import os

class RenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        # training_env es un VecEnv; accedemos al primer env real
        try:
            real_env = self.training_env.envs[0]
            if hasattr(real_env, "render"):
                real_env.render()
        except Exception:
            pass
        return True

def main():
    # Creamos explicitamente DummyVecEnv (SB3 no lo volverá a envolver)
    env = DummyVecEnv([lambda: SamuraiShodownEnv()])

    model_path = "./RL/ppo_human_vs_model"
    if os.path.exists(f"{model_path}.zip"):
        model = PPO.load(model_path, env=env)
    else:
        model = PPO("CnnPolicy", env, learning_rate=2.5e-4, verbose=1)

    callbacks = CallbackList([RenderCallback()])
    print("Entrenamiento en vivo (humano vs agente) con render en cada paso...")
    model.learn(total_timesteps=2_000_000, callback=callbacks, reset_num_timesteps=False)
    model.save(model_path)
    env.close()

if __name__ == "__main__":
    main()
