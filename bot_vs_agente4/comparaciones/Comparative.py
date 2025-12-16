import numpy as np
import matplotlib.pyplot as plt
from ss_env_Bot import SamuraiShodownEnv
from stable_baselines3 import PPO

# ============================
# FRAME-SKIP IGUAL QUE TRAINING
# ============================
def step_with_frame_skip(env, action):
    obs_final = None
    total_reward = 0
    terminated = False
    truncated = False
    info_final = {}

    for _ in range(env.n):
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        obs_final = obs
        info_final = info
        if term or trunc:
            terminated = term
            truncated = trunc
            break

    return obs_final, total_reward, terminated, truncated, info_final


# ============================
# EVALUAR AGENTE
# ============================
def evaluar_agente(env, model, episodios=7, repeticiones=10):
    todas_recompensas = []
    todas_victorias = []

    for rep in range(repeticiones):
        recompensas_totales = []
        victorias = 0

        for ep in range(episodios):
            obs, _ = env.reset()
            done = False
            recompensa_ep = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = step_with_frame_skip(env, action)
                recompensa_ep += reward
                done = terminated or truncated

            recompensas_totales.append(recompensa_ep)

            # win/loss del episodio
            if info.get("enemy_health", 1) <= 0:
                victorias += 1

        todas_recompensas.append(recompensas_totales)
        todas_victorias.append(victorias)

    return todas_recompensas, todas_victorias


# ============================
# EVALUAR RANDOM
# ============================
def evaluar_random(env, episodios=7, repeticiones=10):
    todas_recompensas = []
    todas_victorias = []

    for rep in range(repeticiones):
        recompensas_totales = []
        victorias = 0

        for ep in range(episodios):
            obs, _ = env.reset()
            done = False
            recompensa_ep = 0

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = step_with_frame_skip(env, action)
                recompensa_ep += reward
                done = terminated or truncated

            recompensas_totales.append(recompensa_ep)

            if info.get("enemy_health", 1) <= 0:
                victorias += 1

        todas_recompensas.append(recompensas_totales)
        todas_victorias.append(victorias)

    return todas_recompensas, todas_victorias


# ============================
# GRÁFICAS
# ============================
def graficar_comparacion(todas_recompensas_model, todas_recompensas_random, victorias_model, victorias_random):
    import numpy as np
    import matplotlib.pyplot as plt

    data_model = np.array(todas_recompensas_model)
    data_random = np.array(todas_recompensas_random)

    plt.figure(figsize=(8,6))
    posiciones = [1, 1.3]

    box = plt.boxplot(
        [data_model.flatten(), data_random.flatten()],
        patch_artist=True,
        positions=posiciones,
        widths=0.2,
        labels=["Modelo PPO", "Agente Random"]
    )

    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    promedio_model = np.mean(data_model)
    promedio_random = np.mean(data_random)

    plt.scatter([posiciones[0]], [promedio_model], color='blue')
    plt.scatter([posiciones[1]], [promedio_random], color='green')

    plt.title("Recompensas Totales por Episodio")
    plt.ylabel("Recompensa")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('comparacion_recompensas.png')
    plt.close()

    # Gráfico de victorias
    plt.figure(figsize=(6,4))
    labels = ["Modelo PPO", "Random"]
    valores = [np.mean(victorias_model), np.mean(victorias_random)]

    plt.bar(labels, valores)
    plt.title("Tasa de Victorias por 7 Episodios")
    plt.ylabel("Victorias promedio (0-7)")
    plt.grid(axis='y')
    plt.savefig("comparacion_victorias.png")
    plt.close()


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    env = SamuraiShodownEnv()
    model_path = "../RL/ppo_SS_model"
    model = PPO.load(model_path)

    episodios = 7
    repeticiones = 10

    recompensas_model, victorias_model = evaluar_agente(env, model, episodios, repeticiones)
    recompensas_random, victorias_random = evaluar_random(env, episodios, repeticiones)

    graficar_comparacion(recompensas_model, recompensas_random, victorias_model, victorias_random)

    env.close()

    print("\nResultados:")
    print(f"Victorias modelo: {victorias_model}")
    print(f"Victorias random: {victorias_random}")

