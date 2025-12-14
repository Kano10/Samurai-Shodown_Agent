import numpy as np
from ss_env_Bot import SamuraiShodownEnv

# -------------------------------
# EDITA AQUÍ tus botones permitidos
# -------------------------------
BOTONES_PERMITIDOS = ['A', 'B', 'C']
# -------------------------------

# Diccionario esperado
BOTONES_VALIDOS = {
    'A': 0,
    'B': 1,
    'C': 2,
    'BLOCK': 3,
    'LEFT': 4,
    'RIGHT': 5,
    'UP': 6,
    'DOWN': 7,
    'SPECIAL': 8
}

def check_botones():
    print("=============== CHEQUEO DE BOTONES ===============")

    print("\n1) Tipo de BOTONES_PERMITIDOS:")
    print("   ->", type(BOTONES_PERMITIDOS))
    
    print("\n2) Tipos individuales:")
    for b in BOTONES_PERMITIDOS:
        print(f"   '{b}' -> {type(b)}")

    print("\n3) ¿Son válidos?")
    for b in BOTONES_PERMITIDOS:
        if b in BOTONES_VALIDOS:
            print(f"   OK: '{b}' está en BOTONES_VALIDOS (índice {BOTONES_VALIDOS[b]})")
        else:
            print(f"   ERROR: '{b}' NO existe en los botones válidos.")

    print("\n4) Revisando acción generada:")
    env = SamuraiShodownEnv()
    tam = env.action_space.shape[0]
    print("   El entorno espera un vector de tamaño:", tam)

    # Crear acción de prueba
    accion = np.zeros(tam, dtype=int)

    try:
        for b in BOTONES_PERMITIDOS:
            idx = BOTONES_VALIDOS[b]
            accion[idx] = 1

        print("   Acción generada correctamente:", accion)

    except Exception as e:
        print("   ERROR generando acción:")
        print("   ->", e)

    print("\n=============== CHEQUEO COMPLETO ===============")

if __name__ == "__main__":
    check_botones()
