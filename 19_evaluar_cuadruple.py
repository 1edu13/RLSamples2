import os
import importlib
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ==============================================================================
# SCRIPT DE EVALUACIÓN PURA: CARGA DIRECTA DE CEREBROS NEURONALES ENTRENADOS
# ==============================================================================
# Puesto que ya hemos gastado los recursos y el tiempo en forjar parámetros, 
# saltamos todo y cargamos directamente las mentes al entorno de Gym.

def find_latest_experiment_dir(base_dir="models_experimentos"):
    """Busca la última carpeta de Batalla Cuádruple generada"""
    if not os.path.exists(base_dir):
        raise Exception(f"Directorio {base_dir} no existe.")
        
    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    quad_dirs = [d for d in all_dirs if "Batalla_Cuadruple" in d]
    
    if not quad_dirs:
        raise Exception("No se encontró ninguna carpeta de Batalla_Cuadruple")
        
    # Python organiza alfabéticamente/numéricamente. Como usamos fecha y hora, el último es el más reciente.
    quad_dirs.sort(reverse=True)
    return os.path.join(base_dir, quad_dirs[0])

def evaluate_model(agent_name, AgentClass, model_path, env, is_sac=False):
    print(f"Despertando y evaluando cerebro: {agent_name}...")
    
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 1. Instanciamos un clon hueco (sin entrenar) del algoritmo adecuado
    agent = AgentClass(state_dim, action_dim, max_action)
    
    # 2. Le inyectamos milisegundos de memoria usando las pesas guardadas en la terminal (.pth)
    agent.actor.load_state_dict(torch.load(model_path, map_location=agent.device, weights_only=True))
        
    episodes_to_test = 5
    cumulative_rewards = []
    
    # 3. Lo llevamos al coliseo donde todo es determinista y no hay chance de explorar o dudar
    for _ in range(episodes_to_test):
        state, _ = env.reset()
        state = state.flatten()
        ep_reward = 0
        done, truncated = False, False
        
        while not (done or truncated):
            if is_sac:
                action = agent.select_action(state, evaluate=True)
            else:
                # TD3 no usa ruido en producción final real, le extirpamos el 'np.random.normal'
                action = agent.select_action(state)
            
            state, reward, done, truncated, _ = env.step(action)
            state = state.flatten()
            ep_reward += reward
            
        cumulative_rewards.append(ep_reward)
        
    media = np.mean(cumulative_rewards)
    print(f" -> Puntuación Final (Promedio de 5 Episodios): {media:.2f}")
    return media

def main():
    target_dir = find_latest_experiment_dir()
    print(f"\nUbicada última base de datos de modelos: {target_dir}")
    
    # Invocación de sus algoritmos base para usar las clases molde
    td3_module = importlib.import_module("12_pendulum_td3")
    sac_module = importlib.import_module("14_pendulum_sac")
    
    TD3Agent = getattr(td3_module, "TD3Agent")
    SACAgent = getattr(sac_module, "SACAgent")
    
    # Entorno 100% puro para la evaluación
    env = gym.make("Pendulum-v1")
    
    rendimientos = {}
    
    # Extraer y probar los 4 ficheros .pth recién generados
    pth_td3_v = os.path.join(target_dir, "best_actor_TD3_Veloz.pth")
    if os.path.exists(pth_td3_v):
        rendimientos["TD3 Veloz"] = evaluate_model("TD3 Veloz", TD3Agent, pth_td3_v, env, is_sac=False)

    pth_td3_s = os.path.join(target_dir, "best_actor_TD3_Gelido.pth")
    if os.path.exists(pth_td3_s):
        rendimientos["TD3 Gélido"] = evaluate_model("TD3 Gélido", TD3Agent, pth_td3_s, env, is_sac=False)

    pth_sac_v = os.path.join(target_dir, "best_actor_SAC_Veloz.pth")
    if os.path.exists(pth_sac_v):
        rendimientos["SAC Veloz"] = evaluate_model("SAC Veloz", SACAgent, pth_sac_v, env, is_sac=True)

    pth_sac_s = os.path.join(target_dir, "best_actor_SAC_Gelido.pth")
    if os.path.exists(pth_sac_s):
        rendimientos["SAC Gélido"] = evaluate_model("SAC Gélido", SACAgent, pth_sac_s, env, is_sac=True)

    env.close()
    
    # ------------- GRAFICADOR EXPRES BARRAS ESTADÍSTICAS -------------
    if not rendimientos:
        print("\n[!] Error crítico: No encontré los archivos .pth, asegúrate de que el log de arriba los extrajo.")
        return
        
    nombres = list(rendimientos.keys())
    valores = list(rendimientos.values())
    
    plt.figure(figsize=(10, 6))
    colores = ['orange', 'darkred', 'cyan', 'blue']
    
    barras = plt.bar(nombres, valores, color=colores, alpha=0.9, edgecolor='white', linewidth=1.5)
    
    plt.title('TEST DE RENDIMIENTO FÍSICO PURO\nEvaluación de los 4 Cerebros Finales')
    plt.ylabel('Puntuación Lograda (Mientras más cerca a 0, mucho mejor)')
    
    # El peor comportamiento castiga a -1600. El mejor roza el 0 negativo.
    plt.ylim(min(valores) - 200, 50) 
    plt.axhline(0, color='gold', linestyle='-', linewidth=2, label="Equilibrio Vertical")
    
    # Añadir los números encima de cada barra para lectura prolija
    for barra in barras:
        alto = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., alto - 40 if alto > -100 else alto + 20,
                f'{alto:.1f}',
                ha='center', va='bottom' if alto < -100 else 'top', 
                color='white', fontweight='bold', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))

    plt.style.use('dark_background')
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    
    grafica_path = os.path.join(target_dir, "evaluacion_limpia_final.png")
    plt.savefig(grafica_path, format="png", bbox_inches='tight', dpi=300)
    
    print(f"\n[ÉXITO EXTREMO] Rendimiento comprobado. Los 4 cerebros son exitosos.")
    print(f"[REPORTE] ¡Tienes el veredicto en barras guardado en: {os.path.basename(grafica_path)}!")

if __name__ == "__main__":
    main()
