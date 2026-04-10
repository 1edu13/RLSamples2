import os
import glob
import importlib
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

# Importar las arquitecturas localmente
# Usamos importlib porque los archivos empiezan con números
dqn_module = importlib.import_module("05_highway_dqn")
td3_module = importlib.import_module("06_highway_td3")
ppo_module = importlib.import_module("07_highway_ppo")
sac_module = importlib.import_module("08_highway_sac")

def get_latest_model_path(base_dir, prefix, file_name):
    """Busca la carpeta más reciente de un algoritmo y devuelve la ruta del modelo."""
    if not os.path.exists(base_dir):
        return None
    dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    dirs.sort(reverse=True) # El más reciente primero gracias a YYYYMMDD-HHMMSS
    model_path = os.path.join(base_dir, dirs[0], file_name)
    if os.path.exists(model_path):
        return model_path
    return None

def evaluate_model(env_name, agent_type, episodes=10):
    print(f"\n[{agent_type}] Iniciando evaluación para {episodes} episodios...")
    
    # Configuración de Entorno similar a los entrenamientos
    env = gym.make(env_name, render_mode="rgb_array")
    
    is_discrete = (agent_type == "DQN")
    env.unwrapped.configure({
        "action": {"type": "DiscreteMetaAction" if is_discrete else "ContinuousAction"},
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True
        },
        "duration": 40
    })
    env.reset()
    
    state_shape = env.observation_space.shape
    try:
        action_dim = env.action_space.n if is_discrete else env.action_space.shape[0]
    except:
        action_dim = 2 # Backup por defecto si falla la lectura de ContinuousAction en evaluacion
        
    # Inicializar y Cargar pesos del Agente correspondiente
    agent_instance = None
    if agent_type == "DQN":
        agent_instance = dqn_module.DQNAgent(state_shape, action_dim)
        path = get_latest_model_path("models", "DQN_Highway", "best_model.pth")
        if path:
            agent_instance.main_net.load_state_dict(torch.load(path, map_location=agent_instance.device))
        else:
            print(f"⚠️  No se encontró un modelo entrenado para DQN.")
            return 0, 0
            
    elif agent_type == "TD3":
        max_action = float(env.action_space.high[0]) if not is_discrete else 1.0
        state_dim = np.prod(state_shape)
        agent_instance = td3_module.TD3Agent(state_dim, action_dim, max_action)
        path = get_latest_model_path("models", "TD3_Highway", "best_actor.pth")
        if path:
            agent_instance.actor.load_state_dict(torch.load(path, map_location=agent_instance.device))
        else:
            print(f"⚠️  No se encontró modelo para TD3.")
            return 0, 0
            
    elif agent_type == "PPO":
        state_dim = np.prod(state_shape)
        agent_instance = ppo_module.PPOAgent(state_dim, action_dim)
        path = get_latest_model_path("models", "PPO_Highway", "best_policy.pth")
        if path:
            agent_instance.policy.load_state_dict(torch.load(path, map_location=agent_instance.device))
        else:
            print(f"⚠️  No se encontró modelo para PPO.")
            return 0, 0
            
    elif agent_type == "SAC":
        max_action = float(env.action_space.high[0]) if not is_discrete else 1.0
        state_dim = np.prod(state_shape)
        agent_instance = sac_module.SACAgent(state_dim, action_dim, max_action)
        path = get_latest_model_path("models", "SAC_Highway", "best_actor.pth")
        if path:
            agent_instance.actor.load_state_dict(torch.load(path, map_location=agent_instance.device))
        else:
            print(f"⚠️  No se encontró modelo para SAC.")
            return 0, 0

    print(f"✔️ Pesos cargados: {path}")

    # Envolver para grabar videos SOLO de la evaluación
    video_dir = f"eval_videos/{agent_type}"
    # Grabamos el primer episodio y luego algunos saltados para no saturar memoria
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, 
        episode_trigger=lambda ep: ep < 3
    )
    
    rewards = []
    crashes = 0
    
    for ep in range(episodes):
        state, info = env.reset()
        if not is_discrete:
            state = state.flatten()
            
        ep_reward = 0
        done, truncated = False, False
        
        while not (done or truncated):
            # Seleccionar acción Greedy (Sin exploración)
            if agent_type == "DQN":
                action = agent_instance.select_action(state, evaluate=True)
            elif agent_type == "TD3":
                action = agent_instance.select_action(state)
            elif agent_type == "PPO":
                state_t = torch.FloatTensor(state.reshape(1, -1)).to(agent_instance.device)
                mean = agent_instance.policy.actor_mean(state_t)
                action = mean.cpu().data.numpy().flatten()
            elif agent_type == "SAC":
                action = agent_instance.select_action(state, evaluate=True)
                
            # Ejecutar acción
            next_state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            
            if not is_discrete:
                next_state = next_state.flatten()
            state = next_state
            
        rewards.append(ep_reward)
        
        # Analizar si terminó por choque
        # En highway-env "crashed" es la propiedad oficial de colisión
        if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle.crashed:
            crashes += 1
            
        print(f"  > Ep {ep+1}/{episodes} | Recompensa: {ep_reward:.2f} | Choque: {'Sí' if env.unwrapped.vehicle.crashed else 'No'}")
            
    env.close()
    
    avg_reward = np.mean(rewards)
    crash_rate = (crashes / episodes) * 100
    print(f"[{agent_type}] FIN - Media Rtdo: {avg_reward:.2f} | Tasa de Choques: {crash_rate:.1f}%")
    
    return avg_reward, crash_rate

if __name__ == "__main__":
    os.makedirs("eval_videos", exist_ok=True)
    algos = ["DQN", "TD3", "PPO", "SAC"]
    
    results = {}
    
    for algo in algos:
        avg_r, cr = evaluate_model("highway-v0", algo, episodes=10)
        results[algo] = {"reward": avg_r, "crash": cr}
        
    print("\n" + "="*50)
    print("🏆 RESULTADOS FINALES DE LA COMPARACIÓN 🏆")
    print("="*50)
    for algo in algos:
        r = results[algo]["reward"]
        c = results[algo]["crash"]
        print(f"{algo:>4} -> Recompensa Media: {r:>6.2f} | Choques: {c:>5.1f}%")
        
    # --- GRÁFICA COMPARATIVA ---
    try:
        labels = list(results.keys())
        rewards = [results[a]["reward"] for a in labels]
        crashes = [results[a]["crash"] for a in labels]
        
        x = np.arange(len(labels))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Barras de Recompensa
        color = 'tab:blue'
        ax1.set_xlabel('Algoritmos de Reinforcement Learning')
        ax1.set_ylabel('Recompensa Total Promedio', color=color)
        rects1 = ax1.bar(x - width/2, rewards, width, label='Recompensa Promedio', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Segundo Eje para Choques
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Tasa de Choques (%)', color=color)
        rects2 = ax2.bar(x + width/2, crashes, width, label='Choques (%)', color=color, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 100])

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        plt.title('Comparación de Modelos Entrenados en Highway-Env')
        
        # Leyendas
        lines, labels_1 = ax1.get_legend_handles_labels()
        lines2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels_1 + labels_2, loc='upper left')

        # Guardar gráfico
        plt.tight_layout()
        plt.savefig("comparacion_resultados.png")
        print("\n📊 Gráfico guardado exitosamente como 'comparacion_resultados.png'")
        
        # plt.show() # Descomentar si se quiere ver la gráfica en ventana interactiva
        
    except Exception as e:
        print(f"\n⚠️ No se pudo generar la gráfica matplotlib: {e}")
