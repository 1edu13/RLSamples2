import os
import sys
import importlib
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

# ==============================================================================
# AUDITORÍA FÍSICA: CRITERIOS DE JUICIO (LÍMITES SEVEROS APROBADOS)
# ==============================================================================
TEST_EPISODES = 10        # Diez vidas enteras para medir cuán perfectos son
EXITO_THRESHOLD = -200    # Un umbral extremadamente demandante para Pendulum
FRACASO_THRESHOLD = -400  # Debajo de esto, el agente simplemente no entiende la física

# ==============================================================================

def load_agent(agent_code, model_path, env):
    """ Función Creadora Inversa: Da vida al cerebro del archivo .pth en la clase adecuada """
    state_dim = np.prod(env.observation_space.shape)
    action_dim = 3 if agent_code == "DQN" else env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    if agent_code == "DQN":
        mod = importlib.import_module("11_pendulum_dqn")
        agent = mod.DQNAgent(env.observation_space.shape, action_dim)
        agent.main_net.load_state_dict(torch.load(model_path, map_location=agent.device, weights_only=True))
        return agent
        
    elif agent_code == "TD3":
        mod = importlib.import_module("12_pendulum_td3")
        agent = mod.TD3Agent(state_dim, action_dim, max_action)
        agent.actor.load_state_dict(torch.load(model_path, map_location=agent.device, weights_only=True))
        return agent
        
    elif agent_code == "SAC":
        mod = importlib.import_module("14_pendulum_sac")
        agent = mod.SACAgent(state_dim, action_dim, max_action)
        agent.actor.load_state_dict(torch.load(model_path, map_location=agent.device, weights_only=True))
        return agent
        
    elif agent_code == "PPO":
        mod = importlib.import_module("20_ppo_ultimate_convergence")
        agent = mod.PPOAgent(state_dim, action_dim)
        agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device, weights_only=True))
        return agent


def evaluate_agent(agent, agent_name, video_dir):
    print(f"\n[🎥] Entrando al Quirófano Visual. Grabando y Auditando a {agent_name}...")
    
    # Gym Wrapper: Instala una cámara imaginaria en el mundo físico y lo escupe a .mp4
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    # Grabamos TODOS LOS EPISODIOS forzosamente (RecordVideo por defecto a veces no graba continuos)
    env = RecordVideo(env, video_folder=video_dir, name_prefix=f"Videografia_{agent_name}", episode_trigger=lambda x: True)
    
    rewards = []
    
    for ep in range(TEST_EPISODES):
        state, _ = env.reset()
        state = state.flatten()
        ep_reward = 0
        done, truncated = False, False
        
        while not (done or truncated):
            if agent_name == "DQN":
                action = agent.select_action(state, evaluate=True)  # Quitamos Epsilon Randomness
                mapped_action = [[-2.0], [0.0], [2.0]][action]
                state, reward, done, truncated, _ = env.step(mapped_action)
                
            elif agent_name == "SAC":
                action = agent.select_action(state, evaluate=True)  # Solicitamos su Media Gausiana directa
                state, reward, done, truncated, _ = env.step(action)
                
            elif agent_name == "TD3":
                action = agent.select_action(state)                 # TD3 No usa Noise by default salvo si se lo metemos
                state, reward, done, truncated, _ = env.step(action)
                
            elif agent_name == "PPO":
                # Forzamos evaluación cruda tomando Tensor y Mean() puramente ignorando distribuciones
                state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(agent.device)
                mean_action = agent.policy.actor_mean(state_tensor)
                action = mean_action.cpu().data.numpy().flatten()
                action_clip = np.clip(action * 2.0, -2.0, 2.0)
                state, reward, done, truncated, _ = env.step(action_clip)
                
            state = state.flatten()
            ep_reward += reward
            
        rewards.append(ep_reward)
    env.close()
    
    # ================= ESTADÍSTICAS MATEMÁTICAS =================
    media = np.mean(rewards)
    desviacion = np.std(rewards) # Calcula cuán errático o ruidoso es al balancearse
    
    aciertos = sum(1 for r in rewards if r >= EXITO_THRESHOLD)
    mediocres = sum(1 for r in rewards if EXITO_THRESHOLD > r >= FRACASO_THRESHOLD)
    fracasos = sum(1 for r in rewards if r < FRACASO_THRESHOLD)
    
    tasa_acierto = (aciertos / TEST_EPISODES) * 100
    tasa_regular = (mediocres / TEST_EPISODES) * 100
    tasa_fracaso = (fracasos / TEST_EPISODES) * 100
    
    return {
        "media": media,
        "std": desviacion,
        "acierto": tasa_acierto,
        "regular": tasa_regular,
        "fracaso": tasa_fracaso,
        "rewards_raw": rewards
    }


def draw_plots(results, output_dir):
    agentes = list(results.keys())
    plt.style.use('dark_background') # Aesthetic hacker radar
    
    # -------- GRÁFICA 1: Puntuación Media y Lineas de Severidad --------
    medias = [results[a]["media"] for a in agentes]
    plt.figure(figsize=(10, 6))
    barras = plt.bar(agentes, medias, color=['purple', 'crimson', 'dodgerblue', 'limegreen'])
    plt.title('Puntuación Físíca Promedio (Evaluación Post-3000 Episodios)')
    
    plt.axhline(0, color='gold', linewidth=2, label="Centro de Gravedad Perfecto")
    plt.axhline(EXITO_THRESHOLD, color='mediumspringgreen', linestyle='--', linewidth=1.5, label=f'Raya de Éxito Estricto ({EXITO_THRESHOLD})')
    plt.axhline(FRACASO_THRESHOLD, color='red', linestyle='--', linewidth=1.5, label=f'Raya de Fracaso Absoluto ({FRACASO_THRESHOLD})')
    
    for b in barras:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h - 20 if h > -200 else h + 10, f'{h:.1f}', 
                 ha='center', va='bottom', color='white', fontweight='bold', 
                 bbox=dict(facecolor='black', alpha=0.5))
    plt.legend()
    plt.savefig(os.path.join(output_dir, "01_Rendimiento_Medio.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # -------- GRÁFICA 2: Stacked Bar Chart (Tasa de Éxito / Mediocridad / Fracaso) --------
    aciertos = np.array([results[a]["acierto"] for a in agentes])
    regulares = np.array([results[a]["regular"] for a in agentes])
    fracasos = np.array([results[a]["fracaso"] for a in agentes])
    
    plt.figure(figsize=(10, 6))
    plt.bar(agentes, aciertos, color='mediumspringgreen', edgecolor='white', label=f'Éxito Absoluto (> {EXITO_THRESHOLD})')
    plt.bar(agentes, regulares, bottom=aciertos, color='orange', edgecolor='white', label=f'Rendimiento Regular')
    plt.bar(agentes, fracasos, bottom=aciertos+regulares, color='red', edgecolor='white', label=f'Fracaso Matemático (< {FRACASO_THRESHOLD})')
    
    plt.title('Rigurosidad Probabilística: Tasa de Aciertos y Fiabilidad (%)')
    plt.ylabel('Porcentaje Total de Partidas Cumplido (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(output_dir, "02_Tasas_de_Clasificacion.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # -------- GRÁFICA 3: Varianza / Error Bars (Nivel de Temblores) --------
    stds = [results[a]["std"] for a in agentes]
    plt.figure(figsize=(10, 6))
    
    # Dibujamos las desviaciones estándar. Una linea larga = El robot fue incapaz de lograr constancia.
    plt.errorbar(agentes, medias, yerr=stds, fmt='D', color='white', ecolor='cyan', elinewidth=3, capsize=8, markersize=10, markerfacecolor='black')
    
    plt.title('Estabilidad Mecánica Fina: Constancia entre Partidas (Varianza)')
    plt.ylabel('Varianza Acumulada\n(Barras de error ultra-cortas representan fiabilidad blindada)')
    plt.axhline(0, color='gold', linewidth=1)
    
    for i, a in enumerate(agentes):
         plt.text(i + 0.1, medias[i], f'±{stds[i]:.1f}', color='cyan', fontsize=12, fontweight='bold')
         
    plt.savefig(os.path.join(output_dir, "03_Estabilidad_Mecanica_Varianza.png"), bbox_inches='tight', dpi=300)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("[X] Uso desde CMD: python 22_evaluate_all_methods.py [Nombre_Directorio_Maestro]")
        sys.exit(1)
        
    run_name = sys.argv[1]
    base_dir = os.path.join("Torneo_Maestro", run_name)
    models_dir = os.path.join(base_dir, "models")
    video_dir = os.path.join(base_dir, "videos_evaluacion")
    graficas_dir = os.path.join(base_dir, "graficas_finales")
    
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(graficas_dir, exist_ok=True)
    
    print("\n" + "#"*60)
    print(" INICIANDO TRIBUNAL DE EVALUACIÓN CLÍNICA (CON GRABACIÓN)")
    print("#"*60)
    
    # Diccionario de campeones que el script 21 debe haber dejado
    agentes_modelos = {
        "DQN": os.path.join(models_dir, "best_DQN.pth"),
        "TD3": os.path.join(models_dir, "best_TD3.pth"),
        "SAC": os.path.join(models_dir, "best_SAC.pth"),
        "PPO": os.path.join(models_dir, "best_PPO.pth")
    }
    
    resultados = {}
    
    for nombre, ruta in agentes_modelos.items():
        if os.path.exists(ruta):
            agente_vivo = load_agent(nombre, ruta, gym.make("Pendulum-v1"))
            stats = evaluate_agent(agente_vivo, nombre, video_dir)
            resultados[nombre] = stats
            
            print(f"  ⤷ ✅ Eval Completa. Éxito: {stats['acierto']}% | Fracaso: {stats['fracaso']}% | Estabilidad Varianza: ±{stats['std']:.1f}")
        else:
            print(f"  ⤷ ⚠️ Error Crítico: La mente del campeón {nombre} no fue exportada a {ruta}")
            
    # Compilación final
    if resultados:
        draw_plots(resultados, graficas_dir)
        print(f"\n🏆 ¡Auditoría concluida en su totalidad!")
        print(f"📷 Los archivos .mp4 crudos con los robots renderizados actuando están en: {video_dir}")
        print(f"📊 Los Dashboards Analíticos generados están en: {graficas_dir}")
    else:
        print("\n[!] No se pudieron fabricar las gráficas porque faltaron los cerebros en los directorios.")

if __name__ == "__main__":
    main()
