import os
import importlib
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# ==============================================================================
# HIPERPARÁMETROS DEL EXPERIMENTO (LA BATALLA DE LAS 4 VARIANTES)
# ==============================================================================
EPISODES = 150  # Episodios evaluados por testeo

# --- 1. TD3 Veloz y Feroz (Agresividad en Aprendizaje) ---
TD3_FAST_LR = 1e-3
TD3_FAST_TAU = 0.01
TD3_FAST_BATCH_SIZE = 128
TD3_FAST_WARMUP = 2000

# --- 2. TD3 Gélido y Preciso (Estabilidad) ---
TD3_STABLE_LR = 1e-4
TD3_STABLE_TAU = 0.002
TD3_STABLE_BATCH_SIZE = 512
TD3_STABLE_WARMUP = 5000

# --- 3. SAC Veloz y Feroz (Agresividad Entrópica) ---
SAC_FAST_LR = 1e-3
SAC_FAST_TAU = 0.01
SAC_FAST_BATCH_SIZE = 128
SAC_FAST_WARMUP = 2000

# --- 4. SAC Gélido y Preciso (Estabilidad Matemática) ---
SAC_STABLE_LR = 1e-4
SAC_STABLE_TAU = 0.002
SAC_STABLE_BATCH_SIZE = 512
SAC_STABLE_WARMUP = 5000
# ==============================================================================

# Invocaciones Seguras a tus códigos maestros (sin tocarlos)
td3_module = importlib.import_module("12_pendulum_td3")
sac_module = importlib.import_module("14_pendulum_sac")

TD3Agent = getattr(td3_module, "TD3Agent")
TD3Replay = getattr(td3_module, "ReplayBuffer")

SACAgent = getattr(sac_module, "SACAgent")
SACReplay = getattr(sac_module, "ReplayBuffer")


def train_agent(agent_name, AgentClass, ReplayClass, env, hiperparams, writer, models_dir):
    print(f"\n{'='*60}\n INICIANDO ENTRENAMIENTO EXPERIMENTAL: {agent_name} \n{'='*60}")
    
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = AgentClass(state_dim, action_dim, max_action)
    replay_buffer = ReplayClass(state_dim, action_dim)
    
    # -------------------------------------------------------------
    # MUTACIÓN DE VARIABLES EN TIEMPO REAL SOBRE MEMORIA
    # -------------------------------------------------------------
    agent.tau = hiperparams['tau']
    
    # Engañar a los optimizadores internos de PyTorch
    if hasattr(agent, 'actor_optimizer'):
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = hiperparams['lr']
    if hasattr(agent, 'critic_optimizer'):
        for param_group in agent.critic_optimizer.param_groups:
            param_group['lr'] = hiperparams['lr']
            
    if hasattr(agent, 'alpha_optim'):
         for param_group in agent.alpha_optim.param_groups:
            param_group['lr'] = hiperparams['lr']
    # -------------------------------------------------------------

    total_steps = 0
    best_reward = -float('inf')
    rewards_history = []

    for episode in range(1, EPISODES + 1):
        state, info = env.reset()
        state = state.flatten()
        
        ep_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            total_steps += 1

            if total_steps < hiperparams['warmup']:
                action = env.action_space.sample() 
            else:
                if "TD3" in agent_name:
                    EXPLORATION_NOISE = 0.1
                    action = (
                        agent.select_action(state)
                        + np.random.normal(0, max_action * EXPLORATION_NOISE, size=action_dim)
                    ).clip(-max_action, max_action)
                else:
                    action = agent.select_action(state)

            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state.flatten()
            
            replay_buffer.add(state, action, reward, next_state, (done or truncated))
            state = next_state
            ep_reward += reward

            if total_steps >= hiperparams['warmup']:
                agent.train(replay_buffer, batch_size=hiperparams['batch_size'])

        writer.add_scalar(f'Batalla_Cuadruple/{agent_name}_Reward', ep_reward, episode)
        rewards_history.append(ep_reward)
        
        # Extractor de campeones parciales (Guarda el modelo)
        if ep_reward > best_reward and total_steps >= hiperparams['warmup']:
            best_reward = ep_reward
            if hasattr(agent, 'actor'):
                torch.save(agent.actor.state_dict(), os.path.join(models_dir, f"best_actor_{agent_name}.pth"))
            elif hasattr(agent, 'policy'):
                torch.save(agent.policy.state_dict(), os.path.join(models_dir, f"best_policy_{agent_name}.pth"))
                
        if episode % 10 == 0 or episode == 1:
            print(f"[{agent_name}] Episodio {episode:3d}/{EPISODES} | Puntos logrados: {ep_reward:7.2f} | Pasos motor: {total_steps}")
            
    return rewards_history


def main():
    # -------------------------------------------------------------
    # SISTEMA DE CARPETAS TOTALMENTE APARTADAS POR FECHA Y HORA
    # -------------------------------------------------------------
    run_name = f"Batalla_Cuadruple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Se genera una carpeta específica para la Batalla de este minuto.
    # Todas las configuraciones y gráficos se tirarán a una mega-carpeta aislada.
    log_dir = os.path.join("runs_experimentos", run_name)
    models_dir = os.path.join("models_experimentos", run_name)
    os.makedirs(models_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make("Pendulum-v1")
    
    # ========================== FASE 1 ========================== 
    td3_fast = {'lr': TD3_FAST_LR, 'tau': TD3_FAST_TAU, 'batch_size': TD3_FAST_BATCH_SIZE, 'warmup': TD3_FAST_WARMUP}
    td3_f_rewards = train_agent("TD3_Veloz", TD3Agent, TD3Replay, env, td3_fast, writer, models_dir)

    # ========================== FASE 2 ==========================
    td3_stable = {'lr': TD3_STABLE_LR, 'tau': TD3_STABLE_TAU, 'batch_size': TD3_STABLE_BATCH_SIZE, 'warmup': TD3_STABLE_WARMUP}
    td3_s_rewards = train_agent("TD3_Gelido", TD3Agent, TD3Replay, env, td3_stable, writer, models_dir)

    # ========================== FASE 3 ==========================
    sac_fast = {'lr': SAC_FAST_LR, 'tau': SAC_FAST_TAU, 'batch_size': SAC_FAST_BATCH_SIZE, 'warmup': SAC_FAST_WARMUP}
    sac_f_rewards = train_agent("SAC_Veloz", SACAgent, SACReplay, env, sac_fast, writer, models_dir)

    # ========================== FASE 4 ==========================
    sac_stable = {'lr': SAC_STABLE_LR, 'tau': SAC_STABLE_TAU, 'batch_size': SAC_STABLE_BATCH_SIZE, 'warmup': SAC_STABLE_WARMUP}
    sac_s_rewards = train_agent("SAC_Gelido", SACAgent, SACReplay, env, sac_stable, writer, models_dir)

    env.close()
    writer.close()
    
    # -------------------------------------------------------------
    # GRAFICACIÓN MATPLOTLIB: DIBUJO DE LAS 4 TENDENCIAS JUNTAS
    # -------------------------------------------------------------
    print("\nGenerando renderizado de la arena cuádruple...")
    plt.figure(figsize=(12, 7))
    
    # Sistema purificador de saltos (líneas de tendencia clara)
    def smooth(scalars, weight=0.85):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # 1. Trazado TD3 (Gama de Rojos-Naranjas)
    plt.plot(smooth(td3_f_rewards), label='TD3 Veloz (Feroz)', color='red', linewidth=3)
    plt.plot(smooth(td3_s_rewards), label='TD3 Gélido (Estable)', color='darkred', linestyle='--', linewidth=2)
    
    # 2. Trazado SAC (Gama de Azules)
    plt.plot(smooth(sac_f_rewards), label='SAC Veloz (Feroz)', color='cyan', linewidth=3)
    plt.plot(smooth(sac_s_rewards), label='SAC Gélido (Estable)', color='blue', linestyle='--', linewidth=2)

    plt.title('Batalla Final Péndulo: Velocidad vs Precisión (TD3 vs SAC)')
    plt.xlabel('Episodios Testeados')
    plt.ylabel('Recompensa (Puntuación Óptima = 0)')
    
    # Línea central dorada para marcar la meta final
    plt.axhline(0, color='gold', linestyle='-', alpha=0.9, label='Posición Perfecta Péndulo Vertical')
    
    plt.legend()
    plt.style.use('dark_background') # Esto dará un look brutal a la comparativa de 4 colores cyan/rojos
    plt.grid(True, alpha=0.2)
    
    graph_path = os.path.join(models_dir, "comparativa_cuadruple_HD.png")
    plt.savefig(graph_path, format="png", bbox_inches='tight', dpi=300)
    
    print(f"\n[ÉXITO EXTREMO] ¡La Cuádruple Batalla Ha Finalizado!")
    print(f"[CARPETAS] Todo empaquetado en tu propia carpeta única: {models_dir}")
    print(f"[GRÁFICA] La imagen con la superposición a color salvada como: {os.path.basename(graph_path)}")

if __name__ == "__main__":
    main()
