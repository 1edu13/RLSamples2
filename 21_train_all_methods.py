import os
import sys
import importlib
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# ==============================================================================
# LAS 4 RECETAS DE ORO: HIPERPARÁMETROS MAESTROS (EDITABLES)
# ==============================================================================
# Estos son los parámetros que el evaluador inyectará matemáticamente a la RAM
# de tus agentes para que corran a su máximo rendimiento físico.

EPISODES = 3000

# 1. DQN (Conservador y Lento)
DQN_LR = 1e-3
DQN_BATCH = 128
DQN_EPS_DECAY = 0.999       # Decae el azar extremadamente lento para alcanzar los 3000 episodios
DQN_TARGET_UPDATE = 500

# 2. TD3 (Agresividad Determinista)
TD3_LR = 1e-3
TD3_TAU = 0.01
TD3_BATCH = 256
TD3_WARMUP = 2000

# 3. SAC (Suavidad Entrópica)
SAC_LR = 3e-4
SAC_TAU = 0.005
SAC_BATCH = 256
SAC_WARMUP = 2000

# 4. PPO (Convergencia Controlada Segura)
PPO_ACTOR_LR = 3e-4
PPO_CRITIC_LR = 1e-3
PPO_UPDATE_STEPS = 2000
PPO_K_EPOCHS = 80
# ==============================================================================


def train_dqn(env, log_dir, models_dir):
    print("\n" + "="*40 + "\n🔥 ENTRANDO A LA ARENA: DQN (Discreto)\n" + "="*40)
    mod = importlib.import_module("11_pendulum_dqn")
    agent = mod.DQNAgent(env.observation_space.shape, 3)
    memory = mod.ReplayBuffer(capacity=50000)
    
    # Inyección de parámetros
    agent.optimizer.param_groups[0]['lr'] = DQN_LR
    agent.batch_size = DQN_BATCH
    agent.epsilon_decay = DQN_EPS_DECAY
    agent.target_update_freq = DQN_TARGET_UPDATE
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "DQN"))
    best_reward = -float('inf')
    
    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            action = agent.select_action(state)
            # DQN cree que juega Atari. Nosotros mapeamos sus 3 botones a Torques Newtonianos físicos
            mapped_action = [[-2.0], [0.0], [2.0]][action]
            next_state, reward, done, truncated, _ = env.step(mapped_action)
            
            memory.push(state, action, reward, next_state, (done or truncated))
            state = next_state
            ep_reward += reward
            agent.update_epsilon()
            agent.train(memory)
            
        writer.add_scalar('Reward', ep_reward, ep)
        if ep_reward > best_reward and ep > 100:
            best_reward = ep_reward
            torch.save(agent.main_net.state_dict(), os.path.join(models_dir, "best_DQN.pth"))
        if ep % 250 == 0: 
            print(f"[{datetime.now().strftime('%H:%M:%S')}] DQN | Ep {ep:4d}/{EPISODES} | Reward Máxima Histórica: {best_reward:.1f}")
    writer.close()


def train_td3(env, log_dir, models_dir):
    print("\n" + "="*40 + "\n🔥 ENTRANDO A LA ARENA: TD3 (Off-Policy)\n" + "="*40)
    mod = importlib.import_module("12_pendulum_td3")
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    agent = mod.TD3Agent(state_dim, action_dim, float(env.action_space.high[0]))
    memory = mod.ReplayBuffer(state_dim, action_dim)
    
    agent.tau = TD3_TAU
    agent.actor_optimizer.param_groups[0]['lr'] = TD3_LR
    agent.critic_optimizer.param_groups[0]['lr'] = TD3_LR
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "TD3"))
    best_reward, total_steps = -float('inf'), 0
    max_action = float(env.action_space.high[0])
    
    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            total_steps += 1
            if total_steps < TD3_WARMUP:
                action = env.action_space.sample()
            else:
                action = (agent.select_action(state.flatten()) + np.random.normal(0, max_action * 0.1, size=action_dim)).clip(-max_action, max_action)
            
            next_state, reward, done, truncated, _ = env.step(action)
            memory.add(state.flatten(), action, reward, next_state.flatten(), (done or truncated))
            state = next_state
            ep_reward += reward
            if total_steps >= TD3_WARMUP: agent.train(memory, batch_size=TD3_BATCH)
            
        writer.add_scalar('Reward', ep_reward, ep)
        if ep_reward > best_reward and ep > 100:
            best_reward = ep_reward
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, "best_TD3.pth"))
        if ep % 250 == 0: 
            print(f"[{datetime.now().strftime('%H:%M:%S')}] TD3 | Ep {ep:4d}/{EPISODES} | Reward Máxima Histórica: {best_reward:.1f}")
    writer.close()


def train_sac(env, log_dir, models_dir):
    print("\n" + "="*40 + "\n🔥 ENTRANDO A LA ARENA: SAC (Off-Policy)\n" + "="*40)
    mod = importlib.import_module("14_pendulum_sac")
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    agent = mod.SACAgent(state_dim, action_dim, float(env.action_space.high[0]))
    memory = mod.ReplayBuffer(state_dim, action_dim)
    
    agent.tau = SAC_TAU
    agent.actor_optimizer.param_groups[0]['lr'] = SAC_LR
    agent.critic_optimizer.param_groups[0]['lr'] = SAC_LR
    agent.alpha_optim.param_groups[0]['lr'] = SAC_LR
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "SAC"))
    best_reward, total_steps = -float('inf'), 0
    
    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            total_steps += 1
            if total_steps < SAC_WARMUP:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state.flatten())
                
            next_state, reward, done, truncated, _ = env.step(action)
            memory.add(state.flatten(), action, reward, next_state.flatten(), (done or truncated))
            state = next_state
            ep_reward += reward
            if total_steps >= SAC_WARMUP: agent.train(memory, batch_size=SAC_BATCH)
            
        writer.add_scalar('Reward', ep_reward, ep)
        if ep_reward > best_reward and ep > 100:
            best_reward = ep_reward
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, "best_SAC.pth"))
        if ep % 250 == 0: 
            print(f"[{datetime.now().strftime('%H:%M:%S')}] SAC | Ep {ep:4d}/{EPISODES} | Reward Máxima Histórica: {best_reward:.1f}")
    writer.close()


def train_ppo(env, log_dir, models_dir):
    print("\n" + "="*40 + "\n🔥 ENTRANDO A LA ARENA: PPO (On-Policy)\n" + "="*40)
    mod = importlib.import_module("20_ppo_ultimate_convergence")
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    
    mod.K_EPOCHS = PPO_K_EPOCHS
    mod.LR_ACTOR = PPO_ACTOR_LR
    mod.LR_CRITIC = PPO_CRITIC_LR
    
    agent = mod.PPOAgent(state_dim, action_dim)
    buffer = mod.RolloutBuffer()
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "PPO"))
    best_reward, time_step = -float('inf'), 0
    
    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            time_step += 1
            action = agent.select_action(state.flatten(), buffer)
            action_clip = np.clip(action * 2.0, -2.0, 2.0)
            next_state, reward, done, truncated, _ = env.step(action_clip)
            buffer.rewards.append(reward)
            buffer.dones.append(done or truncated)
            state = next_state
            ep_reward += reward
            
            if time_step % PPO_UPDATE_STEPS == 0:
                agent.train(buffer)
                buffer.clear()
                
        writer.add_scalar('Reward', ep_reward, ep)
        if ep_reward > best_reward and ep > 100:
            best_reward = ep_reward
            torch.save(agent.policy.state_dict(), os.path.join(models_dir, "best_PPO.pth"))
        if ep % 250 == 0: 
            print(f"[{datetime.now().strftime('%H:%M:%S')}] PPO | Ep {ep:4d}/{EPISODES} | Reward Máxima Histórica: {best_reward:.1f}")
    writer.close()


def main():
    # Detectar nombre del torneo transmitido por el Pipeline Master
    run_name = f"Fecha_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
    
    log_dir = os.path.join("Torneo_Maestro", run_name, "logs")
    models_dir = os.path.join("Torneo_Maestro", run_name, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    env = gym.make("Pendulum-v1")
    
    print("\n" + "#"*60)
    print(" INICIANDO COMPILADOR MAESTRO DE REDES NEURONALES ")
    print(f" Almacén designado: {models_dir}")
    print("#"*60)
    
    train_dqn(env, log_dir, models_dir)
    train_td3(env, log_dir, models_dir)
    train_sac(env, log_dir, models_dir)
    train_ppo(env, log_dir, models_dir)
    
    env.close()
    print("\n✅ ¡Todos los gladiadores han sido entrenados y empaquetados exitosamente!")

if __name__ == "__main__":
    main()
