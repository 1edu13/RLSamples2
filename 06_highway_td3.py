"""
Implementación desde cero de TD3 (Twin Delayed Deep Deterministic Policy Gradient) 
para el entorno highway-env.

Algoritmo TD3:
A diferencia de DQN (que usa un panel de botones discretos), TD3 está hecho para
Controles Continuos (Acelerador de 0 a 100%, volante de -1 a 1).
Usa la Arquitectura Actor-Crítico:
- Actor: Mira el estado y decide físicamente cuánto pisar el acelerador y girar.
- Crítico (Gemelo): Mira lo que hizo el Actor y le da una "nota" (Q-Value). Usamos 
  dos críticos para evitar sobreestimar esa nota, cogiendo siempre la nota más pesimista.
"""

import math
import random
import os
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import highway_env

# ============================================================
#  Bloque 1: Memoria de Experiencia (Replay Buffer)
# ============================================================
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # En vez de usar un deque (lista), TD3 prefiere matrices Numpy pesadas preasignadas 
        # para que el muestreo en GPU sea extremandamente rápido.
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        # Escoger índices aleatorios
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )

# ============================================================
#  Bloque 2: Redes Neuronales (Actor y Críticos Gemelos)
# ============================================================

class Actor(nn.Module):
    """
    Toma las fotos de la carretera y escupe un vector (Volante, Acelerador)
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Tanh capa las salidas entre -1 y 1
        )
        self.max_action = max_action

    def forward(self, state):
        # Multiplicamos el [-1, 1] por el máximo real que permite el coche
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """
    Crítico Gemelo: Dos redes que evalúan independientemente la misma jugada.
    Toma (Estado + Acción de conducción) de entrada y emite una previsión de recompensa.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Red Crítica 1 (Q1)
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Red Crítica 2 (Q2)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # Concatena el estado del juego con la decisión del actor
        sa = torch.cat([state, action], 1)

        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

# ============================================================
#  Bloque 3: Agente TD3
# ============================================================
class TD3Agent:
    def __init__(
        self, state_dim, action_dim, max_action,
        discount=0.99, tau=0.005, policy_noise=0.2, 
        noise_clip=0.5, policy_freq=2
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"[Sistema] TD3 usando: {self.device}")

        # Actor Principal y Objetivo (Retrasado)
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        # Críticos Principales y Objetivos
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq # La D extra en TD3: Delayed policy updates

        self.total_it = 0

    def select_action(self, state):
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state_t).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Muestreo desde la memoria
        state, action, reward, next_state, done = replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            # Seleccionar la próxima acción usando la red objetivo del actor y 
            # añadimos un ligero "ruido" cortado (Target Policy Smoothing) para evitar que 
            # aprenda trampas específicas.
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Obtener Qs del crítico objetivo
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            # TRUCO MÁS IMPORTANTE DEL TD3: La T de "Twin". Cogemos el MÍNIMO (más pesimista)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Calificar acciones tomadas realmente (con el crítico principal)
        current_Q1, current_Q2 = self.critic(state, action)

        # El crítico quiere minimizar el error entre lo que predijo y el Target
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_lossVal = 0.0
        
        # TRUCO DELAY: El actor se actualiza con menos frecuencia (cada 2 pasos) 
        # para que el Crítico aprenda a juzgar bien antes de que el actor mueva nada.
        if self.total_it % self.policy_freq == 0:
            
            # El actor quiere maximizar el valor de Q1 para sus acciones
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            actor_lossVal = actor_loss.item()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # "Suavemente" actualizar las redes objetivo (Polyak averaging)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return critic_loss.item(), actor_lossVal

# ============================================================
#  Main y Ejecución
# ============================================================
if __name__ == "__main__":
    run_name = f"TD3_Highway_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    video_dir = os.path.join("videos", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"[Tensorboard] Guarda corriendo datos en la carpeta: {log_dir}")

    # Configuración entorno continuo!
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure({
        "action": {
            "type": "ContinuousAction", # Volante [-1, 1] y Gas [-1, 1]
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True
        },
        "duration": 40
    })
    env.reset()
    
    # env = gym.wrappers.RecordVideo(
    #     env, video_folder=video_dir, episode_trigger=lambda ep: ep % 50 == 0
    # )

    state_dim = np.prod(env.observation_space.shape) # Aplanado a vector
    action_dim = env.action_space.shape[0]           # Array de tamaño [Volante, Gas]
    max_action = float(env.action_space.high[0])     # Generalmente 1.0

    agent = TD3Agent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Hiperparámetros
    EPISODIOS = 500
    EXPLORATION_NOISE = 0.1
    WARMUP_STEPS = 5000 # Pasos dando volantazos aleatorios para reunir base de datos inicial

    total_steps = 0
    best_reward = -float('inf')

    for episode in range(1, EPISODIOS + 1):
        state, info = env.reset()
        state = state.flatten() # Aplanar matriz visual
        
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        train_steps = 0
        
        done, truncated = False, False

        while not (done or truncated):
            total_steps += 1

            # Acción
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample() # Aleatorio puro para calentar memoria
            else:
                action = (
                    agent.select_action(state)
                    + np.random.normal(0, max_action * EXPLORATION_NOISE, size=action_dim)
                ).clip(-max_action, max_action)

            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state.flatten()
            
            is_terminal = done or truncated

            # Guardar experiencia
            replay_buffer.add(state, action, reward, next_state, is_terminal)

            state = next_state
            episode_reward += reward

            # Entrenamiento
            if total_steps >= WARMUP_STEPS:
                c_loss, a_loss = agent.train(replay_buffer, batch_size=256)
                
                episode_critic_loss += c_loss
                if a_loss != 0.0:  # El actor no se entrena cada vez
                    episode_actor_loss += a_loss
                train_steps += 1
                
                # Logs a lo bestia en el tensorboard para depurar
                writer.add_scalar('Paso/Critic_Loss', c_loss, total_steps)

        # Logs por episodio completos
        writer.add_scalar('Episodio/Recompensa', episode_reward, episode)
        if train_steps > 0:
            writer.add_scalar('Episodio/Media_Critic_Loss', episode_critic_loss / train_steps, episode)
            writer.add_scalar('Episodio/Media_Actor_Loss', episode_actor_loss / train_steps, episode)

        fase = "(Calentamiento)" if total_steps < WARMUP_STEPS else "(Entrenando)"
        
        # Guardado de modelo óptimo
        if episode_reward > best_reward and total_steps >= WARMUP_STEPS:
            best_reward = episode_reward
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, "best_actor.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(models_dir, "best_critic.pth"))

        print(f"Episodio: {episode}/{EPISODIOS} {fase} | Recompensa: {episode_reward:5.2f} | Pasos Globales: {total_steps} | Mejor R: {best_reward:.2f}")

    # Guardado final
    torch.save(agent.actor.state_dict(), os.path.join(models_dir, "final_actor.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(models_dir, "final_critic.pth"))
    
    print("¡TD3 ha terminado! Videos y Modelos salvados.")
    writer.close()
    env.close()
