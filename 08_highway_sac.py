"""
Implementación desde cero de SAC (Soft Actor-Critic) 
para el entorno highway-env.

Algoritmo SAC:
Es actualmente el "Estado del Arte" (lo mejor que hay) para control continuo.
Híbrido matemático entre TD3 (Off-Policy con Replay Buffer) y PPO (Actor con distribución).

La Magia del "Soft" (Entropía):
Normalmente los RL solo quieren "Maximizar la recompensa".
SAC quiere "Maximizar Recompensa + Maximizar Entropía (Aleatoriedad)".
El coche se le premia por conducir bien PERO también se le premia por "hacer cosas al azar", 
mientras esas cosas no le hagan chocar. Esto hace que el agente no se estanque, sea súper 
robusto ante los cambios (ruido), y nunca deje de explorar sutilmente nuevas trazadas.
"""

import os
import math
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import highway_env

# Rango matemático seguro para la variabilidad (evitar nan/infinitos)
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# ============================================================
#  Bloque 1: Buffers
# ============================================================
class ReplayBuffer:
    """Buffer gigante para Off-Policy, similar a TD3 o DQN"""
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )

# ============================================================
#  Bloque 2: Redes SAC (Actor Estocástico + Crítico Gemelo)
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # En vez de sacar una acción limpia (TD3) SAC saca una Media y una "Log Varianza"
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        """TRUCO DE REPARAMETRIZACIÓN: Cómo hacer backpropagation por algo aleatorio."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Generar campana de gauss
        normal = Normal(mean, std)
        
        # Simular una extracción de acción (con ruido para explorar)
        # rsample() permite que las matemáticas fluyan hacia atrás (backprop)
        x_t = normal.rsample()
        
        # Aplicamos la Tanh matemática (suavizar la curva para no pasarse del volante +-1)
        # "Squash" matemático en SAC
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Ajuste de las matemáticas por deformar la distribución campana con Tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.max_action
        return action, log_prob, mean

class Critic(nn.Module):
    # Críticos son idénticos a TD3
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

# ============================================================
#  Bloque 3: Agente SAC con Entropía Automática (Alpha Tuning)
# ============================================================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Sistema] SAC usando: {self.device}")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005 # Actualización sueve de redes
        
        # Alpha representa la "Temperatura" o peso de la Aleatoriedad sobre la recompensa.
        # Aquí configuramos una pequeña sub-red (solo de 1 parámetro) para que el propio SAC
        # se auto-ajuste la locura que necesita en cada estadio de la partida.
        self.target_entropy = -action_dim # Heurística estándar para SAC contínuo
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if evaluate:
            # En evaluación pura no queremos locuras (entropía), usamos solo la Media
            _, _, action = self.actor.sample(state_t)
        else:
            # En entrenamiento usamos el muestreo "rígido" de campana Gaussiana completa
            action, _, _ = self.actor.sample(state_t)
            
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size, self.device)

        # ---------------------------- #
        #  1. ACTUALIZAR CRÍTICO (Q)   #
        # ---------------------------- #
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # Pesimista + Entropía
            target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha.exp() * next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- #
        #  2. ACTUALIZAR ACTOR (Pi)    #
        # ---------------------------- #
        # Importante: Las acciones "pi" VUELVEN a muestrearse aquí del estado actual
        # porque los pesos del actor pasaron por backprob en el loop anterior.
        pi_action, log_pi, _ = self.actor.sample(state)
        Q1_pi, Q2_pi = self.critic(state, pi_action)
        min_Q_pi = torch.min(Q1_pi, Q2_pi)
        
        # Recompensa + Entropía (Maximizamos ambas al mismo tiempo)
        actor_loss = ((self.log_alpha.exp() * log_pi) - min_Q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------- #
        #  3. ACTUALIZAR ALPHA         #
        # ---------------------------- #
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        alpha = self.log_alpha.exp().item()

        # ---------------------------- #
        #  4. POLÍTICATARGET TARDIA    #
        # ---------------------------- #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item(), alpha

# ============================================================
#  Main y Ejecución
# ============================================================
if __name__ == "__main__":
    run_name = f"SAC_Highway_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    video_dir = os.path.join("videos", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure({
        "action": {"type": "ContinuousAction"},
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
    #     env, video_folder=video_dir,
    #     episode_trigger=lambda ep: ep % 50 == 0
    # )

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action)
    memory = ReplayBuffer(state_dim, action_dim)

    EPISODES = 500
    WARMUP_STEPS = 5000

    total_steps = 0
    best_reward = -float('inf')

    for episode in range(1, EPISODES + 1):
        state, info = env.reset()
        state = state.flatten()
        
        ep_reward = 0
        done, truncated = False, False
        c_loss_tot, a_loss_tot, tr_steps = 0, 0, 0

        while not (done or truncated):
            total_steps += 1

            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state.flatten()
            
            is_terminal = done or truncated

            memory.add(state, action, reward, next_state, is_terminal)
            
            state = next_state
            ep_reward += reward

            if total_steps >= WARMUP_STEPS:
                # Entrenar la red a cada paso del simulador
                c_l, a_l, alpha = agent.train(memory, batch_size=256)
                c_loss_tot += c_l
                a_loss_tot += a_l
                tr_steps += 1
                
                # Logs exhaustivos en Tensorboard
                writer.add_scalar('Paso/Critic_Loss', c_l, total_steps)
                writer.add_scalar('Paso/Actor_Loss', a_l, total_steps)
                writer.add_scalar('Paso/Entropía_Alpha', alpha, total_steps)

        writer.add_scalar('Episodio/Recompensa_Total', ep_reward, episode)
        
        fase = "WARMUP" if total_steps < WARMUP_STEPS else "ENTRENANDO"
        
        if ep_reward > best_reward and total_steps >= WARMUP_STEPS:
            best_reward = ep_reward
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, "best_actor.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(models_dir, "best_critic.pth"))
            
        print(f"Episodio: {episode}/{EPISODES} ({fase}) | "
              f"Gas: {action[1]:.2f} Volante: {action[0]:.2f} | "
              f"Recompensa: {ep_reward:5.2f} | Pasos: {total_steps} | Mejor R: {best_reward:.2f}")

    torch.save(agent.actor.state_dict(), os.path.join(models_dir, "final_actor.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(models_dir, "final_critic.pth"))

    print("SAC completado y guardado.")
    writer.close()
    env.close()
