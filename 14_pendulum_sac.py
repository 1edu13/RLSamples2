"""
Implementación desde cero de SAC (Soft Actor-Critic) 
adaptado orgánicamente al simulador mecánico Pendulum-v1.

================================================================================
OVERVIEW DEL MÉTODO (SAC) Y ENTORNO PENDULUM 
================================================================================
Soft Actor-Critic reescribe el concepto clásico buscando una base exploratoria robusta 
mediante la "Optimización de Entropía Máxima".

ENTORNO Pendulum-v1 (Contexto Físico):
- ESTADO (s): Geometría tensorial en tiempo real del péndulo. Un array crudo `[cos, sin, vel_angular]`.
- ACCIÓN (a): Impulso de fuerza circular o "Torque" en un rango asintótico exacto de `[-2.0, 2.0]`.

¿CÓMO SE PRODUCE EL APRENDIZAJE EN SAC?
1. Acción Distribucional Continua: Al enfrentarse a los 3 inputs geométricos del péndulo, 
   sacará la Medía que promete el Actor, junto con su duda de dispersión paramétrica (Std. Deviation).
2. Reparametrización Crítica (RSAMPLE): Como dependemos de una campana de Gauss, en SAC es primordial 
   poder derivar el gradiente desde valores aleatorios simulados a un nivel C++ por Pytorch hasta los tensores madres.
3. Premio de Entropía "Alfa": SAC se autoexige no "sobreaprender". El coeficiente Alfa sube o reduce de facto
   la recompensa logarítmica si el péndulo peca ser monótono en su balanceo vertical o se enclaustra.
"""

import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6 

# ============================================================
#  Bloque 1: Buffer Replay
# ============================================================
class ReplayBuffer:
    """
    [CLASE: ReplayBuffer] Archivo de transiciones persistentes. 
    Atesora los movimientos gravitacionales en arrays planos masivos.
    """
    def __init__(self, state_dim, action_dim, max_size=100000):
        """
        Entradas: 
           - state_dim: 3 (Parámetros del ángulo)
           - action_dim: 1 (Torque del eje)
        """
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        """[MÉTODO: add] Mapeo de vectores directos temporales de Pendulum-v1 en los slots Numpy"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        """[MÉTODO: sample] Empaquetado aleatorio en GPU. Salida: 5 Tensores Batch."""
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )

# ============================================================
#  Bloque 2: Redes Estocásticas SAC 
# ============================================================
class Actor(nn.Module):
    """
    [CLASE: Actor Blando Computacional]
    Responsable unario del control dinámico de giro de fuerza del péndulo.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(256, action_dim)     
        self.log_std_layer = nn.Linear(256, action_dim)  
        self.max_action = max_action

    def forward(self, state):
        """[MÉTODO: forward] De las métricas extrae la fuerza en Media y Varianza log."""
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        """
        [MÉTODO: sample con Squash de Acción]
        Entradas: 
           - state (tensor): Vectores de gravedad
        Salidas:
           - action (tensor): El Torque crudo extraído al azar de la distribución normal y forzado por tanh a su límite.
           - log_prob (tensor): Corrección estocástica necesaria para Alfa.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True) 
        
        mean = torch.tanh(mean) * self.max_action
        return action, log_prob, mean

class Critic(nn.Module):
    """[CLASE: Evaluador Doble] Jueces Q1 y Q2 contra el entorno."""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        """Concatenación y evaluación base del estado de gravedad y fuerza usada."""
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

# ============================================================
#  Bloque 3: Agente Core SAC
# ============================================================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005 
        
        self.target_entropy = -action_dim 
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state, evaluate=False):
        """
        [MÉTODO: select_action] Interfaz Física.
        Entrada: NumPy array espacial. Salida: Float escalar de Torque mecánico.
        """
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if evaluate:
            _, _, action = self.actor.sample(state_t)
        else:
            action, _, _ = self.actor.sample(state_t)
            
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        """
        [MÉTODO: train - Máquina SAC de Oscilación Entrópica]
        Entradas: Tensor database Replay buffer.
        Lógica:
        1. Target Q: Exige pronosticar pesimismo e INCLUYE Entropía (Restar la log_probablidad compensadora).
        2. Crítico: Descenso estándar.
        3. Policy Update: Elevar las métricas pero cobrando caro a favor del azar Alfa.
        """
        state, action, reward, next_state, done = replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha.exp() * next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi_action, log_pi, _ = self.actor.sample(state)
        Q1_pi, Q2_pi = self.critic(state, pi_action)
        min_Q_pi = torch.min(Q1_pi, Q2_pi) 
        
        actor_loss = ((self.log_alpha.exp() * log_pi) - min_Q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        alpha = self.log_alpha.exp().item()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item(), alpha

if __name__ == "__main__":
    run_name = f"SAC_Pendulum_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    state_dim = np.prod(env.observation_space.shape) # Geometría
    action_dim = env.action_space.shape[0]           
    max_action = float(env.action_space.high[0])     # Máx torque permitido mecánicamente (2.0)

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

        while not (done or truncated):
            total_steps += 1

            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample() 
            else:
                action = agent.select_action(state) 

            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state.flatten()
            
            memory.add(state, action, reward, next_state, (done or truncated))
            
            state = next_state
            ep_reward += reward

            if total_steps >= WARMUP_STEPS:
                c_l, a_l, alpha = agent.train(memory, batch_size=256)
                
        writer.add_scalar('Episodio/Total_Reward', ep_reward, episode)
        
        if ep_reward > best_reward and total_steps >= WARMUP_STEPS:
            best_reward = ep_reward
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, "best_actor.pth"))

        print(f"Episodio: {episode}/{EPISODES} | Torque Medio: {action[0]:.2f} | Rtdo: {ep_reward:5.2f}")

    torch.save(agent.actor.state_dict(), os.path.join(models_dir, "final_actor.pth"))
    writer.close()
    env.close()
