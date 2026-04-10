"""
Implementación desde cero de TD3 (Twin Delayed Deep Deterministic Policy Gradient) 
adaptado puramente a las físicas algorítmicas de Pendulum-v1.

================================================================================
OVERVIEW DEL MÉTODO (TD3) Y EL ENTORNO (PENDULUM)
================================================================================
TD3 es un algoritmo pilar para Espacios de Acciones Continuas, superando las 
limitaciones de DQN y DDPG.

ENTORNO Pendulum-v1 (Contexto Físico):
- ESTADO (s): Vector de tamaño 3 [cos(theta), sin(theta), theta_dot] describiendo 
  la posición espacial y vibración de un palo que cae bajo gravedad. 
- ACCIÓN (a): Vector único continuo continuo equivalente al empuje del motor acoplado  
  al eje rotatorio del palo. El rango admitido (Torque) varía orgánicamente entre [-2.0 y 2.0].

¿CÓMO SE PRODUCE EL APRENDIZAJE EN TD3?
1. Arquitectura Actor-Crítico Continua original: 
   - El Actor emite un número final (Ej: -1.74 de Torque). 
   - El Crítico lo toma y lo evalúa.
2. Críticos Gemelos (Twin): TD3 activa 2 críticos al mismo tiempo para evaluar la promesa del Actor, 
   y toma LA NOTA MÁS PESIMISTA siempre. El péndulo es castigador; si sobreestimas su ascenso terminarás estropeado cayendo al revés.
3. Actualizaciones Retrasadas (Delay): El Actor no se perfecciona de inmediato. Se permite que 
   los críticos vean más y más vueltas y se vuelvan estables en sus sentencias matemáticas 
   y sólo en pases alternos se enciende el autómata actorial.
4. Suavizado (Smoothing): Al calcular el futuro se le inyecta un mini ruido al giro proyectado 
   para garantizar generalización.
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

# ============================================================
#  Bloque 1: Memoria de Experiencia (Replay Buffer)
# ============================================================
class ReplayBuffer:
    """
    [CLASE: ReplayBuffer]
    Arrays numpy gigantes para absorber simulaciones mecánicas con acceso hiper rápido a la GPU.
    """
    def __init__(self, state_dim, action_dim, max_size=100000):
        """
        Entradas:
          - state_dim (int): 3 (El tamaño de la métrica física [cos, sin, vel]).
          - action_dim (int): 1 (Tamaño del empuje en el eje o Torque).
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """
        [MÉTODO: add - Inyección Cruda]
        Sobrescribe un slot temporal.
        Entradas: Vectores brutos sin procesador correspondientes a un Frame(t) estático.
        Salida: Internamente modifica el Array NumPy.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        """
        [MÉTODO: sample]
        Entradas:
          - batch_size (int): Cantidad general (usualmente 256 instantes rotos recogidos al azar).
        Salidas: 
          - 5 Tensores PyTorch mapeados nativamente a la VRAM.
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )

# ============================================================
#  Bloque 2: Redes Neuronales (Actor y Críticos)
# ============================================================

class Actor(nn.Module):
    """
    [CLASE: Actor]
    Red Neuronal en cargada de ser el controlador mecánico (Regulador PID)
    directo al motor del entorno.
    """
    def __init__(self, state_dim, action_dim, max_action):
        """
        Entradas: 
           state_dim = 3
           action_dim = 1
           max_action = 2.0 (Fuerza límite de la ley del torque)
        """
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim), # Al final arroja un sólo número abstracto
            nn.Tanh() # Transforma cualquera asimetría en un intervalo pulcro de [-1 a 1].
        )
        self.max_action = max_action

    def forward(self, state):
        """
        [MÉTODO: forward - Regulación de Fuerza]
        Toma el vector del péndulo, pasa las capas. 
        Salida: Multiplicamos el [-1 a 1] arrojado por el Max_Action físico (Torque de límite 2.0).
        """
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """
    [CLASE: Critic - El evaluador gemelo]
    Toma los estados corporales junto al Torque que aplicó el Actor y 
    emite 2 sentencias matemáticas (Q1 y Q2).
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Retorna predicción de recompensa a futuro.
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        """
        [MÉTODO: forward - Veredicto conjunto]
        Entradas:
          - state (tensor): Orientaciones geométricas
          - action (tensor): Torque real empleado
        Salidas: 2 Tensores Float independientes
        """
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

# ============================================================
#  Bloque 3: Agente TD3 Central
# ============================================================
class TD3Agent:
    def __init__(
        self, state_dim, action_dim, max_action,
        discount=0.99, tau=0.005, policy_noise=0.2, 
        noise_clip=0.5, policy_freq=2
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Sistema] TD3 usando: {self.device}")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount          
        self.tau = tau                    
        self.policy_noise = policy_noise  
        self.noise_clip = noise_clip      
        self.policy_freq = policy_freq    

        self.total_it = 0 

    def select_action(self, state):
        """
        [MÉTODO: select_action]
        Entradas: 
           - state (array): Los 3 números de sensores físicos 
        Salidas: 
           - array: Un NumPy array des-tensorizado con el valor del Torque a encender.
        """
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state_t).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        """
        [MÉTODO: train - Flujo del Aprendizaje]
        Entradas: 
          - replay_buffer: La base de datos cruda
        Salidas:
          - critic_loss.item() y actor_lossVal (Errores medidos para graficar).
          
        Lógica del Aprendizaje TD3:
        1. Recoge Batch aleatorios donde la vara caía/subía con torques específicos.
        2. Actualiza ambos Críticos: Compara su evaluación frente a la proyección de ganancia "pesimista" (T).
        3. Frecuencia Demorada: Cada 'N' iteraciones de críticos entrenados, SE ENCIENDE EL ACTOR.
        4. El Actor se reescala intentando ascender su Torque a una acción que el Crítico 1 catalogó como superior.
        5. Se aplican actualizaciones Soft-Weight (Tau) a las redes congeladas objetivo.
        """
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size, self.device)

        with torch.no_grad(): 
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            # EL NÚCLEO FILOSOFICO T: Ser un Pesimista Vitalicio - Elegir el MENOR 
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_lossVal = 0.0
        
        # El retraso del Actor (Delayed Update)
        if self.total_it % self.policy_freq == 0:
            
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            actor_lossVal = actor_loss.item()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return critic_loss.item(), actor_lossVal

if __name__ == "__main__":
    run_name = f"TD3_Pendulum_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"[Tensorboard] Guarda corriendo datos en la carpeta: {log_dir}")

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset()

    state_dim = np.prod(env.observation_space.shape) # Para Péndulo: 3 variables
    action_dim = env.action_space.shape[0]           # 1 Fuerza de Torque físico
    max_action = float(env.action_space.high[0])     # Potencia del motor límite = 2.0 

    agent = TD3Agent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    EPISODIOS = 500
    EXPLORATION_NOISE = 0.1 
    WARMUP_STEPS = 5000     

    total_steps = 0
    best_reward = -float('inf')

    for episode in range(1, EPISODIOS + 1):
        state, info = env.reset()
        state = state.flatten() 
        
        episode_reward = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        train_steps = 0
        
        done, truncated = False, False

        while not (done or truncated):
            total_steps += 1

            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample() 
            else:
                action = (
                    agent.select_action(state)
                    + np.random.normal(0, max_action * EXPLORATION_NOISE, size=action_dim)
                ).clip(-max_action, max_action)

            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state.flatten()
            
            is_terminal = done or truncated

            replay_buffer.add(state, action, reward, next_state, is_terminal)

            state = next_state
            episode_reward += reward

            if total_steps >= WARMUP_STEPS:
                c_loss, a_loss = agent.train(replay_buffer, batch_size=256)
                
                episode_critic_loss += c_loss
                if a_loss != 0.0:  
                    episode_actor_loss += a_loss
                train_steps += 1
                
                writer.add_scalar('Paso/Critic_Loss', c_loss, total_steps)

        writer.add_scalar('Episodio/Recompensa', episode_reward, episode)
        
        if train_steps > 0: 
            writer.add_scalar('Episodio/Media_Critic_Loss', episode_critic_loss / train_steps, episode)
            writer.add_scalar('Episodio/Media_Actor_Loss', episode_actor_loss / train_steps, episode)

        fase = "(Calentamiento)" if total_steps < WARMUP_STEPS else "(Entrenando)"
        
        if episode_reward > best_reward and total_steps >= WARMUP_STEPS:
            best_reward = episode_reward
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, "best_actor.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(models_dir, "best_critic.pth"))

        print(f"Episodio: {episode}/{EPISODIOS} {fase} | Recompensa: {episode_reward:5.2f} | Pasos: {total_steps}")

    torch.save(agent.actor.state_dict(), os.path.join(models_dir, "final_actor.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(models_dir, "final_critic.pth"))
    
    writer.close()
    env.close()
