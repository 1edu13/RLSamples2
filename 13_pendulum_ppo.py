"""
Implementación de PPO (Proximal Policy Optimization)
reconfigurado nativamente para el entorno mecánico Pendulum-v1.

================================================================================
OVERVIEW DEL MÉTODO (PPO) Y SU IMPACTO EN EL PÉNDULO
================================================================================
PPO es estrictamente "On-Policy". Es decir, el agente maneja el péndulo con las creencias 
que tiene *ahora mismo*, graba esos eventos transitorios, ajusta su cerebro sutilmente, y 
borra de inmediato esos históricos cortos porque ya no le representan.

ENTORNO Pendulum-v1 (Contexto Físico):
- ESTADO (s): Vector de gravedad/posición tamaño 3 -> [cos(ang), sin(ang), vel_angular].
- ACCIÓN (a): Impulso real del eje (Torque). Único valor continuo entre -2.0 y 2.0.

¿CÓMO SE PRODUCE EL APRENDIZAJE EN PPO?
1. Distribución Físico-Estocástica: El Actor PPO NO predice números absolutos para el Torque. 
   Estima la "Fuerza Media Deseada" y un "Margen de duda Estándar". Del resultado probabilístico, 
   se extrae el toque mecánico real para enviarlo al motor.
2. Clip de Mutación (Epsilon): Para impedir que el péndulo descubra un balanceo fantástico 
   aleatoriamente y corrompa/sobreescriba su lógica de un golpe, se impone el "Clipping". 
   La política no puede distanciarse un 20% más o menos frente a lo que calculaba hace segundos.
3. Ventaja GAE: Compara si la decisión de dar un empuje específico aportó un resultado que sobrepasó 
   las propias expectativas del Agente para ese estado puntual en el vacío gravitacional.
"""

import math
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym

from datetime import datetime

# ============================================================
#  Bloque 1: Buffer PPO (Memoria Temporal Desechable)
# ============================================================
class RolloutBuffer:
    """
    [CLASE: RolloutBuffer]
    Bandeja efímera PPO. Los historiales del péndulo se desechan antes del siguiente milisegundo de física real.
    """
    def __init__(self):
        """Listas dinámicas listas para capturar series temporales cortas."""
        self.states = []
        self.actions = []
        self.logprobs = [] # Ojo aquí, guardamos la Confianza Estocástica original de Pytorch!
        self.rewards = []
        self.values = []   
        self.dones = []

    def clear(self):
        """[MÉTODO: clear] - Borrado absoluto y agresivo On-Policy."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]

# ============================================================
#  Bloque 2: Actor Probabilístico y Crítico 
# ============================================================
class ActorCritic(nn.Module):
    """
    [CLASE: Actor-Crítico Combinado]
    Mapeo paramétrico desde la trigonometría del estado [cos, sin, vel_angular] 
    hacia la decisión rotacional con varianza controlada.
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),  
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1) # Evalúa la gravedad y el ángulo para predecir si va a caer.
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Emite su predicción media [rango -1 a 1].
        )

        # Matriz de la incertidumbre: El péndulo asume sus dudas matemáticas al inicio, 
        # pero es factor entrenable (baja si confirma que balancea bien).
        self.action_std = nn.Parameter(torch.ones(1, action_dim) * 0.5)

    def select_action(self, state):
        """
        [MÉTODO: select_action]
        Entradas: 
           - state (tensor): Tensor físico instantáneo del frame de caída.
        Salidas: 
           - action (tensor puro): El impulso seleccionado de la Gaussiana de variables.
           - action_logprob (tensor): El rastro de probabilidad estadístico.
           - state_value (tensor): El vaticinio pesimista o realista arrojado por el Crítico.
        """
        mean = self.actor_mean(state)
        std = self.action_std.expand_as(mean)
        
        dist = Normal(mean, std)
        action = dist.sample()
        
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        """
        [MÉTODO: evaluate - El Peritaje PPO Tardío]
        Entradas: 
           - state, action procedentes de batidas viejas del RolloutBuffer.
        Salidas:
           - Regresa las matemáticas recalculadas (Logprobabilidad y Entropía estocástica actualizadas) para que GAE aplique el clipping.
        """
        mean = self.actor_mean(state)
        std = self.action_std.expand_as(mean)
        dist = Normal(mean, std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1) 
        state_values = self.critic(state)
        
        return action_logprobs, state_values.squeeze(-1), dist_entropy

# ============================================================
#  Bloque 3: Agente PPO y GAE Logic
# ============================================================
class PPOAgent:
    """
    [CLASE: PPO Agent]
    Algoritmo unificado maestro que orquesta los pesos de reentrenamiento,
    los K_epochs iterativos sobre sí mismos, y el Clipping natural. 
    """
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Sistema] PPO ejecutándose en: {self.device}")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # Política Ancla o "Vieja", necesaria para ver la evolución y el recorte
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.gamma = 0.99           
        self.gae_lambda = 0.95      
        self.eps_clip = 0.2         
        self.K_epochs = 10          
        self.entropy_coef = 0.01    
        self.value_coef = 0.5       

    def select_action(self, state, buffer):
        """
        [MÉTODO: select_action con Buffer Feed]
        Entradas:
           - state (array NumPy): El paquete vectorial del simulador.
           - buffer (RolloutBuffer): La base de datos.
        Salidas:
           - action (array plano): La fuerza del motor en valor físico directo.
        Lógica: Inyecta paralelamente a la memoria cada iteración del cerebro en vivo "Old".
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, logprob, state_value = self.policy_old.select_action(state)

        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.logprobs.append(logprob)
        buffer.values.append(state_value)

        return action.cpu().data.numpy().flatten()

    def train(self, buffer):
        """
        [MÉTODO: train - Iteración On-Policy]
        Entrada:
           - buffer (RolloutBuffer): Lleno hasta los topes del límite pautado.
        Salidas:
           - Retorna promedios finales Loss Total, Loss del Actor, Loss Crítico.
           
        Lógica del Aprendizaje (GAE y Funciones Surrogate):
        1. Desensambla recuerdos crudos hacia RAM tensorial de forma descendente.
        2. Proyecta Matemáticamente si el péndulo obtuvo ventaja respecto a su calificación histórica.
        3. Realiza "K" re-exámenes intentando maximizar el Gradient.
        4. Las matemáticas Clipping dictaminan que ningún reajuste mutará la capa de la neurona un 20%
           mayor que el frame anterior originario.
        """
        states = torch.cat(buffer.states).detach()
        actions = torch.cat(buffer.actions).detach()
        old_logprobs = torch.cat(buffer.logprobs).detach()
        old_values = torch.cat(buffer.values).squeeze().detach()
        
        rewards = buffer.rewards
        dones = buffer.dones

        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = 0
            else:
                next_val = old_values[i + 1]
                
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - old_values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[i])

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss, total_p_loss, total_v_loss = 0, 0, 0
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(state_values, returns)

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_p_loss += actor_loss.item()
            total_v_loss += critic_loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return total_loss / self.K_epochs, total_p_loss / self.K_epochs, total_v_loss / self.K_epochs

if __name__ == "__main__":
    run_name = f"PPO_Pendulum_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    state_dim = np.prod(env.observation_space.shape) # 3 Vectores de orientación
    action_dim = env.action_space.shape[0]           # 1 Motor central

    agent = PPOAgent(state_dim, action_dim)
    buffer = RolloutBuffer()

    MAX_EPISODES = 500
    UPDATE_TIMESTEP = 2000 

    time_step = 0
    global_episode = 0
    best_reward = -float('inf')

    while global_episode <= MAX_EPISODES:
        state, info = env.reset()
        state = state.flatten()
        ep_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            time_step += 1

            action = agent.select_action(state, buffer)
            
            # EL CLAMPING MECÁNICO PENDULUM: Rango admitido [-2.0, 2.0]
            # Ojo: aquí PPO emite Tanh a -1.0 a 1.0, reescalar multiplicando!
            action_clip = np.clip(action * 2.0, -2.0, 2.0)
            next_state, reward, done, truncated, info = env.step(action_clip)
            next_state = next_state.flatten()
            
            is_terminal = done or truncated

            buffer.rewards.append(reward)
            buffer.dones.append(is_terminal)

            state = next_state
            ep_reward += reward

            if time_step % UPDATE_TIMESTEP == 0:
                print("\n[PPO] Pausa Activa Cómputo Profundo... ", end="")
                loss, p_loss, v_loss = agent.train(buffer)
                print(f"Completado! (Loss Actor: {p_loss:.3f}, Critic: {v_loss:.3f})")
                
                buffer.clear()
                
                writer.add_scalar('Paso/Loss', loss, time_step)

        global_episode += 1
        writer.add_scalar('Episodio/Total_Reward', ep_reward, global_episode)
        
        if ep_reward > best_reward and global_episode > 1:
            best_reward = ep_reward
            torch.save(agent.policy.state_dict(), os.path.join(models_dir, "best_policy.pth"))

        print(f"Episodio: {global_episode}/{MAX_EPISODES} | Rtdo: {ep_reward:5.2f} | Mejor R: {best_reward:.2f}")

    torch.save(agent.policy.state_dict(), os.path.join(models_dir, "final_policy.pth"))
    writer.close()
    env.close()
