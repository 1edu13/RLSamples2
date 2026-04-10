"""
Implementación desde cero de PPO (Proximal Policy Optimization) 
para el entorno highway-env.

Algoritmo PPO:
A отличие de DQN y TD3 (que son algoritmos "Off-Policy" donde se reciclan memorias antiguas), 
PPO es "On-Policy". PPO juega un rato en el entorno, recolecta datos con su nivel actual de
conducción, entrena unas cuantas épocas con ESO MISMO, tira todo a la basura y repite.

Conceptos Clave de PPO:
1. Actor da una "Distribución de Probabilidad" (Media y Desviación).
   El coche no sabe 100% fijo qué hacer, así que da una media (ej: gira 0.5) y una 
   inseguridad (ej: +- 0.1).
2. Clip (Recorte): Cuando encuentra algo bueno, PPO evita volverse loco y cambiar
   el cerebro del coche demasiado rápido. Lo "recorta" para aprender pasito a pasito (estable).
3. GAE (Generalized Advantage Estimation): Forma matemáticamente elegante de calcular
   si una acción concreta fue mejor que el promedio esperado.
"""

import math
import random
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import highway_env

# ============================================================
#  Bloque 1: Buffer PPO (Memoria Temporal Desechable)
# ============================================================
class RolloutBuffer:
    """A diferencia del ReplayBuffer, este buffer SE VACÍA cada vez que terminamos de entrenar"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]

# ============================================================
#  Bloque 2: Actor y Crítico Modificados para Distribuciones
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # El Crítico es como antes, emite el "Value" del estado
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # El Actor emite la "Media" de la acción a tomar
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Salida en rango [-1, 1] porque el coche pide eso
        )

        # La desviación estándar la hacemos un parámetro entrenable
        # Esto significa que el coche aprende a estar "menos inseguro" con el tiempo
        self.action_std = nn.Parameter(torch.ones(1, action_dim) * 0.5)

    def select_action(self, state):
        """Actúa midiendo las probabilidades durante la recolección de datos"""
        mean = self.actor_mean(state)
        # Aseguramos que la Desviación sea positiva (con un Softplus o similar matemático) pero aquí basta con expandir el parámetro
        std = self.action_std.expand_as(mean)
        
        # Creamos una "Campana de Gauss" con esa Media y esa Desviación
        dist = Normal(mean, std)
        
        # Tiramos los dados y sacamos una acción
        action = dist.sample()
        
        # Sacamos el Logaritmo de Probabildad de esa acción (lo pide la matemática de PPO)
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        # Predecimos cuánto vale el estado en el que estamos (Crítico)
        state_value = self.critic(state)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        """Re-evalúa las acciones ya tomadas cuando llega la hora de aprender (PPO Loss)"""
        mean = self.actor_mean(state)
        std = self.action_std.expand_as(mean)
        dist = Normal(mean, std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values.squeeze(-1), dist_entropy

# ============================================================
#  Bloque 3: Agente PPO
# ============================================================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Sistema] PPO ejecutándose en: {self.device}")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # La política "vieja" la usamos para comparar cuánto hemos cambiado y recortar (Clip)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Hiperparámetros de PPO
        self.gamma = 0.99           # Factor descuento
        self.gae_lambda = 0.95      # Suavizado de GAE (ventaja)
        self.eps_clip = 0.2         # Límite dictatorial de PPO: ¡No cambies más de un 20% tu actitud antigua a la vez!
        self.K_epochs = 10         # Número de pasadas o batidas que daremos a los mismos datos nuevos
        self.entropy_coef = 0.01    # Bono al coche por improvisar/explorar
        self.value_coef = 0.5       # Peso de la pérdida del crítico

    def select_action(self, state, buffer):
        """El agente mira el estado, guarda su respuesta mental e la inyecta al buffer"""
        # Aplanar numpy a pytorch
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        action, logprob, state_value = self.policy_old.select_action(state)

        buffer.states.append(state)
        buffer.actions.append(action)
        buffer.logprobs.append(logprob)
        buffer.values.append(state_value)

        # Regresamos la acción en formato simple para el entorno
        return action.cpu().data.numpy().flatten()

    def train(self, buffer):
        # 1. Preparar datos crudos desde el buffer recolectado
        states = torch.cat(buffer.states).detach()
        actions = torch.cat(buffer.actions).detach()
        old_logprobs = torch.cat(buffer.logprobs).detach()
        old_values = torch.cat(buffer.values).squeeze().detach()
        
        rewards = buffer.rewards
        dones = buffer.dones

        # 2. Calcular los "Returns" y el "Advantage" usando GAE
        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = 0
            else:
                next_val = old_values[i + 1]
                
            # ¿Superó la recompensa mis expectativas originales?
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - old_values[i]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[i])

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalizar Advantage las ayuda una barbaridad matemáticamente
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss, total_p_loss, total_v_loss = 0, 0, 0
        
        # 3. Optimizar "K_epochs" veces (La gracia de PPO) exprimiendo la misma data
        for _ in range(self.K_epochs):
            # Recalcular las probabilidades bajo la red *actual* (que se está moviendo)
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)

            # Ratios de cuánto difiere mi opinión actual de mi opinión vieja 
            # de hace 5 minutos (cuando recolecté los datos)
            ratios = torch.exp(logprobs - old_logprobs)

            # Matemáticas de PPO: Funciones "Surrogate"
            surr1 = ratios * advantages
            # Recortamos el ratio para no pasarse del límite de confianza (ej 0.8 a 1.2)
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Pérdida del Actor (El mínimo entre ser optimista o el límite realista)
            actor_loss = -torch.min(surr1, surr2).mean()

            # Pérdida del Crítico
            critic_loss = nn.MSELoss()(state_values, returns)

            # Fórmula final conjunta: Actor + Peso del Crítico - Bono de Exploración(Entropía)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * dist_entropy.mean()

            # Descender por el gradiente
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_p_loss += actor_loss.item()
            total_v_loss += critic_loss.item()

        # Copiar los nuevos pesos como "viejos" para la próxima recolección
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Retornar métricas para logs
        return total_loss / self.K_epochs, total_p_loss / self.K_epochs, total_v_loss / self.K_epochs

# ============================================================
#  Main y Ejecución
# ============================================================
if __name__ == "__main__":
    run_name = f"PPO_Highway_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    video_dir = os.path.join("videos", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)

    # Configuración de Entorno
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.unwrapped.configure({
        "action": {
            "type": "ContinuousAction" # Volante y Gas
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
    #     env, video_folder=video_dir,
    #     episode_trigger=lambda ep: ep % 50 == 0
    # )

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]

    # Instanciamos PPO
    agent = PPOAgent(state_dim, action_dim)
    buffer = RolloutBuffer()

    # Hiperparámetros de control principal
    MAX_EPISODES = 500
    UPDATE_TIMESTEP = 2000 # Cuántos pasos juega PPO en la pista sin parar ANTES de sentarse a estudiar.

    time_step = 0
    global_episode = 0
    best_reward = -float('inf')

    # Lógica On-Policy
    while global_episode <= MAX_EPISODES:
        state, info = env.reset()
        state = state.flatten()
        ep_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            time_step += 1

            # Seleccionar acción ON-POLICY (con su inseguridad actual)
            action = agent.select_action(state, buffer)
            
            # Limitar la acción físicamente
            action_clip = np.clip(action, -1.0, 1.0)
            next_state, reward, done, truncated, info = env.step(action_clip)
            next_state = next_state.flatten()
            
            is_terminal = done or truncated

            # Guardar extras en el buffer
            buffer.rewards.append(reward)
            buffer.dones.append(is_terminal)

            state = next_state
            ep_reward += reward

            # Si alcanzó el límite de pasos, PPO se retira, estudia y limpia la mesa.
            if time_step % UPDATE_TIMESTEP == 0:
                print("\n[PPO] Pausa para Entrenamiento... ", end="")
                loss, p_loss, v_loss = agent.train(buffer)
                print(f"Completado! (Loss Actor: {p_loss:.3f}, Critic: {v_loss:.3f})")
                
                buffer.clear()
                
                # Logs
                writer.add_scalar('Paso/Perdida_Total_PPO', loss, time_step)
                writer.add_scalar('Paso/Perdida_Actor', p_loss, time_step)
                writer.add_scalar('Paso/Perdida_Critico', v_loss, time_step)

        global_episode += 1
        writer.add_scalar('Episodio/Recompensa_Total', ep_reward, global_episode)
        
        # Guardado de modelo óptimo
        if ep_reward > best_reward and global_episode > 1:
            best_reward = ep_reward
            torch.save(agent.policy.state_dict(), os.path.join(models_dir, "best_policy.pth"))

        print(f"Episodio: {global_episode}/{MAX_EPISODES} | Recompensa: {ep_reward:5.2f} | Pista_Step: {time_step} | Mejor: {best_reward:.2f}")

    # Guardado de versión final
    torch.save(agent.policy.state_dict(), os.path.join(models_dir, "final_policy.pth"))

    print("Entrenamiento PPO Finalizado. Modelos guardados.")
    writer.close()
    env.close()
