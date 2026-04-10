import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym
from datetime import datetime

# ==============================================================================
# HIPERPARÁMETROS MAESTROS DEL PPO (TOTALMENTE MODIFICABLES)
# ==============================================================================
# El santo grial de PPO es dominar la amnesia y la agresividad 
# (Overfitting de Policy vs Pérdida de Gradiente).

EPISODES = 2500           # Subimos episodios radicalmente: PPO es ineficiente y necesita horas de práctica.
UPDATE_TIMESTEP = 2000    # Equivalente a guardar 10 Vidas. Mucha variedad física antes de evaluar.
K_EPOCHS = 80             # Clave de la convergencia profunda: Repasa 80 veces los 2000 pasos en RAM.
                          # Exprimimos la data extensa en lugar de tirar pocas vidas a la basura.

LR_ACTOR = 3e-4           # Seguro y sedoso (Mucha cautela para que no haga colapso catastrófico)
LR_CRITIC = 1e-3          # El juez (Crítico) sí debe de ajustarse hiper rápido con tasa agresiva.

GAMMA = 0.99              # Visión macro-futura profunda. Péndulo necesita no caer a largo plazo.
EPS_CLIP = 0.2            # La matemática del "Proximal": Ninguna corrección neuronal sobrepasará el +-20% del valor inicial.
ENTROPY_COEF = 0.01       # Bono por realizar balanceo errático/creativo (Ayuda para exploración estocástica)
VALUE_COEF = 0.5          # Importanci de la predicción de Error del juez dentro de la ecuación final
# ==============================================================================

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def clear(self):
        del self.states[:], self.actions[:], self.logprobs[:]
        del self.rewards[:], self.values[:], self.dones[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Red Actor (Motor de Acciones)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, action_dim), nn.Tanh() # Salida predecible dentro de un Rango [-1, 1]
        )
        
        # Red Crítica (Juez Evaluador de Entorno Predictivo)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Parámetro Estadístico Estocástico de la Campana de Gauss vivo en la Red
        self.action_std = nn.Parameter(torch.ones(1, action_dim) * 0.5)

    def select_action(self, state):
        mean = self.actor_mean(state)
        std = self.action_std.expand_as(mean)
        dist = Normal(mean, std)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_value = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        mean = self.actor_mean(state)
        std = self.action_std.expand_as(mean)
        dist = Normal(mean, std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1) 
        state_values = self.critic(state)
        
        return action_logprobs, state_values.squeeze(-1), dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Preparado en Dispositivo Integrado: {self.device}")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        
        # Dividimos la sutileza de entrenar. Al Juez sí lo dejamos ser agresivo.
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_mean.parameters(), 'lr': LR_ACTOR},
            {'params': [self.policy.action_std], 'lr': LR_ACTOR},
            {'params': self.policy.critic.parameters(), 'lr': LR_CRITIC}
        ])
        
        # Ancla estática PPO
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state, buffer):
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, logprob, state_value = self.policy_old.select_action(state_t)

        buffer.states.append(state_t)
        buffer.actions.append(action)
        buffer.logprobs.append(logprob)
        buffer.values.append(state_value)

        return action.cpu().data.numpy().flatten()

    def train(self, buffer):
        states = torch.cat(buffer.states).detach()
        actions = torch.cat(buffer.actions).detach()
        old_logprobs = torch.cat(buffer.logprobs).detach()
        old_values = torch.cat(buffer.values).squeeze().detach()
        rewards, dones = buffer.rewards, buffer.dones

        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            next_val = 0 if i == len(rewards) - 1 else old_values[i + 1]
            delta = rewards[i] + GAMMA * next_val * (1 - dones[i]) - old_values[i]
            gae = delta + GAMMA * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[i])

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        
        # Extraemos profundo K_EPOCHS veces el cálculo usando Surrogate clipping.
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            
            # Penalización negativa
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * dist_entropy.mean()

            # Descendiente retro-gradiente real
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return total_loss / K_EPOCHS

def main():
    run_name = f"PPO_Convergencia_Real_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make("Pendulum-v1")
    
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)
    buffer = RolloutBuffer()

    time_step = 0
    best_reward = -float('inf')

    print(f"\n{'='*60}\n SPRINT PPO DEFINITIVO DE CURVA SUPERIOR \n{'='*60}")
    print(f"Episodios Set: {EPISODES} | Bloques Memoria (Update_Steps): {UPDATE_TIMESTEP} | Re-Lecturas (K_Epochs): {K_EPOCHS}\n")

    for global_episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        state = state.flatten()
        ep_reward = 0
        done, truncated = False, False

        while not (done or truncated):
            time_step += 1
            action = agent.select_action(state, buffer)
            
            # La salida de la capa convolucional Tanh es en rango [-1.0 a 1.0].
            # Lo escalamos puramente al motor físico del péndulo que es -2 a 2.
            action_clip = np.clip(action * 2.0, -2.0, 2.0)
            next_state, reward, done, truncated, _ = env.step(action_clip)
            next_state = next_state.flatten()
            
            buffer.rewards.append(reward)
            buffer.dones.append(done or truncated)

            state = next_state
            ep_reward += reward

            # Feedback: Explotar Buffer cada X Episodios Acumulados
            if time_step % UPDATE_TIMESTEP == 0:
                print("\n[Mente PPO] Activando retro-propagación de gradiente (Actualizando Juez y Actor)...", end="")
                loss = agent.train(buffer)
                print(f" ¡Terminado! (Total Loss: {loss:.3f})")
                buffer.clear()
                writer.add_scalar('PPO_Paso_Interno/Loss', loss, time_step)

        # Volcado Gráfico Vectorial
        writer.add_scalar('PPO_Episodios/Reward', ep_reward, global_episode)
        
        # Checkpoint Automático del Mejor Descubrimiento Histórico
        if ep_reward > best_reward and global_episode > 50:
            best_reward = ep_reward
            torch.save(agent.policy.state_dict(), os.path.join(models_dir, "best_policy.pth"))

        if global_episode % 10 == 0 or global_episode == 1:
            print(f"Episodio PPO: {global_episode:4d}/{EPISODES} | Puntos Vivos Reales: {ep_reward:7.2f} | Pasos motor: {time_step}")

    torch.save(agent.policy.state_dict(), os.path.join(models_dir, "final_policy.pth"))
    env.close()
    writer.close()
    
    print(f"\n[ÉXITO EXTREMO] Entrenamiento On-Policy Finalizado.")
    print(f"[REPORTE] Tienes toda la curva salvada en tu Tensorboard ('runs/'{run_name}).")
    print(f"[REPORTE] Además el mejor cerebro ha quedado grabado en {models_dir}.")

if __name__ == "__main__":
    main()
