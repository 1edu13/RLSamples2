import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from datetime import datetime

# ==============================================================================
# HIPERPARÁMETROS MAESTROS DEL PPO (TOTALMENTE MODIFICABLES)
# ==============================================================================

EPISODES = 2500           
UPDATE_TIMESTEP = 2000    
K_EPOCHS = 80             
                          
LR_ACTOR = 3e-4           
LR_CRITIC = 1e-3          

GAMMA = 0.99              
EPS_CLIP = 0.2            
ENTROPY_COEF = 0.01       
VALUE_COEF = 0.5          
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
        
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, action_dim), nn.Tanh() 
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        
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
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_mean.parameters(), 'lr': LR_ACTOR},
            {'params': [self.policy.action_std], 'lr': LR_ACTOR},
            {'params': self.policy.critic.parameters(), 'lr': LR_CRITIC}
        ])
        
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
        
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return total_loss / K_EPOCHS

def main():
    run_name = f"PPO_Convergencia_Real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("runs_experimentos", run_name)
    models_dir = os.path.join("models_experimentos", run_name)
    os.makedirs(models_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make("Pendulum-v1")
    
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)
    buffer = RolloutBuffer()

    time_step = 0
    best_reward = -float('inf')
    rewards_history = []  # <--- SE GUARDAN LAS PUNTUACIONES PARA LA GRÁFICA MATPLOTLIB

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
            
            action_clip = np.clip(action * 2.0, -2.0, 2.0)
            next_state, reward, done, truncated, _ = env.step(action_clip)
            next_state = next_state.flatten()
            
            buffer.rewards.append(reward)
            buffer.dones.append(done or truncated)

            state = next_state
            ep_reward += reward

            if time_step % UPDATE_TIMESTEP == 0:
                print("\n[Mente PPO] Activando retro-propagación de gradiente (Actualizando Juez y Actor)...", end="")
                loss = agent.train(buffer)
                print(f" ¡Terminado! (Total Loss: {loss:.3f})")
                buffer.clear()
                writer.add_scalar('PPO_Paso_Interno/Loss', loss, time_step)

        writer.add_scalar('PPO_Episodios/Reward', ep_reward, global_episode)
        rewards_history.append(ep_reward)  # Acumular para pintar a lo último
        
        if ep_reward > best_reward and global_episode > 50:
            best_reward = ep_reward
            torch.save(agent.policy.state_dict(), os.path.join(models_dir, "best_policy.pth"))

        if global_episode % 10 == 0 or global_episode == 1:
            print(f"Episodio PPO: {global_episode:4d}/{EPISODES} | Puntos Vivos Reales: {ep_reward:7.2f} | Pasos motor: {time_step}")

    torch.save(agent.policy.state_dict(), os.path.join(models_dir, "final_policy.pth"))
    env.close()
    writer.close()
    
    # ==============================================================================
    # FASE 2: EVALUACIÓN POST-ENTRENAMIENTO AL MEJOR CEREBRO
    # ==============================================================================
    print("\n\n" + "="*60)
    print(" FASE 2: EVALUACIÓN PURA AL MEJOR CEREBRO DESCUBIERTO")
    print("="*60)

    eval_env = gym.make("Pendulum-v1")
    eval_agent = PPOAgent(state_dim, action_dim)
    
    # Inyectamos de vuelta el cerebro Campeón a este agente huésped
    best_path = os.path.join(models_dir, "best_policy.pth")
    if os.path.exists(best_path):
        eval_agent.policy.load_state_dict(torch.load(best_path, map_location=eval_agent.device, weights_only=True))
    
    eval_rewards = []
    print("\nEvaluando 5 vidas de forma 100% determinista y clínica (Sin ruido gaussiano de exploración)...")
    
    for test in range(5):
        state, _ = eval_env.reset()
        state = state.flatten()
        ep_reward = 0
        done, truncated = False, False
        
        while not (done or truncated):
            # IMPORTANTE: En Evaluación Pura, extirpamos la capa estocástica. Acudimos EXCLUSIVAMENTE a la MEDIA.
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(eval_agent.device)
            pura_media_accion = eval_agent.policy.actor_mean(state_tensor)
            
            action = pura_media_accion.cpu().data.numpy().flatten()
            action_clip = np.clip(action * 2.0, -2.0, 2.0)
            
            state, reward, done, truncated, _ = eval_env.step(action_clip)
            state = state.flatten()
            ep_reward += reward
            
        eval_rewards.append(ep_reward)
    
    eval_env.close()
    media_eval = np.mean(eval_rewards)
    print(f"\n -> RESULTADO DEL PERITAJE FINAL (Promedio de las 5 Evaluaciones): {media_eval:.2f} PUNTOS.")
    
    # ==============================================================================
    # FASE 3: GENERACIÓN AUTOMÁTICA DE LA GRÁFICA (.PNG) VISTOSA
    # ==============================================================================
    print("\nGenerando renderizado gráfico de la Curva Intelectual...")
    plt.figure(figsize=(11, 7))
    
    def smooth(scalars, weight=0.9):
        if not scalars: return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.plot(rewards_history, label='Recompensa Cruda por Episodio', color='limegreen', alpha=0.3)
    plt.plot(smooth(rewards_history, weight=0.95), label='Tendencia Principal Suavizada', color='whitesmoke', linewidth=2.5)

    plt.title('Curva de Aprendizaje Definitiva: PPO (Entrenamiento + Evaluación Final)')
    plt.xlabel('Episodios Entrenados On-Policy')
    plt.ylabel('Puntuación Lograda (0 = Equilibrio Óptimo)')
    
    plt.axhline(0, color='gold', linestyle='-', alpha=0.9, linewidth=2, label='Perfección Vertical Inalcanzable (0)')
    
    # Anotación llamativa incrustada en la gráfica mostrando el veredicto del examen
    plt.annotate(f' Veredicto Test Limpio:\n  {media_eval:.1f} pts ', 
                 xy=(EPISODES, media_eval), 
                 xytext=(EPISODES * 0.70, max(-1500, media_eval - 400)), # Evitar salir de cámara
                 arrowprops=dict(facecolor='deepskyblue', shrink=0.08, width=3, headwidth=10),
                 fontsize=13, color='white', fontweight='bold', 
                 bbox=dict(facecolor='black', edgecolor='deepskyblue', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.legend(loc='lower right')
    plt.style.use('dark_background')
    plt.grid(True, alpha=0.3)
    
    graph_path = os.path.join(models_dir, "curva_aprendizaje_ppo_evaliada.png")
    plt.savefig(graph_path, format="png", bbox_inches='tight', dpi=300)
    
    print(f"\n[ÉXITO EXTREMO] Todo el pipeline ha concluido.")
    print(f"[REPORTE] Tienes toda la curva y evaluación visual salvada en: {os.path.abspath(graph_path)}")


if __name__ == "__main__":
    main()
