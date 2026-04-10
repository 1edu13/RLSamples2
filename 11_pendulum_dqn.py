"""
Implementación desde cero de DQN (Deep Q-Network) adaptado al entorno físico Pendulum-v1.

================================================================================
OVERVIEW DEL MÉTODO (DQN - Deep Q-Network) Y ENTORNO PENDULUM
================================================================================
DQN combina "Q-Learning" con Redes Neuronales Profundas para aprender la función 
Q(s, a). Esta función evalúa la "Calidad" (recompensa futura) de tomar una acción 
discreta 'a' en un estado 's'.

ENTORNO Pendulum-v1 (Contexto Físico):
- ESTADO (s): No son imágenes. Es un vector numérico de 3 elementos:
  [cos(theta), sin(theta), theta_dot], donde theta es el ángulo del péndulo 
  y theta_dot es su velocidad angular.
- ACCIÓN (a): Originalmente, Pendulum requiere una fuerza continua de Torque en el eje 
  (rango [-2.0 a 2.0]). Como DQN fue diseñado para videojuegos con botones discretos, 
  aquí "discretizamos" la acción en 3 posibles botones:
  - Botón 0: Torque máximo a la izquierda (-2.0)
  - Botón 1: Torque nulo / Caída libre (0.0)
  - Botón 2: Torque máximo a la derecha (+2.0)

¿CÓMO SE PRODUCE EL APRENDIZAJE?
1. Exploración: El agente del péndulo empieza probando torques al azar para comprender las físicas.
2. Memoria: Almacena sus vivencias en un "Replay Buffer".
3. Ecuación de Bellman: Entrena a la red para que sus predicciones del valor Q igualen a la
   recompensa inmediata más el mejor Q-value predicho para el siguiente instante.
4. Target Net: Usa una red clonada congelada para estabilizar las estimaciones matemáticas futuras.
"""

import math
import random
import os
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

# ============================================================
#  Bloque 1: Memoria de Experiencia (Replay Buffer)
# ============================================================
class ReplayBuffer:
    """
    [CLASE: ReplayBuffer]
    El "Disco Duro" temporal donde el agente guarda transiciones del péndulo.
    """
    def __init__(self, capacity=100000):
        """
        Entradas: 
            capacity (int): Máximo de transiciones guardadas antes de sobrescribir.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        [MÉTODO: push - Guardar experiencia]
        Entradas (Inputs):
          - state (array): Estado del péndulo [cos, sin, vel_angular] (Tamaño 3).
          - action (int): La acción discreta tomada (botón 0, 1 o 2).
          - reward (float): Recompensa obtenida de la física (negativa según la caída del péndulo).
          - next_state (array): Estado resultante del péndulo (Tamaño 3).
          - done (bool): Indica si el episodio físico llegó a su fin (Límite de tiempo alcanzado).
        Salidas: Ninguna.
        Lógica: Simplemente almacena la tupla completa en la cola.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        [MÉTODO: sample - Extracción de recuerdos]
        Entradas:
          - batch_size (int): Cantidad de vivencias a extraer de golpe.
        Salidas (Outputs):
          - Tupla de matrices numpy empaquetadas (states, actions, rewards, next_states, dones).
        Lógica:
          Extrae aleatoriamente experiencias. El desorden aleatorio rompe la correlación del péndulo 
          (caer consecutivamente) para que la red aprenda la física de forma descontextualizada.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state), np.array(action), np.array(reward, dtype=np.float32),
            np.array(next_state), np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================
#  Bloque 2: Red Neuronal (Q-Network)
# ============================================================
class QNetwork(nn.Module):
    """
    [CLASE: QNetwork]
    Es la mente matemática del péndulo. Mapea la física (estado numérico) a 
    calificaciones Q para cada posible acción de Torque.
    """
    def __init__(self, input_dim, output_dim):
        """
        Entradas:
          - input_dim (int): 3 (Los datos del péndulo).
          - output_dim (int): 3 (Los botones a pulsar: Torq -, Torq 0, Torq +).
        """
        super(QNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim) 
        )

    def forward(self, x):
        """
        [MÉTODO: forward - Inferencia Neuronal]
        Entradas:
          - x (tensor): Vector(es) del estado físico del péndulo.
        Salidas:
          - Tensor con las tres puntuaciones Q (Una por cada torque discreto).
        Lógica: Evaluación matemática Feed-Forward pura.
        """
        x = self.flatten(x)
        return self.net(x)

# ============================================================
#  Bloque 3: Agente DQN
# ============================================================
class DQNAgent:
    """
    [CLASE: DQNAgent]
    El núcleo que entrelaza la Red Q, la Red Objetivo, el Optimizador y la 
    regla de actualización de Bellman aplicada a las físicas estáticas.
    """
    def __init__(self, state_shape, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Sistema] DQN ejecutándose sobre el dispositivo: {self.device}")
        
        self.input_dim = int(np.prod(state_shape))
        self.action_dim = action_dim
        
        self.main_net = QNetwork(self.input_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.input_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval() 
        
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss() 
        
        self.gamma = 0.99           # Ansiedad por recompensas retardadas (cuánto le importa balancearlo bien a largo plazo).
        self.batch_size = 64        # Set de muestras simultáneas procesadas en GPU.
        self.epsilon = 1.0          # Curva de exploración: Forzará Torques alocados al principio.
        self.epsilon_min = 0.05     
        self.epsilon_decay = 0.995  
        self.train_step_count = 0
        self.target_update_freq = 1000 
        
    def select_action(self, state, evaluate=False):
        """
        [MÉTODO: select_action - Toma de decisión]
        Entradas:
          - state (array): Los 3 floats de la física instantánea del péndulo.
          - evaluate (bool): Indicador que desactiva los movimientos asíncronos curiosos.
        Salidas:
          - (int): Un índice de acción discreto (0, 1 o 2).
        Lógica: Epsilon-Greedy. Si quiere explotar conocimiento, pasa los 3 datos por la red,
        visualiza las 3 notas Q posibles para los 3 torques, y se queda con el mejor dictaminado.
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad(): 
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_net(state_t)
            return q_values.argmax().item()
            
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def train(self, replay_buffer):
        """
        [MÉTODO: train - Flujo del Aprendizaje]
        Entradas: 
          - replay_buffer (ReplayBuffer): Acceso libre a la memoria física histórica.
        Salidas:
          - (float o None): Devuelve el error ("Loss") que representaba su ignorancia antes del ajuste, o nada si está vacío.
        
        Lógica del Aprendizaje DQN:
        1. Roba 64 fragmentos de historia aleatoria sobre caídas o balanceos del péndulo.
        2. Q-Actual: Le pregunta a la Red Principal qué opinaba (calidad teórica) de haber aplicado
           ese exacto torque impulsivo en ese instante 's'.
        3. Q-Target: Calcula mecánicamente la "Verdad Ponderada Final": (Recompensa del instante del simulador) + 
           Lo que vaticina la red congelada sobre la máxima excelencia del estado que sucedió luego 's_next'.
        4. Distancia/Loss: Si la Q-Actual dijo 5 y la Verdad Ponderada fue -2, el error cuadrático aprieta los tensores de PyTorch.
        5. Gradient Descent: Se purgan gradientes y se inyecta la realidad matemática corrigiendo lentamente las capas.
        """
        if len(replay_buffer) < self.batch_size:
            return None 
            
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q = self.main_net(states_t).gather(1, actions_t) 
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q
            
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad() 
        loss.backward()            
        nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=10)
        self.optimizer.step()      
        
        self.train_step_count += 1
        
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
            
        return loss.item()

if __name__ == "__main__":
    run_name = f"DQN_Pendulum_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)

    # Entorno Físico Simulado Pendulum-v1 (Simulador de palo con gravedad)
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    
    state_shape = env.observation_space.shape  # Dimensiones de las métricas (3)
    action_dim = 3 # Acciones discretizadas 
    
    agent = DQNAgent(state_shape, action_dim)
    memory = ReplayBuffer(capacity=50000)

    EPISODIOS = 500
    global_step = 0
    best_reward = -float('inf')
    
    for episode in range(1, EPISODIOS + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_steps = 0
        done, truncated = False, False
        
        while not (done or truncated):
            global_step += 1
            
            # EL ABUELO DQN TOMANDO EL CONTROL DEL PÉNDULO CONTINUO:
            action = agent.select_action(state)
            
            # Mapeamos botón a Torque Físico: 0 -> (-2.0), 1 -> (0.0), 2 -> (+2.0)
            mapped_action = [[-2.0], [0.0], [2.0]][action]
            next_state, reward, done, truncated, info = env.step(mapped_action)
            
            # Acopiamos los vectores geométricos del péndulo en la memoria
            memory.push(state, action, reward, next_state, (done or truncated))
            
            state = next_state
            episode_reward += reward
            agent.update_epsilon() 
            
            loss = agent.train(memory)
            if loss is not None:
                episode_loss += loss
                loss_steps += 1
                
                writer.add_scalar('Paso/Loss', loss, global_step)
                writer.add_scalar('Paso/Epsilon', agent.epsilon, global_step)

        avg_loss = (episode_loss / loss_steps) if loss_steps > 0 else 0.0

        writer.add_scalar('Episodio/Total_Reward', episode_reward, episode)
        writer.add_scalar('Episodio/Avg_Loss', avg_loss, episode)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.main_net.state_dict(), os.path.join(models_dir, "best_model.pth"))

        print(f"Ep 1/500 | Rtdo: {episode_reward:5.2f} | Pasos: {global_step} | Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.main_net.state_dict(), os.path.join(models_dir, "final_model.pth"))
    print("DQN en Pendulum finalizado")
    writer.close()
    env.close()
