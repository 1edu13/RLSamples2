"""
Implementación desde cero de DQN (Deep Q-Network) para el entorno highway-env.
Diseñado para correr de manera óptima en tu hardware (PyTorch utilizará CPU o GPU, lo
que encuentre, pero la CPU será muy potente para este entorno).
También guarda los logs para TensorBoard y videos MP4 del entrenamiento.

Algoritmo DQN:
El algoritmo DQN asume un espacio de acción DISCRETO. Las acciones son decisiones
finitas (ej: 0=cambiar izquierda, 1=mantener carril, 2=cambiar derecha, 3=acelerar, 4=frenar).
Aprende una función Q(estado, acción) que nos dice qué tanta recompensa futura
obtendremos si tomamos una acción concreta en un estado en particular.

Incluye:
1. Replay Buffer: Memoria de experiencias para romper correlación de datos temporales.
2. Target Network: Una copia congelada de la red que da estabilidad al cálculo del objetivo.
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
import highway_env

# ============================================================
#  Bloque 1: Memoria de Experiencia (Replay Buffer)
# ============================================================
class ReplayBuffer:
    """
    Guarda las transiciones (estado, acción, recompensa, próximo estado, done) cruzadas.
    Permite sacar muestreos (batches) aleatorios para que la red neuronal no "olvide"
    experiencias pasadas y no memorice solo lo último que vio.
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Escoge un "puñado" aleatorio de memoria
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ============================================================
#  Bloque 2: Red Neuronal (Q-Network)
# ============================================================
class QNetwork(nn.Module):
    """
    Red Neuronal FeedForward (Perceptrón Multicapa).
    Recibe el estado actual (la foto de la carretera) y devuelve un valor "Q" simulado
    para cada posible acción. La acción con mayor "Q" es la que decidimos tomar.
    """
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # highway-env da una matriz (vehículos x características). La "aplanamos"
        self.flatten = nn.Flatten()
        
        # Red de 3 capas densamente conectadas
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim) # Una salida para cada acción posible
        )

    def forward(self, x):
        # Aplanar tensor si viene en forma de matriz 2D por estado
        x = self.flatten(x)
        return self.net(x)

# ============================================================
#  Bloque 3: Agente DQN
# ============================================================
class DQNAgent:
    def __init__(self, state_shape, action_dim):
        # 1. Configurar Hardware
        # Intenta usar la mejor tarjeta o acelerador disponible
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"[Sistema] DQN ejecutándose sobre el dispositivo: {self.device}")
        
        # Tamaño de entrada aplanado. Ej: Si shape es (5, 5), input_dim = 25
        self.input_dim = int(np.prod(state_shape))
        self.action_dim = action_dim
        
        # 2. Instanciar Redes Neuronales
        # main_net es la que aprende activamente
        self.main_net = QNetwork(self.input_dim, self.action_dim).to(self.device)
        # target_net se congela y solo se actualiza cada X pasos (Da mucha estabilidad matemática)
        self.target_net = QNetwork(self.input_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        
        # 3. Optimizador (Adam = Gradient Descent con inercia mejorada)
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        
        # 4. Hiperparámetros del Algoritmo
        self.gamma = 0.99           # Factor de descuento futuro (Ponderar el futuro)
        self.batch_size = 64        # Muestras por corrección
        self.epsilon = 1.0          # Exploración inicial (100% de acciones aleatorias)
        self.epsilon_min = 0.05     # Nunca bajar del 5% de exploración
        self.epsilon_decay = 0.995  # Qué tan rápido decae la exploración por paso
        self.train_step_count = 0
        self.target_update_freq = 1000 # Actualizar red objetivo cada 1000 pasos
        
    def select_action(self, state, evaluate=False):
        """Estrategia Epsilon-Greedy: Elige aleatorio a veces para explorar, o usa la red para explotar"""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_net(state_t)
            # Retorna el índice de la acción con la puntuación Q más alta (argmax)
            return q_values.argmax().item()
            
    def update_epsilon(self):
        """Reduce gradualmente la probabilidad de hacer locuras (exploración a explotación)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def train(self, replay_buffer):
        """
        Paso matemático principal: Q-Learning con Retrollamada de Error (Backpropagation)
        Fórmula de Bellman: Q(s,a) = R + Gamma * max(Q_target(s', a'))
        """
        if len(replay_buffer) < self.batch_size:
            return None # No hay suficiente experiencia en memoria
            
        # 1. Sacar memoria aleatoria
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
        
        # Pasar tensores al hardware correspondiente
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 2. Predicción Actual: ¿Qué valor Q predecía mi red para esta acción?
        q_values = self.main_net(states_t)
        current_q = q_values.gather(1, actions_t) # Tomar exactamente el Q de la acción seleccionada
        
        # 3. Predicción Objetivo (Target): ¿Cuál era la "verdadera" recompensa futura vista tras ejecutarla?
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            # Solo consideramos futuro si NO hemos chocado (done=1 corta el futuro a 0)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q
            
        # 4. Calcular el Error (Mean Squared Error) entre la predicción y el teórico real
        loss = self.loss_fn(current_q, target_q)
        
        # 5. Optimización (Ajuste de Pesos de las neuronas)
        self.optimizer.zero_grad() # Limpiar gradientes anteriores
        loss.backward()            # Propagación de error hacia atrás
        nn.utils.clip_grad_norm_(self.main_net.parameters(), max_norm=10) # Control de estabilidad (explosión de gradiente)
        self.optimizer.step()      # Mover pesos
        
        self.train_step_count += 1
        
        # 6. Actualización de red congelada periódica
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
            
        return loss.item()

# ============================================================
#  Main y Configuración de Tensorboard + Wrappers
# ============================================================
if __name__ == "__main__":
    # --- Configurar Carpetas para Videos, Logs y Modelos ---
    run_name = f"DQN_Highway_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    video_dir = os.path.join("videos", run_name)
    models_dir = os.path.join("models", run_name)
    os.makedirs(models_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[Tensorboard] Guarda corriendo datos en la carpeta: {log_dir}")
    print("Para ver los logs, corre en otra terminal: tensorboard --logdir=runs")

    # --- Crear Entorno (Simulador Highway) ---
    # render_mode="rgb_array" permite poder grabar video sin renderizar en pantalla
    env = gym.make("highway-v0", render_mode="rgb_array")
    
    # 1. Configurar Entorno a DISCRETO (Para DQN)
    env.unwrapped.configure({
        "action": {
            "type": "DiscreteMetaAction",
        },
        "observation": {
            "type": "Kinematics", # Devuelve matriz VxF (5 vehículos, 5 propiedades)
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True
        },
        "duration": 40  # Duración máxima de un episodio sin estrellarse
    })
    env.reset() # Inicializar nueva configuración
    
    # 2. Wrapper para grabación (Graba 1 vídeo de muestra cada cierta cantidad de episodios)
    # env = gym.wrappers.RecordVideo(
    #     env, 
    #     video_folder=video_dir, 
    #     episode_trigger=lambda ep_id: ep_id % 50 == 0 # Graba cada 50 episodios
    # )

    # --- Inicializar Agente y Memoria ---
    state_shape = env.observation_space.shape  # ej: (5,5)
    action_dim = env.action_space.n            # ej: 5 comandos
    
    agent = DQNAgent(state_shape, action_dim)
    memory = ReplayBuffer(capacity=50000)

    # --- Bucle Principal de Entrenamiento ---
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
            
            # 1. Decidir y Actuar
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Cuidado: highway-env llama "crashed" o terminación si te sales/chocas.
            # Convertimos truncated temporalmente por seguridad.
            is_terminal = done or truncated
            
            # 2. Guardar a Memoria
            memory.push(state, action, reward, next_state, is_terminal)
            
            state = next_state
            episode_reward += reward
            agent.update_epsilon() # Vamos reduciendo locura
            
            # 3. Entrenar a cada paso (si hay suficiente memoria)
            loss = agent.train(memory)
            if loss is not None:
                episode_loss += loss
                loss_steps += 1
                writer.add_scalar('Paso_de_Entrenamiento/Perdida (Loss)', loss, global_step)
                writer.add_scalar('Paso_de_Entrenamiento/Exploracion (Epsilon)', agent.epsilon, global_step)

        # Promediar Perdida del episodio
        avg_loss = (episode_loss / loss_steps) if loss_steps > 0 else 0.0

        # Escribir Log del Episodio entero
        writer.add_scalar('Resultados_Episodio/Recomepnsa_Total', episode_reward, episode)
        writer.add_scalar('Resultados_Episodio/Perdida_Media', avg_loss, episode)
        
        # Guardado de modelo óptimo (Mejor rendimiento del episodio entero)
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.main_net.state_dict(), os.path.join(models_dir, "best_model.pth"))
            # print(" -> [Guardado] Nuevo mejor modelo!")

        # Printear para tener métrica en Consola
        print(f"Episodio: {episode}/{EPISODIOS} | "
              f"Recompensa: {episode_reward:5.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Pasos Globales: {global_step} | "
              f"Perdida Media: {avg_loss:.4f} | "
              f"Mejor Rtdo: {best_reward:.2f}")

    # Guardar la última versión general terminada
    torch.save(agent.main_net.state_dict(), os.path.join(models_dir, "final_model.pth"))
    
    print("¡Entrenamiento Completo! Revisa la carpeta 'videos' y 'models'.")
    writer.close()
    env.close()
