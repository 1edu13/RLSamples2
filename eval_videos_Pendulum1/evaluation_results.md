# Informe de Resultados: Evaluación Pendulum-v1

Este informe resume los resultados de la evaluación de cuatro diferentes algoritmos de Aprendizaje por Refuerzo (DQN, TD3, PPO y SAC) ejecutados secuencialmente en el entorno **Pendulum-v1**. La evaluación ha consistido en probar cada uno de los mejores modelos durante 10 episodios y medir tanto su recompensa media como su tasa de fallo (porcentaje de choques o caídas).

## Resumen de Rendimiento

| Algoritmo | Recompensa Media | Tasa de Choques (%) | Rendimiento Relativo |
|:----------|:----------------:|:-------------------:|:---------------------|
| **TD3**   | **-132.61**      | 10.0%               | 🏆 **1º (El Mejor)**  |
| **DQN**   | -146.20          | 10.0%               | 🥈 2º Lugar           |
| **SAC**   | -159.21          | 20.0%               | 🥉 3º Lugar           |
| **PPO**   | -1043.24         | 100.0%              | ❌ 4º (El Peor)       |

> [!TIP]
> **El Modelo Más Sólido: TD3**
> El agente entrenado bajo TD3 (Twin Delayed DDPG) ha demostrado ser el más estable y eficiente, obteniendo la mejor recompensa de la sesión (-132.61, muy cercana al umbral de control perfecto del péndulo) y empatando con el mejor índice de seguridad (10% de tasa de fallos).

> [!WARNING]
> **El caso de PPO: Requiere Ajustes**
> El modelo PPO presenta una puntuación extremadamente baja (-1043.24) y una tasa de fallo del 100%. Esto suele indicar que PPO no llegó a convergir durante sus episodios de entrenamiento. Como algoritmo "On-Policy", PPO requiere generalmente muchísimas más iteraciones reales o una optimización muy fina de hiperparámetros respecto a algoritmos "Off-Policy" (como TD3 y SAC) para lograr buenos resultados en control continuo con Pendulum.

## Análisis Detallado

1. **TD3 vs SAC (Algoritmos de Control Continuo Avanzado):** 
   Ambos son excelentes para control continuo. En esta ejecución, TD3 resultó ligeramente superior, con un aprendizaje de la política aparentemente más conservador y preciso, sufriendo la mitad de choques (10% vs 20%). Esto hace que TD3 sea la opción preferente en este problema si se busca control preciso y sin riesgos bruscos de exploración terminal.
   
2. **El rendimiento atípico de DQN:** 
   DQN está diseñado de base para entornos *discretos*, sin embargo mediante una técnica de "discretización del espacio de acciones", logró un sorprendentemente buen 2º lugar (-146.20). Esto demuestra que para problemas de baja dimensionalidad como Pendulum (donde el control puede simplificarse a acelerar fuerte a un lado o al otro), un buen DQN sigue compitiendo muy de cerca contra el Estado del Arte de los espacios continuos.
   
3. **El desafío físico del péndulo:** 
   Recordemos que en `Pendulum-v1` la recompensa teórica máxima ronda los `0.0` puntos (si el péndulo ya nace vertical y se mantiene ahí con perfecto equilibrio y torque = 0). Las penalizaciones se basan en el ángulo de caída y la cantidad de fuerza ("torque") que el motor gasta para intentar subirlo. Cualquier promedio por encima de `-200` generalmente se considera una prueba de que el agente **conoce perfectamente cómo balancear el péndulo de pie**.

## Gráfico de Comparación

A continuación, la gráfica histórica con los resultados obtenidos por los modelos al ser enfrentados en evaluación directa:

![Gráfico comparativo de rendimientos](C:\Users\emped\.gemini\antigravity\brain\0392147b-d465-4301-a4c7-3d320bedfd77\comparacion_resultados.png)

## Hiperparámetros de Entrenamiento

A continuación se detallan los hiperparámetros utilizados para entrenar cada uno de los modelos evaluados en este experimento. Todos los modelos se entrenaron por un máximo de **500 episodios**, pero difieren en sus configuraciones base.

### TD3 (Twin Delayed DDPG)
- **Learning Rate**: `3e-4` (Actor y Crítico)
- **Gamma (Descuento)**: `0.99` | **Tau**: `0.005`
- **Batch Size**: `256` | **Buffer**: `100,000`
- **Policy Frequency (Delay)**: `2`
- **Policy Noise**: `0.2` | **Noise Clip**: `0.5` | **Exploration Noise**: `0.1`
- **Warmup Steps**: `5000`

### DQN (Deep Q-Network - Acción Discretizada)
- **Learning Rate**: `1e-3`
- **Gamma**: `0.99`
- **Batch Size**: `64` | **Buffer**: `50,000`
- **Epsilon (Exploración)**: Inicio `1.0`, Decaimiento `0.995`, Mínimo `0.05`
- **Target Update Frequency**: `1000` pasos

### SAC (Soft Actor-Critic)
- **Learning Rate**: `3e-4` (Actor, Crítico y Ajuste de Entropía Alpha)
- **Gamma**: `0.99` | **Tau**: `0.005`
- **Batch Size**: `256` | **Buffer**: `100,000`
- **Target Entropy**: `-1.0`
- **Warmup Steps**: `5000`

### PPO (Proximal Policy Optimization)
- **Learning Rate**: `3e-4`
- **Gamma**: `0.99` | **GAE Lambda**: `0.95`
- **Update Timestep (Batch)**: `2000` pasos
- **K_epochs (Pasadas por batch)**: `10`
- **Epsilon Clip**: `0.2`
- **Coef. Crítico (Value)**: `0.5` | **Coef. Entropía**: `0.01`
