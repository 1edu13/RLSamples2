# Entendiendo Profundamente el Aprendizaje en Reinforcement Learning

## Introducción al Ciclo de Aprendizaje
En Reinforcement Learning (RL), todos los agentes operan bajo el mismo principio básico: **Interactúan con el entorno, observan resultados (recompensas/castigos) y ajustan su cerebro (la red neuronal) para tomar mejores decisiones en el futuro.**

Existen diversas filosofías para lograr esto, y los algoritmos principales (DQN, PPO, TD3 y SAC) son como estudiantes con métodos de estudio completamente distintos.

---

## 1. Los 4 Algoritmos Principales

### DQN (Deep Q-Network): *El Memorizador Matemático*
*(Basado en Valor | Off-policy | Acciones Discretas)*

*   **La Intuición:** DQN entrena a una red neuronal para que funcione como una enorme tabla de puntajes. Para cada estado del entorno busca encontrar qué valor tiene cada movimiento posible.
*   **¿Cómo aprende de verdad?:** Utiliza un **"Objetivo Móvil" (Q-Target)**. Cuando realiza una acción y ve la recompensa, ajusta la suposición inicial de su red neuronal comparándola con la suma de la recompensa obtenida más la máxima recompensa futura calculada desde el nuevo estado. Esa diferencia es el "TD-Error", y usa descenso de gradiente para corregir la red minimizando ese error.
*   **Clave de Estabilidad:** Usa un **Replay Buffer** para mezclar y aprender de recuerdos pasados, y una **Red Target** que congela su red objetivo durante unos instantes para que no colapse mientras aprende continuamente.

### PPO (Proximal Policy Optimization): *El Deportista Cauteloso*
*(Actor-Crítico | On-policy | Acciones Continuas y Discretas)*

*   **La Intuición:** PPO aprende directamente a tomar decisiones mediante los consejos de un experto. Tiene a un **Actor** (el deportista que juega) y un **Crítico** (el entrenador que le indica si lo hizo mejor o peor de lo habitual).
*   **¿Cómo aprende de verdad?:** Calcula la **"Ventaja" (Advantage)**. El crítico dice si una acción fue superadora del promedio esperado. El sistema ajusta al actor para que repita más las acciones con ventaja positiva.
*   **Clave de Estabilidad (El Clipping):** PPO prohíbe que el Actor altere los pesos de su red neuronal más allá de una cantidad prefijada de un solo golpe (ej. límite `epsilon=0.2`). Esto impide el "olvido catastrófico" (catastrophic forgetting) tras una racha de buena o mala suerte en el simulador. Aprende a pasos sumamente controlados.

### TD3 (Twin Delayed DDPG): *El Evaluador Extremadamente Pesimista*
*(Actor-Crítico | Off-policy | Acciones Continuas)*

*   **La Intuición:** Soluciona el problema del falso optimismo. En otros algoritmos previos (como DDPG), la red a veces se "engañaba" al sobreestimar acciones erradas y se enfocaba en puntajes irreales. TD3 diseña un modelo para curar esto.
*   **¿Cómo aprende de verdad?:** Utiliza **Críticos Gemelos (Twin)**. El agente usa dos redes críticas y, al recibir la evaluación de ambas, se queda exclusivamente con **la predicción más baja**. Al penalizar despiadadamente las sobreestimaciones, el Actor deja de engañarse.
*   **Clave de Estabilidad (Retraso y Ruido):** Se llama "Delayed" porque actualiza al Actor solo después de entrenar múltiples veces al Crítico, asegurándose de que primero el "juez" tenga buena noción antes de decirle al "jugador" qué hacer. Además, empapa los cálculos de las predicciones futuras en un pequeño ruido matemático liso para suavizar la estimación global.

### SAC (Soft Actor-Critic): *El Explorador Creativo*
*(Actor-Crítico basado en Entropía máxima | Off-policy | Acciones Continuas)*

*   **La Intuición:** A diferencia de los algoritmos que buscan *EL* encadenamiento perfecto y único de movimientos para ganar, SAC se auto-premia por **maximizar la Recompensa manteniéndose a la par lo más creativo y aleatorio posible (Máxima Entropía)**.
*   **¿Cómo aprende de verdad?:** Al clásico objetivo de RL, SAC le suma directamente un bono matemático por estocasticidad. Su Actor escupe siempre una curva de distribución normal de probabilidades de donde elige aleatoriamente para moverse sobre un abanico funcional.
*   **Clave de Estabilidad:** Ajusta dinámicamente cuánta "Entropía" necesita usar mediante un agresivo parámetro de Temperatura ($\alpha$). Si el ambiente está en una sección extremadamente difícil y de alto riesgo, se vuelve clínico y preciso (baja la temperatura); si el robot no corre peligro de chocar, aumenta agresivamente la entropía para forzar a que su curiosidad explore infinitas soluciones alternativas válidas. Es genial para escapar de minimos locales.

---

## 2. Abriendo el Motor del Entorno: El caso del Péndulo (Pendulum-v1)

En este entorno base, el objetivo físico es aplicar fuerza angular controlada desde un motor central en la base para lograr levantar un tubo péndulo desafiando la gravedad y mantenerlo en equilibrio exacto en posición superior vertical.

### ¿Qué datos originarios (Estados) fluyen por la red?
En cada ciclo del micro-simulador, el script del entorno (`env.step()`) suministra exactamente un arreglo de 3 observaciones numéricas, todas limitadas:
1.  `[coseno(theta)]`: Posición cartográfica del péndulo por el eje horizontal X (valores de -1.0 a 1.0).
2.  `[seno(theta)]`: Posición vertical en Y del mismo (-1.0 a 1.0).
3.  `[velocidad angular]`: Fricción rotacional actual midiendo qué tan rápido está cayendo el tubo (-8.0 a 8.0).
*(Ej.: un tensor arrojando `[0.98, 0.17, -0.5]` indicaría que el tubo está a nada de ponerse de pie pero derrocándose liviana y velozmente a la derecha)*

### ¿Cómo procesa y muta esa señal hacia a la Acción?
Esos 3 variables penetran en la capa de entrada del `Actor`. La red opera con sus "Ponderaciones matriciales" (`Weights `) en las conexiones intermedias (Capas `Dense`). Al culminar y comprimirse, la salida suelta **1 solo nodo activo**, aplicándole arriba una compresión Tanh para limitarlo a la física de par máximo del motor:
*   `Acción/Torque Aplicado`: Un número real forzado al intervalo numérico `[-2.0, 2.0]`. Esto equivale directo al acelerador o el esfuerzo del engranaje interior impulsando a la izquierda (`-2`) o la derecha (`2`).

### El Castigo Físico y Matemático (Reward)
Una vez enviado ese par (`env.step(torque)`), el Motor inercial evalúa la fuerza empleada más el error originario sobre los ángulos contraídos frente al suelo en tiempo de respuesta. 
La recompensa finaliza con: `Recompensa = -(Ángulo_de_error estricto al cuadrado) - 0.1(Velocidad rotante al cuadrado) - 0.001(Fuerza empleada en el torque al cuadrado)`. 
Por definición rigurosa de Gym, la recompensa idílica inalcanzable supone el estado perfecto pasivo otorgando `0`. Generalmente a lo largo del proceso del episodio recogen decenas de recompensas crueles negativas desde el `-11` y la lucha de la arquitectura es intentar no despegar de rozar el `-0.01`.

---

## 3. La Fuerte División Conceptual: Arquitecturas On-Policy vs. Off-Policy

Esta distinción delimita arquitectónicamente a la IA de cómo y dónde obtiene y bebe experiencias informáticas de su ecosistema para procesar el descenso sobre el gradiente de error.

### On-Policy (El caso PPO, A2C) - *El Jugador Reflexivo Atorado al Presente*
El motor neuronal solo es capaz de moldearse nutriéndose exclusívamente en escenarios producidos bajo su red de configuración base al tiempo justo en el que existieron (Hace microsegundos).
**Explicación en uso de metáfora:** Un deportista instruido se graba entrenando por la mañana. Utiliza aquel video nocturnamente para detectar lo que falló e intentar mutar su físico general para erradicar las discrepancias. Luego **desecha las viejas cintas grabadas**. Comprende que las cintas pasadas retratan "vulnerabilidades viejas y decisiones que ya superó". El video actual queda inservible debido a haber mutado en conocimiento.
*   **Ventaja:** Notoriamente estable, casi incapaz de sufrir desbordes matemáticos, se traza gradualmente un trayecto escalable y controlable a prueba de errores lógicos. 
*   **Desventaja:** Resulta abismalmente ineficiente e impráctico consumiendo sus bases probatorias estadísticas frente a datos masivos (*Sample Inefficient*). Quema iteraciones para generar iteraciones.

### Off-Policy (El caso DQN, SAC, TD3) - *La Enciclopedia Reutilizada del Analista Ciego*
Esta ingeniería alberga rigurosamente todos los recorridos históricos desde la existencia primitiva hasta las épocas cumbre, en un baúl en disco enorme dominado como `Replay Buffer` masivo (`1M iteraciones`).
**Explicación en uso de metáfora:** Opuesto al On-policy, su equivalente físico prefiere coleccionar de memoria *todos*, los aciertos y desórdenes generados en sus vidas primerizas (O en su defecto grabaciones que pertenecen a robots no enlazados, políticas basura puramente generadoras base y experimentos aleatorios sin sentido). El algoritmo extrae maravillosamente relaciones de "Reglas Totales de Interacción Universal" frente a entornos concretos y lo interioriza sin importar cuándo, cómo ni bajo qué forma intelectual anterior extrajeron esa misma observación. 
*   **Ventaja:** Resulta soberbiamente óptimo, pragmático y exprimble en datos crudos a diferencia de la política local (*Sample Efficient*). Exige interacciones irrisorias de tiempo con los entornos y despliega lo que más precisan para su vida útil: Reutilizar millones de recuerdos viejos reciclándolos sobre las iteraciones continuadas para reducir tiempo de cálculo ambiental. Dominan de forma masiva robótica real física para prevenir rupturas materiales de simuladores lentificados.
*   **Desventaja:** Intrinsecamente propenso al caos o sobreoptimizaciones colaterales. Su flexibilidad ocasiona gradientes ciegos en la matemática abstracta del aprendizaje, necesitando que obligadamente utilices redes dobles estabilizadoras retrasadas o parches como en TD3.
