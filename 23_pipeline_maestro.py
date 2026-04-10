import os
import sys
import subprocess
from datetime import datetime

def main():
    print("\n" + "█"*70)
    print(" 🏆 INICIANDO PIPELINE MAESTRO DEL GRAN TORNEO: (DQN vs TD3 vs SAC vs PPO) ")
    print("█"*70)
    
    # Marcador de tiempo inamovible para este torneo especifico
    run_name = f"Torneo_Final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n[Fase 1/2] 🟢 ARRANCANDO COMPILADOR DE ENTRENAMIENTO (3,000 Episodios c/u)")
    print(f" > Jurisdicción operativa: Torneo_Maestro/{run_name}")
    print(" > Manten esta consola viva. Va a exprimir los transistores del PC un buen rato...\n")
    print("-" * 70)
    
    # sys.executable asegura que se arranque usando el mismo entorno virtual de Python activo (.venv)
    cmd_train = [sys.executable, "21_train_all_methods.py", run_name]
    try:
        subprocess.run(cmd_train, check=True)
    except subprocess.CalledProcessError:
        print("\n[❌ CRASHEO DEL SISTEMA] Falló el entrenamiento continuo. Misión Abortada.")
        sys.exit(1)
        
    print("\n" + "-" * 70)
    print(f"\n[Fase 2/2] 🎬 INVOCANDO EL TRIBUNAL CLÍNICO (Gráficas Severas y Generador .MP4)")
    print(" > Examinando mentes estáticas y produciendo el film...")
    print("-" * 70)
    
    cmd_eval = [sys.executable, "22_evaluate_all_methods.py", run_name]
    try:
        subprocess.run(cmd_eval, check=True)
    except subprocess.CalledProcessError:
        print("\n[❌ CRASHEO DEL SISTEMA] Error al procesar los vectores físicos de evaluación.")
        sys.exit(1)
        
    print("\n\n" + "█"*70)
    print(" 🎉 PROYECTO MAESTRO COMPLETADO MAGISTRALMENTE AL 100% ")
    print("█"*70)
    print(f" Tu tesis, modelos (.pth), logs físicos (Tensorboard), metraje audiovisual (.mp4)")
    print(f" y los blueprints gráficos (.png) radican juntos en su propia galaxia de carpetas:")
    print(f"\n  Directorio Definitivo 👉 Torneo_Maestro/{run_name}\n")
    print("  ¡Enhorabuena Científico! - Laboratorios Antigravity. ")

if __name__ == "__main__":
    main()
