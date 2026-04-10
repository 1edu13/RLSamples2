"""
Script Maestro de Entrenamiento: Ejecuta todos los algoritmos secuencialmente.

Usar este script te permite dejar el portátil entrenando durante horas sin 
tener que intervenir. Entrenará DQN, luego TD3, luego PPO y finalmente SAC.
Cada uno creará sus propias carpetas de fecha/hora en 'models', 'runs' y 'videos'.
"""

import subprocess
import time

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO ENTRENAMIENTO: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Se llama al proceso de Python con el script correspondiente usando el mismo ejecutable actual
        # Usamos check=True para que si un script falla, detenga toda la cadena y te avise
        import sys
        subprocess.run([sys.executable, script_name], check=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {script_name} completado con éxito en {elapsed_time/60:.2f} minutos.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR CRÍTICO en {script_name}. Ejecución detenida.")
        print(e)
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Entrenamiento de {script_name} cancelado por el usuario (Ctrl+C).")
        return False
        
    return True

if __name__ == "__main__":
    print("Iniciando Suite de Entrenamiento Automatizado (Highway-Env)...")
    
    # Lista de los scripts a ejecutar en orden
    scripts_to_run = [
        "05_highway_dqn.py",
        "06_highway_td3.py",
        "07_highway_ppo.py",
        "08_highway_sac.py"
    ]
    
    total_start_time = time.time()
    
    # Bucle que lanza todos los scripts
    for script in scripts_to_run:
        success = run_script(script)
        if not success:
            print("\nDeteniendo cadena de entrenamiento debido a cancelación manual o error.")
            break
            
    # Resumen Final
    total_time_hours = (time.time() - total_start_time) / 3600
    print(f"\n{'='*60}")
    print(f"🏁 MASTER SCRIPT FINALIZADO.")
    print(f"⏱️ Tiempo total de batería invertido: {total_time_hours:.2f} horas.")
    print(f"📂 Revisa las carpetas 'models', 'runs' y 'videos' para ver los resultados.")
    print(f"{'='*60}")
