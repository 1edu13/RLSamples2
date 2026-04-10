


"""
Script Maestro de Entrenamiento: Ejecuta todos los algoritmos secuencialmente en el entorno PENDULUM.
"""

import subprocess
import time

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO ENTRENAMIENTO PENDULUM: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
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
    print("Iniciando Suite de Entrenamiento Automatizado (Pendulum-v1)...")
    
    # Lista de los scripts a ejecutar en orden
    scripts_to_run = [
        "11_pendulum_dqn.py",
        "12_pendulum_td3.py",
        "13_pendulum_ppo.py",
        "14_pendulum_sac.py"
    ]
    
    total_start_time = time.time()
    
    for script in scripts_to_run:
        success = run_script(script)
        if not success:
            print("\nDeteniendo cadena de entrenamiento debido a cancelación manual o error.")
            break
            
    total_time_hours = (time.time() - total_start_time) / 3600
    print(f"\n{'='*60}")
    print(f"🏁 MASTER SCRIPT PENDULUM FINALIZADO.")
    print(f"⏱️ Tiempo total de batería invertido: {total_time_hours:.2f} horas.")
    print(f"📂 Ahora puedes ejecutar: python 15_evaluate_pendulum.py")
    print(f"{'='*60}")
