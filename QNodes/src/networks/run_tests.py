import sys
import os
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import pandas as pd
import time
import os
import logging
from controllers.manager import Manager
from strategies.q_nodes import QNodes

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_tests(file_in, file_out_csv):
    # Datos fijos del experimento (modifícalos si en el futuro cambian)
    ESTADO_INICIAL = "1000000000"
    CONDICIONES = "1111111111"

    # Lee el archivo de entrada
    df = pd.read_excel(file_in)

    # Verifica nombres flexibles de columna para robustez
    posibles_nombres_alcance = [
        "Alcance o Purview (t+1)", "Alcance", "Purview (t+1)", "Purview", "alcance", "purview"
    ]
    posibles_nombres_mecanismo = [
        "Mecanismo(t)", "Mecanismo", "mecanismo"
    ]
    col_alcance = next((c for c in posibles_nombres_alcance if c in df.columns), None)
    col_mecanismo = next((c for c in posibles_nombres_mecanismo if c in df.columns), None)
    if not col_alcance or not col_mecanismo:
        raise ValueError("No se encuentran las columnas necesarias para alcance y mecanismo en el archivo de entrada.")

    resultados = []

    for idx, row in df.iterrows():
        alcance = str(row[col_alcance])
        mecanismo = str(row[col_mecanismo])

        # Validación de longitudes
        esperado = len(ESTADO_INICIAL)
        if not all(len(x) == esperado for x in [ESTADO_INICIAL, CONDICIONES, alcance, mecanismo]):
            particion, perdida, tiempo = "ERROR_LONGITUD_INCORRECTA", "ERROR_LONGITUD_INCORRECTA", "ERROR_LONGITUD_INCORRECTA"
            logging.error(f"Fila {idx+1} ignorada por longitud incorrecta: esperado {esperado} bits en estado, condiciones, alcance y mecanismo (alcance={alcance}, mecanismo={mecanismo})")
        else:
            try:
                gestor_redes = Manager(ESTADO_INICIAL)
                mpt = gestor_redes.cargar_red()
                analizador_bf = QNodes(mpt)

                t0 = time.time()
                solution = analizador_bf.aplicar_estrategia(
                    ESTADO_INICIAL,
                    CONDICIONES,
                    alcance,
                    mecanismo,
                )
                dt = time.time() - t0

                particion = getattr(solution, "particion", "ERROR")
                perdida = getattr(solution, "perdida", "ERROR")
                tiempo = getattr(solution, "tiempo_total", dt)
            except Exception as e:
                particion, perdida, tiempo = "ERROR", "ERROR", "ERROR"
                logging.error(f"Error procesando fila {idx+1} (alcance={alcance}, mecanismo={mecanismo}): {str(e)}")

        resultados.append([
            idx + 1,  # Prueba (consecutivo empezando en 1)
            alcance,
            mecanismo,
            particion,
            perdida,
            tiempo,
        ])

    result_df = pd.DataFrame(resultados, columns=[
        "Prueba", col_alcance, col_mecanismo, "Partición", "Pérdida", "Tiempo"
    ])
    result_df.to_csv(file_out_csv, index=False)


def main():
    # Descubre automáticamente los archivos N...xlsx en carpeta input
    script_dir = os.path.dirname(os.path.abspath(__file__))
    carpeta_input = os.path.join(script_dir, "input")
    carpeta_output = os.path.join(script_dir, "output")
    
    # Validar que los directorios existan
    if not os.path.exists(carpeta_input):
        logging.error(f"Directorio de entrada no encontrado: {carpeta_input}")
        return
    
    if not os.path.exists(carpeta_output):
        logging.warning(f"Creando directorio de salida: {carpeta_output}")
        os.makedirs(carpeta_output, exist_ok=True)
    
    for file in os.listdir(carpeta_input):
        if file.startswith("N") and file.endswith(".xlsx"):
            base = file[1:-5]  # de 'N10A.xlsx' saca '10A'
            file_in = os.path.join(carpeta_input, file)
            file_out = os.path.join(carpeta_output, f"S{base}.csv")
            print(f"Ejecutando pruebas para {file_in} → {file_out}")
            try:
                run_tests(file_in, file_out)
            except Exception as e:
                logging.error(f"Error en {file}: {str(e)}")

if __name__ == "__main__":
    main()
