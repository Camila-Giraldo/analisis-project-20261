import sys
import os
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import csv
import pandas as pd
import time
import os
import logging
from controllers.manager import Manager
from strategies.q_nodes import QNodes

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run_tests(file_in, file_out_csv):
    # Extraer bits desde el nombre del archivo: N10A.xlsx → 10
    basename = os.path.basename(file_in)
    bits_str = ''.join(c for c in basename if c.isdigit())
    bits = int(bits_str)
    ESTADO_INICIAL = "1" + "0" * (bits - 1)
    CONDICIONES = "1" * bits

    # Lee el archivo de entrada (como string para preservar ceros a la izquierda)
    df = pd.read_excel(file_in, dtype=str)

    # Detecta columnas por palabra clave (sin importar espacios, mayúsculas, paréntesis)
    def _normalizar(nombre: str) -> str:
        return nombre.replace(" ", "").replace("_", "").replace("-", "").lower()

    col_alcance = next(
        (c for c in df.columns if "alcance" in _normalizar(c) or "purview" in _normalizar(c)),
        None,
    )
    col_mecanismo = next(
        (c for c in df.columns if "mecanismo" in _normalizar(c) or "mechanism" in _normalizar(c)),
        None,
    )
    if not col_alcance or not col_mecanismo:
        raise ValueError("No se encuentran las columnas necesarias para alcance y mecanismo en el archivo de entrada.")

    # Cargar red una sola vez (es la misma para todas las filas)
    gestor_redes = Manager(ESTADO_INICIAL)
    mpt = gestor_redes.cargar_red()

    columnas_csv = ["Prueba", col_alcance, col_mecanismo, "Partición", "Pérdida", "Tiempo"]

    with open(file_out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columnas_csv)

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
                    analizador = QNodes(mpt)
                    t0 = time.time()
                    solution = analizador.aplicar_estrategia(
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

            writer.writerow([idx + 1, alcance, mecanismo, particion, perdida, tiempo])


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

    # Filtrar por argumento CLI opcional
    archivos = os.listdir(carpeta_input)
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if not arg.endswith(".xlsx"):
            arg += ".xlsx"
        if not arg.startswith("N"):
            arg = "N" + arg
        archivos = [f for f in archivos if f == arg]
        if not archivos:
            logging.error(f"No se encontró {arg} en input/")
            return

    for file in archivos:
        if file.startswith("N") and file.endswith(".xlsx"):
            base = file[1:-5]  # de 'N10A.xlsx' saca '10A'
            file_in = os.path.join(carpeta_input, file)
            file_out_csv = os.path.join(carpeta_output, f"S{base}.csv")
            print(f"Ejecutando pruebas para {file_in} → {file_out_csv}")
            try:
                run_tests(file_in, file_out_csv)
            except Exception as e:
                logging.error(f"Error en {file}: {str(e)}")

if __name__ == "__main__":
    main()
