import sys
import os
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import csv
import gc
import argparse
import pandas as pd
import time
import os
import logging
from controllers.manager import Manager
from strategies.q_nodes import QNodes

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

MAX_VERTICES_DEFAULT = 37


def _contar_unos(s: str) -> int:
    return sum(1 for c in s if c == '1')


def _parsear_rango_filas(raw: str) -> set[int]:
    filas = set()
    for parte in raw.split(','):
        parte = parte.strip()
        if '-' in parte:
            a, b = parte.split('-', 1)
            filas.update(range(int(a), int(b) + 1))
        else:
            filas.add(int(parte))
    return filas


def run_tests(file_in, file_out_csv, max_vertices=MAX_VERTICES_DEFAULT, solo_filas=None):
    basename = os.path.basename(file_in)
    bits_str = ''.join(c for c in basename if c.isdigit())
    bits = int(bits_str)
    ESTADO_INICIAL = "1" + "0" * (bits - 1)
    CONDICIONES = "1" * bits

    df = pd.read_excel(file_in, dtype=str)

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

    gestor_redes = Manager(ESTADO_INICIAL)
    mpt = gestor_redes.cargar_red()

    columnas_csv = ["Prueba", col_alcance, col_mecanismo, "Partición", "Pérdida", "Tiempo"]

    total = len(df)
    ejecutadas = 0
    saltadas = 0

    with open(file_out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columnas_csv)
        f.flush()

        for idx, row in df.iterrows():
            alcance = str(row[col_alcance])
            mecanismo = str(row[col_mecanismo])
            num_fila = idx + 1

            # Filtro --rows
            if solo_filas is not None and num_fila not in solo_filas:
                continue

            # Validación de longitudes
            esperado = len(ESTADO_INICIAL)
            if not all(len(x) == esperado for x in [ESTADO_INICIAL, CONDICIONES, alcance, mecanismo]):
                writer.writerow([num_fila, alcance, mecanismo, "ERROR_LONGITUD", "ERROR_LONGITUD", "ERROR_LONGITUD"])
                f.flush()
                logging.error(f"Fila {num_fila} ignorada por longitud incorrecta")
                continue

            # Filtro --max-vertices
            m = _contar_unos(alcance)
            n = _contar_unos(mecanismo)
            if (m + n) > max_vertices:
                writer.writerow([num_fila, alcance, mecanismo, "SALTADO", f"vértices={m+n}>{max_vertices}", ""])
                f.flush()
                saltadas += 1
                logging.warning(f"Fila {num_fila} saltada: m={m}, n={n}, vértices={m+n} > max_vertices={max_vertices}")
                continue

            # Ejecución
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

                ejecutadas += 1
                logging.info(f"Fila {num_fila} OK — pérdida={perdida} — {dt:.1f}s")

                del analizador
                gc.collect()
            except Exception as e:
                particion, perdida, tiempo = "ERROR", "ERROR", "ERROR"
                logging.error(f"Error fila {num_fila} (alcance={alcance}, mecanismo={mecanismo}): {str(e)}")

            writer.writerow([num_fila, alcance, mecanismo, particion, perdida, tiempo])
            f.flush()

    logging.info(f"Completado: {ejecutadas} ejecutadas, {saltadas} saltadas, {total} totales en {file_out_csv}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    carpeta_input = os.path.join(script_dir, "input")
    carpeta_output = os.path.join(script_dir, "output")

    parser = argparse.ArgumentParser(description="Ejecuta pruebas de redes QNodes desde archivos Excel")
    parser.add_argument("file", nargs="?", help="Archivo a procesar (ej: 25A o N25A.xlsx)")
    parser.add_argument("--max-vertices", type=int, default=MAX_VERTICES_DEFAULT,
                        help=f"Máximo de vértices (m+n) por prueba (default: {MAX_VERTICES_DEFAULT})")
    parser.add_argument("--rows", type=str, default=None,
                        help="Filas específicas a ejecutar, ej: '7,13-21,33-49'")

    args = parser.parse_args()

    if not os.path.exists(carpeta_input):
        logging.error(f"Directorio de entrada no encontrado: {carpeta_input}")
        return

    if not os.path.exists(carpeta_output):
        logging.warning(f"Creando directorio de salida: {carpeta_output}")
        os.makedirs(carpeta_output, exist_ok=True)

    archivos = os.listdir(carpeta_input)
    if args.file:
        arg = args.file
        if not arg.endswith(".xlsx"):
            arg += ".xlsx"
        if not arg.startswith("N"):
            arg = "N" + arg
        archivos = [f for f in archivos if f == arg]
        if not archivos:
            logging.error(f"No se encontró {arg} en input/")
            return

    solo_filas = _parsear_rango_filas(args.rows) if args.rows else None

    for file in archivos:
        if file.startswith("N") and file.endswith(".xlsx"):
            base = file[1:-5]
            file_in = os.path.join(carpeta_input, file)
            file_out_csv = os.path.join(carpeta_output, f"S{base}.csv")
            logging.info(f"Procesando {file_in} → {file_out_csv}  (max_vertices={args.max_vertices})")
            try:
                run_tests(file_in, file_out_csv, max_vertices=args.max_vertices, solo_filas=solo_filas)
            except Exception as e:
                logging.error(f"Error en {file}: {str(e)}")


if __name__ == "__main__":
    main()
