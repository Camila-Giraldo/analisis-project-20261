from src.controllers.manager import Manager
from src.controllers.strategies.kgeomip import KGeoMIP
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.geometric_generalized import GeometricSIAGeneralizada
# Optional import: this project often runs only geometric strategy.
try:
    from src.controllers.strategies.phi import Phi
except Exception:
    Phi = None
import multiprocessing
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path


GEOMIP_ROOT = Path(__file__).resolve().parents[1]
METHOD2_ROOT = GEOMIP_ROOT

def convertir_a_binario(texto, n_bits=25):
    posiciones = "ABCDEFGHIJKLMNOPQRSTUVWXY"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto:
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)

def ejecutar_con_tiempo(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm_path, k=2):
    try:
        # Carga con memmap: el SO gestiona qué páginas entran en RAM.
        # Para N=25 evita los ~6,7 GB en RAM del proceso hijo.
        tpm_path = Path(tpm_path)
        if tpm_path.suffix == '.npy':
            tpm = np.load(str(tpm_path), mmap_mode='r')
        else:
            tpm = np.genfromtxt(str(tpm_path), delimiter=',', dtype=np.float32)
        analizador_fi = KGeoMIP(config_sistema, k=k)
        sia_dos = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        resultado_queue.put({
            "particion": sia_dos.particion,
            "perdida": float(sia_dos.perdida),
            "tiempo": float(sia_dos.tiempo_ejecucion),
        })

    except Exception as e:
        resultado_queue.put({
            "particion": None,
            "perdida": None,
            "tiempo": None,
        })

def ejecutar_con_tiempo_generalizado(config_sistema, k, resultado_queue):
    try:
        res = GeometricSIAGeneralizada(config_sistema, k=k).aplicar_estrategia()
        resultado_queue.put({
            "particion":  str([sorted(s) for s in res['particion']]),
            "perdida":    float(res['phi']),
            "tiempo":     float(res['tiempo']),
            "k_natural":  res['k_natural'],
            "k_encontrado": res['k_encontrado'],
        })
    except Exception as e:
        print(f"[GeometricSIAGeneralizada] Error: {e}")
        resultado_queue.put({
            "particion": None, "perdida": None,
            "tiempo": None, "k_natural": None, "k_encontrado": None,
        })


def ejecutar_desde_excel_generalizado(
    ruta_excel: Path,
    ruta_salida: Path,
    k: int = 2,
    inicio: int = 0,
    cantidad: int = 50,
    estado_inicio: str | None = None,
):
    """
    Ejecuta GeometricSIAGeneralizada sobre cada red del Excel.

    A diferencia del flujo original, esta función no necesita las columnas
    alcance/mecanismo porque la estrategia generalizada opera sobre el sistema
    completo. El Excel solo determina qué red (estado_inicio) se analiza;
    el parámetro k define el número de particiones deseado.

    Columnas de salida:
        Iteración, Estado inicial, k, Partición, Phi, k_natural,
        k_encontrado, Tiempo de ejecución (s)
    """
    df = pd.read_excel(ruta_excel, sheet_name=2, usecols="B", skiprows=2, names=["Subsistema"])
    filas = df["Subsistema"].dropna().tolist()
    filas = filas[inicio:inicio + cantidad]
    resultados = []

    estado_inicio = estado_inicio or inferir_estado_inicial()

    for i, _ in enumerate(filas, start=inicio + 1):
        print(f"Iteración {i} - estado_inicio={estado_inicio}, k={k}")
        config_sistema = Manager(estado_inicial=estado_inicio)

        resultado_queue = multiprocessing.Queue()
        proceso = multiprocessing.Process(
            target=ejecutar_con_tiempo_generalizado,
            args=(config_sistema, k, resultado_queue),
        )
        proceso.start()
        proceso.join(timeout=3600)

        if proceso.is_alive():
            print(f"Iteración {i} - Tiempo límite alcanzado, terminando proceso...")
            proceso.terminate()
            proceso.join()
            resultado = {"particion": None, "perdida": None, "tiempo": None,
                         "k_natural": None, "k_encontrado": None}
        else:
            resultado = (
                resultado_queue.get()
                if not resultado_queue.empty()
                else {"particion": None, "perdida": None, "tiempo": None,
                      "k_natural": None, "k_encontrado": None}
            )

        resultados.append({
            "Iteración":             i,
            "Estado inicial":        estado_inicio,
            "k":                     k,
            "Partición":             resultado["particion"],
            "Phi":                   resultado["perdida"],
            "k_natural":             resultado["k_natural"],
            "k_encontrado":          resultado["k_encontrado"],
            "Tiempo de ejecución (s)": resultado["tiempo"],
        })

    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en {ruta_salida}")


def _preparar_tpm_binaria(csv_path: Path) -> Path:
    """
    Convierte la TPM de CSV a numpy binario float32 (.npy) si todavía no existe.

    Escribe el archivo en streaming (chunks de 100 K filas) para evitar
    cargar el CSV completo en RAM durante la conversión. Para N=25 el CSV
    pesa ~6,7 GB en float64; el .npy resultante pesa ~3,3 GB en float32.

    En ejecuciones posteriores detecta el .npy existente y lo devuelve
    directamente sin reconvertir.

    Carga (en subprocess): np.load(path, mmap_mode='r') → memmap.
    El SO gestiona qué páginas entran en RAM: con el truncamiento adaptativo
    solo se acceden ~1,8 M de los 33 M estados de N=25.
    """
    npy_path = csv_path.with_suffix('.npy')
    if npy_path.exists():
        return npy_path

    print(f"Convirtiendo {csv_path.name} → float32 binario (operación única, puede tardar varios minutos)...")

    # Determinar dimensiones sin cargar datos
    with open(csv_path, 'r') as f:
        n_cols = len(f.readline().split(','))
    with open(csv_path, 'r') as f:
        n_rows = sum(1 for _ in f)

    # Escribir .npy en streaming usando la API interna de numpy
    import numpy.lib.format as nfmt
    header = {
        'shape': (n_rows, n_cols),
        'fortran_order': False,
        'descr': np.dtype('float32').str,
    }
    procesadas = 0
    with open(npy_path, 'wb') as f_out:
        nfmt.write_array_header_1_0(f_out, header)
        for chunk in pd.read_csv(str(csv_path), header=None, chunksize=100_000, dtype=np.float32):
            chunk.values.tofile(f_out)
            procesadas += len(chunk)
            print(f"  {procesadas:,}/{n_rows:,} filas convertidas...", end='\r')

    size_mb = npy_path.stat().st_size / 1024 / 1024
    print(f"\nGuardado {npy_path.name} ({size_mb:.0f} MB). Próximas ejecuciones usarán memmap.")
    return npy_path


def resolver_tpm_path(estado_inicio: str) -> Path:
    """Find TPM file in common project locations based on state size."""
    sample_name = f"N{len(estado_inicio)}A.csv"
    candidates = (
        METHOD2_ROOT / "src" / ".samples" / sample_name,
        METHOD2_ROOT / ".samples" / sample_name,
        GEOMIP_ROOT / "data" / "samples" / sample_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró la TPM '{sample_name}'. Busqué en: {', '.join(str(c) for c in candidates)}"
    )


def inferir_estado_inicial() -> str:
    """Infer an initial state from available datasets (prefers largest NxA.csv)."""
    sample_dirs = (
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT / "data" / "samples",
    )
    pattern = re.compile(r"N(\d+)[A-Z]\.csv$")
    available_sizes = []

    for sample_dir in sample_dirs:
        if not sample_dir.exists():
            continue
        for sample_file in sample_dir.glob("N*.csv"):
            match = pattern.match(sample_file.name)
            if match:
                available_sizes.append(int(match.group(1)))

    if not available_sizes:
        raise FileNotFoundError("No hay archivos de muestras TPM disponibles en data/samples ni .samples.")

    n_bits = max(available_sizes)
    return "1" + ("0" * (n_bits - 1))


def ejecutar_desde_excel(
    ruta_excel: Path,
    ruta_salida: Path,
    inicio=0,
    cantidad=50,
    estado_inicio: str | None = None,
    condiciones: str | None = None,
    k: int = 2,
):
    df_header = pd.read_excel(ruta_excel, sheet_name=1, header=None, usecols="E", nrows=2, dtype=str)
    df = pd.read_excel(ruta_excel, sheet_name=1, usecols="B", skiprows=2, names=["Subsistema"]) #! here
    filas = df["Subsistema"].dropna().tolist()
    filas = filas[inicio:inicio + cantidad]
    resultados = []

    raw = df_header.iloc[1, 0]
    if isinstance(raw, str) and raw.strip():
        s = raw.strip()
        # Si pandas leyó el float como string científico o con dígitos de ruido,
        # reconstruir usando solo los dígitos binarios válidos (0 y 1).
        if all(c in '01' for c in s):
            estado_inicio_excel = s
        else:
            # Celda guardada como número: reconstruir desde float
            import math
            n_bits = round(math.log10(float(s))) if float(s) > 0 else 0
            estado_inicio_excel = "1" + "0" * n_bits
    else:
        estado_inicio_excel = None
    print(f"Estado inicial leído desde Excel (E2): {estado_inicio_excel}")
    estado_inicio = estado_inicio or estado_inicio_excel or inferir_estado_inicial()
    condiciones = condiciones or ("1" * len(estado_inicio))
    # Convertir CSV a .npy float32 si no existe (operación única).
    # El proceso principal nunca carga el contenido de la TPM en RAM;
    # solo pasa la ruta al subprocess, que la abre con memmap.
    csv_path = resolver_tpm_path(estado_inicio)
    tpm_path = _preparar_tpm_binaria(csv_path)

    for i, fila in enumerate(filas, start=inicio + 1):
        partes = fila.split("|")
        if len(partes) != 2:
            continue

        alcance = convertir_a_binario(partes[0][:len(partes[0]) - 3], n_bits=len(estado_inicio))
        mecanismo = convertir_a_binario(partes[1][:len(partes[1]) - 1], n_bits=len(estado_inicio))
        print(f"Iteración {i} - Alcance: {alcance}, Mecanismo: {mecanismo}")

        config_sistema = Manager(estado_inicial=estado_inicio)

        resultado_queue = multiprocessing.Queue()
        proceso = multiprocessing.Process(target=ejecutar_con_tiempo, args=(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm_path, k))
        
        proceso.start()
        proceso.join(timeout=3600)  

        if proceso.is_alive():
            print(f"Iteración {i} - Tiempo límite alcanzado, terminando proceso...")
            proceso.terminate()
            proceso.join()
            resultado = {"perdida": None, "tiempo": None, "particion": None}
        else:
            resultado = (
                resultado_queue.get()
                if not resultado_queue.empty()
                else {"perdida": None, "tiempo": None, "particion": None}
            )

        resultados.append({
            "Iteración": i,
            "k": k,
            "Alcance": alcance,
            "Mecanismo": mecanismo,
            "Partición": resultado["particion"],
            "Pérdida": resultado["perdida"],
            "Tiempo de ejecución (s)": resultado["tiempo"],
        })
    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en {ruta_salida}")

def iniciar():
    ruta_entrada = Path(
        os.getenv(
            "GEOMIP_INPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "Pruebas_iniciales_Metodo2.xlsx"),
        )
    )
    ruta_salida = Path(
        os.getenv(
            "GEOMIP_OUTPUT_XLSX",
            str(GEOMIP_ROOT / "results/k2" / "resultados_Geometric_10A2.xlsx"),
        )
    )

    ejecutar_desde_excel(
        ruta_entrada,
        ruta_salida,
        k=2,
    ) 