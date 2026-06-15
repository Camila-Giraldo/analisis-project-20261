from dataclasses import dataclass
from pathlib import Path
import time
import os

import numpy as np

from src.models.base.application import aplicacion
from src.constants.base import (
    ABC_START,
    COLON_DELIM,
    CSV_EXTENSION,
    NPY_EXTENSION,
    PATH_SAMPLES,
    PATH_RESOLVER,
)


@dataclass
class Manager:
    """
    El gestor es el encargado de en función al tamaño del estado inicial y la página asociada traer el fichero de formato CSV con las TPM's almacenadas en `.samples/` para hacer una rápida depuración de los datos para la creación de sistemas.

    Args:
    ----
        - `dimensiones` (str): Dado se manejan sistemas binarios es un número base dos de tamaño asociado a la red que se quiera cargar.
        - `pagina` (str): En la ruta de samples se tiene un literal asociado al tamaño de las redes por si se necesita añadir varias de un mismo tamaño.
        ruta_base (Path): Ruta donde se encuentran las muestras de TPMs en representación estado-nodo-on (TPM estado-nodo simplificada).

    Returns:
    -------
        Manager: Así mismo se encarga de asociar el directorio donde se mostrarán análisis de las ejecuciones, donde sea el programador haga uso del módulo de logging y profilling.
    """

    estado_inicial: str
    ruta_base: Path = Path(PATH_SAMPLES)

    @property
    def pagina(self) -> str:
        return aplicacion.pagina_red_muestra

    @property
    def npy_filename(self) -> Path:
        """Ruta del binario `.npy` (formato preferido: float32 crudo, carga sin
        parseo y con pico de memoria mínimo)."""
        return (
            self.ruta_base / f"N{len(self.estado_inicial)}{self.pagina}.{NPY_EXTENSION}"
        )

    @property
    def csv_filename(self) -> Path:
        """Ruta del CSV heredado (`%.8f` en texto); solo para compatibilidad con
        las muestras antiguas ya generadas."""
        return (
            self.ruta_base / f"N{len(self.estado_inicial)}{self.pagina}.{CSV_EXTENSION}"
        )

    @property
    def tpm_filename(self) -> Path:
        """TPM a usar: prioriza el binario `.npy` y recae en el CSV heredado si es
        el único disponible."""
        return self.npy_filename if self.npy_filename.exists() else self.csv_filename

    @property
    def output_dir(self) -> Path:
        return Path(
            f"{PATH_RESOLVER}/N{len(self.estado_inicial)}{self.pagina}/{self.estado_inicial}"
        )

    def preparar_directorio_salida(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cargar_red(self) -> np.ndarray:
        """Carga la TPM del subsistema como ``np.ndarray`` float32.

        Estrategia de carga (de más a menos eficiente):
          1. Binario ``.npy`` si existe → ``np.load`` (sin parseo, pico de RAM
             igual al tamaño del array; viable para N grandes).
          2. CSV ``%.8f`` heredado si es lo único disponible → ``np.loadtxt``
             (compatibilidad con las muestras antiguas; pico de RAM alto).
          3. Si no hay ninguno, genera la red automáticamente (en ``.npy``) y la
             carga.

        Returns:
            np.ndarray: TPM en representación estado-nodo, dtype float32.
        """
        if self.npy_filename.exists():
            return np.load(self.npy_filename).astype(np.float32, copy=False)
        if self.csv_filename.exists():
            return np.loadtxt(
                self.csv_filename, delimiter=COLON_DELIM, dtype=np.float32
            )
        self.generar_red(
            len(self.estado_inicial),
            datos_deterministas=False,
            interactivo=False,
        )
        return np.load(self.npy_filename).astype(np.float32, copy=False)

    def generar_red(
        self,
        dimensiones: int,
        datos_deterministas: bool = False,
        interactivo: bool = True,
    ) -> str:
        """
        Se encarga de generar una red (TPM) en notación little endian para un sistema determinista o estocástico (esto en función a si contiene datos discretos o no respectivamente. Nunca confundir con un "Sistema continuo" puesto apela a otra definición totalmente diferente).
        La red generada almacenará en el "output_dir", un atributo dinámico en función a que si generaste una red de un tamaño X por primera vez, estará etiquetada como "A", si deseas generar otra red del mismo tamaño naturalmente contendrá los mismos datos puesto están determinados por la semilla numpy, de forma que la forma de obtener otra red diferente es actuando sobre el parámetro `datos_deterministas`, siendo estas dos redes distintas en su contenido.

        Args:
            dimensiones (int): Número de nodos/elementos/variables/canales que se desea maneje la red, obteniendo un Sistema que para cada estado en $(t)$ tendrá un canalen $(t+1)$.
            datos_deterministas (bool, optional): Selecciona si se quiere que la red generada sea estocástica, con el valor de probabilidad como siempre, un real positivo entre 0 y 1 inclusivo. Por defecto es False.
            interactivo (bool, optional): Si es True (por defecto), preguntará antes de sobrescribir archivos existentes o si el tamaño supera 1GB. Si es False, generará o sobrescribirá automáticamente sin preguntar.

        Notas:
            La red se persiste en binario ``.npy`` (float32 crudo), no en CSV de
            texto: el archivo ocupa exactamente ``2^n · n · 4`` bytes y se recarga
            con ``np.load`` sin parseo ni pico de memoria. Las muestras antiguas en
            CSV siguen siendo legibles por ``cargar_red`` (fallback).

        Raises:
            ValueError: Si las dimensiones son menores a 1.

        Returns:
            str: El nombre del archivo generado.
        """
        np.random.seed(aplicacion.semilla_numpy)

        if dimensiones < 1:
            raise ValueError("Las dimensiones deben ser positivas")

        # Calcular tamaño y tiempo estimado. El `.npy` guarda float32 crudo
        # (1 byte por celda en el caso determinista int8), de modo que el tamaño
        # real es `2^n · n · bytes_por_celda`.
        num_estados = 1 << dimensiones
        bytes_por_celda = 1 if datos_deterministas else 4
        total_size_gb = (num_estados * dimensiones * bytes_por_celda) / (1024**3)
        estimated_time = total_size_gb * 2

        print(f"Tamaño estimado: {total_size_gb:.6f} GB")
        print(f"Tiempo estimado: {estimated_time:.1f} segundos")

        if (
            interactivo
            and total_size_gb > 1
            and input("El sistema ocupará más de 1GB. ¿Continuar? (s/n): ").lower()
            != "s"
        ):
            return

        # Verificar archivos existentes y generar nuevo nombre
        base_path = Path(PATH_SAMPLES)
        base_path.mkdir(parents=True, exist_ok=True)

        suffix = ABC_START
        while (base_path / f"N{dimensiones}{suffix}.{NPY_EXTENSION}").exists():
            if not interactivo:
                break
            if (
                input(
                    f"Ya existe N{dimensiones}{suffix}.{NPY_EXTENSION}. ¿Generar nueva red? (s/n): "
                ).lower()
                != "s"
            ):
                return f"N{dimensiones}{suffix}.{NPY_EXTENSION}"
            suffix = chr(ord(suffix) + 1)

        filename = f"N{dimensiones}{suffix}.{NPY_EXTENSION}"
        filepath = base_path / filename

        # Generar estados
        print("Generando estados...")
        start_time = time.time()

        if datos_deterministas:
            states = np.random.randint(
                2, size=(num_estados, dimensiones), dtype=np.int8
            )
        else:
            # float32 directo: mitad de RAM que float64 y coincide con el dtype
            # con el que `cargar_red` entrega la TPM.
            states = np.random.random(size=(num_estados, dimensiones)).astype(
                np.float32
            )

        print(f"Generación completada en {time.time() - start_time:.2f} segundos")

        # Guardar archivo en binario (sin formateo de texto: rápido y compacto).
        print(f"Guardando en {filepath}...")
        start_time = time.time()
        np.save(filepath, states)

        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"Archivo guardado: {file_size_gb:.6f} GB")
        print(f"Tiempo de guardado: {time.time() - start_time:.2f} segundos")

        return filename
