from colorama import init, Fore, Style
import pyttsx3
from pyttsx3.engine import Engine
from pyttsx3.voice import Voice
import numpy as np
from threading import Thread
from typing import Optional

from src.constants.models import PYPHI_LABEL
from src.constants.base import FLOAT_ZERO, INT_ZERO, WHITESPACE
from src.models.base.application import aplicacion

# Iniciar colorama
init()


class Solution:
    """
    Esta clase maneja la representación, visualización y anunciación por voz de las soluciones
    encontradas durante el análisis de Integrated Information Theory (IIT). Proporciona
    funcionalidades para mostrar distribuciones de probabilidad, particiones del sistema
    y el valor φ (phi|small phi) asociado a la pérdida de información.

    Args:
        estrategia (str):
            La estrategia utilizada para la resolución del problema.

        perdida (float):
            El valor φ que representa la pérdida de información en el sistema.
            Este valor cuantifica la diferencia entre la distribución del subsistema
            y la distribución de la partición.

        distribucion_subsistema (np.ndarray):
            Array que representa la distribución de probabilidad del subsistema completo.
            Contiene las probabilidades de cada estado posible en el espacio del subsistema.

        distribucion_particion (np.ndarray):
            Array que representa la distribución de probabilidad de la partición.
            Contiene las probabilidades de cada estado en el espacio de la partición
            que minimiza la información integrada.

        particion (str):
            Representación en formato string de la mejor partición encontrada.
            Utiliza notación matemática para mostrar la estructura de la partición.

        hablar (bool, opcional):
            Si es True, anuncia la solución encontrada usando síntesis de voz.
            Por defecto es True.

        voz (Optional[str], opcional):
            Identificador específico de la voz a utilizar para la síntesis.
            Si no se especifica, se busca automáticamente una voz en español.

    Attributes:
        perdida (float):
            El valor φ de la solución.

        distribucion_subsistema (np.ndarray):
            La distribución de probabilidad del subsistema.

        distribucion_particion (np.ndarray):
            La distribución de probabilidad de la partición.

        particion (str):
            La representación de la mejor partición.
    """

    def __init__(
        self,
        estrategia: str,
        perdida: float,
        distribucion_subsistema: np.ndarray,
        distribucion_particion: np.ndarray,
        particion: str,
        tiempo_total: float = FLOAT_ZERO,
        quiere_hablar: bool = False,
        voz: Optional[str] = None,
    ) -> None:
        """
        Inicializa una nueva instancia de Solution.
        Consultar la documentación de la clase para detalles de los parámetros.

        Args:
            estrategia: El nombre de la estrategia utilizada para la resolución del problema.
            perdida: El valor φ (small-phi) que representa la pérdida de información en la "bipartición" (comparado con el subsistema).
            distribucion_subsistema: Array que representa la distribución de probabilidad del subsistema.
            distribucion_particion: Array que representa la distribución de probabilidad de la partición hecha sobre el subsistema.
            particion: La representación en formato string (Literales alfanuméricos en numerador y denominador) de la mejor partición encontrada.
            tiempo_total: El tiempo total de ejecución únicamente del algoritmo (sin preparación del subsistema).
            quiere_hablar: Si es True, anuncia la solución encontrada usando síntesis de voz.
            voz: Un identificador específico de la voz a utilizar para la síntesis (Al no especificarse debería buscar automáticamente una voz en español, ajustable vía OS).
        """
        self.estrategia = estrategia
        self.perdida = perdida
        self.distribucion_subsistema = distribucion_subsistema
        self.distribucion_particion = distribucion_particion
        self.particion = particion
        self.tiempo_ejecucion = tiempo_total
        self.id_voz = voz
        self.hablar = quiere_hablar

    def __obtener_voz_espanol(self, motor: Engine) -> Optional[str]:
        """
        Busca y obtiene un identificador de voz en español del sistema.

        Esta función implementa un sistema de prioridades para seleccionar
        la mejor voz disponible en español, priorizando voces específicas
        de diferentes regiones hispanohablantes.

        Args:
            motor:
                Instancia del motor de síntesis de voz pyttsx3.Engine.

        Returns:
            Optional[str]:
                El identificador de la voz seleccionada, o None si no se
                encuentra ninguna voz.
        """
        voces: list[Voice] = motor.getProperty("voices")

        prioridades = [
            ("sabina", "méxico"),
            ("helena", "españa"),
            ("spanish", None),
            ("español", None),
            ("es-", None),
        ]

        for nombre_buscado, region in prioridades:
            for voz in voces:
                nombre_voz = voz.name.lower()
                id_voz = voz.id.lower()

                if nombre_buscado in nombre_voz or nombre_buscado in id_voz:
                    if region is None or region in nombre_voz:
                        return voz.id

        return voces[INT_ZERO].id if voces else None

    def __anunciar_solucion(self) -> None:
        """
        Anuncia la solución encontrada usando síntesis de voz en español.

        Esta función configura y utiliza el motor de síntesis de voz para anunciar de forma audible que se ha encontrado una solución, incluyendo el valor φ calculado.

        La función se ejecuta en un hilo separado para no bloquear la ejecución principal del programa mientras se realiza la síntesis de voz.
        """
        try:
            motor = pyttsx3.init()

            id_voz = self.id_voz or self.__obtener_voz_espanol(motor)
            if id_voz:
                motor.setProperty("voice", id_voz)

            motor.setProperty("rate", 150)
            motor.setProperty("volume", 0.9)

            mensaje = f"Solución encontrada con {self.estrategia}." + (
                f"El valor de fi es de {self.perdida:.2f}"
                if self.perdida > FLOAT_ZERO
                else "No hubo pérdida."
            )
            motor.say(mensaje)
            motor.runAndWait()
        except Exception as e:
            print(f"Error al inicializar el motor de voz: {e}")

    def __str__(self) -> str:
        """
        Genera una representación en string formateada y coloreadita de la solución.

        Returns:
            str:
                Representación visual de la solución que incluye:
                - Valor φ en verdecito
                - Distribuciones del subsistema y partición
                - Mejor partición encontrada en magenta
                - Elementos decorativos en cyan

        Notes:
            Utiliza la biblioteca colorama para el formato de colores, permitiedo una visualización clara por terminal.
        """
        espaciado = 64
        bilinea = "═" * espaciado
        trilinea = "≡" * espaciado

        def formatear_distribucion(
            distribucion: np.ndarray,
            evitar_desbordamiento=True,
        ):
            rango = distribucion.size
            mensaje_desborde = ""
            if evitar_desbordamiento:
                LIMITE = espaciado
                excedente = rango - LIMITE
                if excedente > FLOAT_ZERO:
                    mensaje_desborde = f" {excedente} valores más.."
                    rango = LIMITE

            datos = WHITESPACE.join(
                f"{Fore.WHITE}{distribucion[idx]:.4f}"
                if distribucion[idx] > FLOAT_ZERO
                else f"{Fore.LIGHTBLACK_EX}0.    "
                for idx in range(rango)
            )
            return f"[ {datos}{mensaje_desborde} {Fore.WHITE}]"

        if self.hablar:
            voz = Thread(target=self.__anunciar_solucion, daemon=True)
            voz.start()
            voz.join(timeout=10)

        es_pyphi = self.estrategia == PYPHI_LABEL
        tipo_distribucion = "tensorial" if es_pyphi else "marginal"

        tiempo_hrs, tiempo_min, tiempo_seg = (
            f"{self.tiempo_ejecucion / 3600:.2f}",
            f"{self.tiempo_ejecucion / 60:.1f}",
            f"{self.tiempo_ejecucion:.4f}",
        )
        return f"""{Fore.CYAN}{bilinea}

{Fore.RED}{self.estrategia} fue la estrategia de solucion.

{Fore.BLUE}Distancia métrica utilizada:
{Fore.WHITE}{aplicacion.distancia_metrica}
{Fore.BLUE}Notación utilizada en indexación:
{Fore.WHITE}{aplicacion.notacion_indexado}

{Fore.YELLOW}Distribucion {tipo_distribucion} del Subsistema:
{Style.RESET_ALL}{formatear_distribucion(self.distribucion_subsistema)}
{Fore.YELLOW}Distribucion {tipo_distribucion} de la Partición:
{Style.RESET_ALL}{formatear_distribucion(self.distribucion_particion)}

{Fore.YELLOW}Mejor Partición:
{Fore.MAGENTA}{self.particion}
{Fore.GREEN}Perdida mínima ( φ ) = {self.perdida:.4f}

{Fore.BLUE}Tiempos de ejecución:
{Fore.WHITE}Horas: {tiempo_hrs} = Minutos: {tiempo_min} = Segundos: {tiempo_seg}

{Fore.CYAN}{trilinea}{Style.RESET_ALL}"""

    def __repr__(self) -> str:
        """
        Implementa la representación oficial de la clase Solution.

        Returns:
        -------
            str:
                La misma representación que __str__ para consistencia.
        """
        return self.__str__()
