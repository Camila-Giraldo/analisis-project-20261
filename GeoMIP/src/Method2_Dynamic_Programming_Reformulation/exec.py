from src.models.base.application import aplicacion
from src.main import iniciar

from src.controllers.manager import Manager
from pathlib import Path

def main():
    """Inicializar el aplicativo."""

    aplicacion.profiler_habilitado = True
    # aplicacion.pagina_sample_network = "B"

    # Generar N25A.csv si no existe
    n25_path = Path("data/samples/N25A.csv")
    if not n25_path.exists():
        m = Manager(estado_inicial="1000000000000000000000000")
        m.generar_red(dimensiones=25, datos_discretos=True)
        

   # iniciar()


if __name__ == "__main__":
    main()
