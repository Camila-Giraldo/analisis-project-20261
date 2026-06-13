from src.models.base.application import aplicacion
from src.main import iniciar

from src.controllers.manager import Manager
from pathlib import Path

def main():
    """Inicializar el aplicativo."""

    aplicacion.profiler_habilitado = True
    # aplicacion.pagina_sample_network = "B"

    # Generar N15A.csv si no existe
    # n15_path = Path("data/samples/N15A.csv")
    # if not n15_path.exists():
    #     m = Manager(estado_inicial="100000000000000")
    #     m.generar_red(dimensiones=15, datos_discretos=True)
        

if __name__ == "__main__":
    iniciar()
    main()
