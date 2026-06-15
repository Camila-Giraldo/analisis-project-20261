from src.models.base.application import aplicacion
from src.main import iniciar

from src.controllers.manager import Manager
from pathlib import Path

def main():
    """Inicializar el aplicativo."""

    aplicacion.profiler_habilitado = False
    # aplicacion.pagina_sample_network = "B"

    #Generar N12A.csv si no existe
    # n12_path = Path("data/samples/N12A.csv")
    # if not n12_path.exists():
    #     m = Manager(estado_inicial="100000000000")
    #     m.generar_red(dimensiones=12, datos_discretos=True)
        

if __name__ == "__main__":
    iniciar()
    main()
