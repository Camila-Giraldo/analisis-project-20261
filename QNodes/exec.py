import argparse

from src.models.base.application import aplicacion
from src.main import iniciar, iniciar_kqnodes


def main():
    """Inicialización del aplicativo.

    Uso:
        python exec.py                            # QNodes bipartición (por defecto)
        python exec.py --estrategia kqnodes --k 3
        python exec.py --estrategia kqnodes --k 4 --metodo recocido
    """
    parser = argparse.ArgumentParser(
        description="Análisis de Partición de Mínima Información (MIP / k-MIP)."
    )
    parser.add_argument(
        "--estrategia",
        choices=["qnodes", "kqnodes"],
        default="qnodes",
        help="Estrategia a ejecutar (default: qnodes).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Número de bloques para KQNodes (default: 3).",
    )
    parser.add_argument(
        "--metodo",
        choices=["auto", "exacto", "voraz", "recocido"],
        default="auto",
        help="Método de búsqueda para KQNodes (default: auto).",
    )
    parser.add_argument(
        "--viz",
        default="",
        metavar="ARCHIVO.png",
        help="Guardar visualización del hipercubo en la ruta indicada (solo KQNodes).",
    )
    args = parser.parse_args()

    aplicacion.activar_profiling()
    aplicacion.set_pagina_red_muestra("A")

    if args.estrategia == "kqnodes":
        iniciar_kqnodes(k=args.k, metodo=args.metodo, guardar_viz=args.viz)
    else:
        iniciar()


if __name__ == "__main__":
    main()
