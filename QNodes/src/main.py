from src.controllers.manager import Manager

# 👇 Importación de estrategias 👇 #
from src.strategies.q_nodes import QNodes
from src.strategies.KQNodes import KQNodes


def iniciar():
    """Punto de entrada QNodes (bipartición)."""

    # N22 #
    estado_inicial = "1" + "0" * 21
    condiciones =    "1" * 22
    alcance =        "1111111111111111111111"
    mecanismo =      "1111111111111111111111"

    gestor_redes = Manager(estado_inicial)
    mpt = gestor_redes.cargar_red()

    analizador = QNodes(mpt)
    sia_cero = analizador.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_cero)


def iniciar_kqnodes(k: int = 3, metodo: str = "auto", guardar_viz: str = ""):
    """Punto de entrada KQNodes — demo de k-Partición de Mínima Información.

    Args:
        k (int): número de bloques de la k-partición (2 ≤ k ≤ |V|).
        metodo (str): "exacto", "voraz", "recocido" o "auto".
        guardar_viz (str): ruta PNG donde guardar la visualización del hipercubo;
            si está vacío no se genera la imagen.
    """
    estado    = "1" + "0" * 24   # N25
    cond      = "1" * 25
    alcance   = "1" * 25
    mecanismo = "1" * 25

    mpt = Manager(estado).cargar_red()
    analizador = KQNodes(mpt)
    sol = analizador.aplicar_estrategia(estado, cond, alcance, mecanismo, k, metodo=metodo)
    print(sol)

    if guardar_viz:
        from src.funcs.visualize import save_kparticion_hypercube
        n_bits = len(estado)
        save_kparticion_hypercube(
            analizador.ultima_kparticion,
            n_bits,
            guardar_viz,
            titulo=f"KQNodes k={k} sobre N{n_bits}",
        )
        print(f"Visualización guardada en: {guardar_viz}")
