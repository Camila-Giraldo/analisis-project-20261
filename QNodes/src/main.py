from src.controllers.manager import Manager

# 👇 Importación de estrategias 👇 #
from src.strategies.q_nodes import QNodes
from src.strategies.KQNodes import KQNodes


def iniciar():
    """Punto de entrada QNodes (bipartición)."""

    estado_inicial = "1000000000"
    condiciones =    "1111111111"
    alcance =        "1111111111"
    mecanismo =      "1111111111"

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
        guardar_viz (str): carpeta raíz donde guardar las visualizaciones.
            Para n_bits ≥ 4 se crea la subcarpeta N{n}_k{k} automáticamente.
            Si está vacío se usa "KQNodes_graphics" como raíz.
            Para n_bits ≤ 3 se trata como ruta directa al archivo PNG.
    """
    estado    = "1000000000"
    cond      = "1111111111"
    alcance   = "1111111111"
    mecanismo = "1111111111"

    n_bits = len(estado)
    mpt = Manager(estado).cargar_red()
    analizador = KQNodes(mpt)
    sol = analizador.aplicar_estrategia(estado, cond, alcance, mecanismo, k, metodo=metodo)
    print(sol)

    if n_bits <= 3:
        if guardar_viz:
            from src.funcs.visualize import save_kparticion_hypercube
            save_kparticion_hypercube(
                analizador.ultima_kparticion,
                n_bits,
                guardar_viz,
                titulo=f"KQNodes k={k} sobre N{n_bits}",
            )
            print(f"Visualización guardada en: {guardar_viz}")
    else:
        from src.funcs.visualize import save_kparticion_multivista

        # Valores por nodo: P(var=1 | presentes del bloque) para futuros,
        # estado inicial (0/1) para presentes.
        dist = analizador.marginal_distributions(analizador.ultima_kparticion)
        valores = {
            **{
                (1, int(cubo.indice)): f"{float(dist[i]):.3f}"
                for i, cubo in enumerate(analizador.sia_subsistema.ncubos)
            },
            **{
                (0, int(d)): str(int(analizador.sia_subsistema.estado_inicial[int(d)]))
                for d in analizador.sia_subsistema.dims_ncubos
            },
        }

        from pathlib import Path
        raiz = Path(guardar_viz) if guardar_viz else Path("KQNodes_graphics")
        carpeta = raiz / f"N{n_bits}_k{k}"
        rutas = save_kparticion_multivista(
            analizador.ultima_kparticion,
            n_bits,
            carpeta,
            titulo=f"KQNodes k={k} · N{n_bits}",
            valores=valores,
        )
        print(f"\nGráficos guardados en: {carpeta}/")
        for r in rutas:
            print(f"  → {r.name}")
