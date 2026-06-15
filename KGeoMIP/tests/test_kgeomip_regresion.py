"""
Regresión de KGeoMIP (estrategia DP).

Responsabilidad única: verificar que la estrategia DP produce exactamente
los mismos valores de partición y pérdida que se obtuvieron en las ejecuciones
de referencia sobre la red N10A (estado_inicial="1000000000",
condiciones="1111111111"). Sirve como red de seguridad para el refactor
del monolito kgeomip.py.

Valores de referencia extraídos de:
  - results/k2/resultados_Geometric_10A2.xlsx  (filas 1-3)
  - results/k3/resultados_Geometric_10A3.xlsx  (filas 1-3)
"""

import pytest

from src.controllers.strategies.KGeoMIP import KGeoMIP

TOL = 1e-5

# (alcance, mecanismo, particion_esperada, perdida_esperada)
CASOS_K2 = [
    (
        "1111111111", "1111111111",
        "|  A,B,C,D,F,G,H,I,J  || E |\n| a,b,c,d,e,f,g,h,i,j || ∅ |",
        0.478515625,
    ),
    (
        "1111111111", "1111111110",
        "| A,B,C,D,E,F,G,H,J || I |\n| a,b,c,d,e,f,g,h,i || ∅ |",
        5.859375e-3,
    ),
    (
        "1111111111", "0111111111",
        "| A,B,C,D,E,G,H,I,J || F |\n| b,c,d,e,f,g,h,i,j || ∅ |",
        9.765625e-4,
    ),
]

CASOS_K3 = [
    (
        "1111111111", "1111111111",
        "| E || F || A,B,C,D,G,H,I,J |\n| e || f || a,b,c,d,g,h,i,j |",
        3.525390625,
    ),
    (
        "1111111111", "1111111110",
        "| I || A,B,C,D,E,F,G,H || J |\n| i || a,b,c,d,e,f,g,h || ∅ |",
        2.8017578125,
    ),
    (
        "1111111111", "0111111111",
        "| F || B,C,D,E,G,I,J || A,H |\n| f || b,c,d,e,g,i,j || h |",
        0.548828125,
    ),
]


class TestKGeoMIPRegresion:
    """La estrategia DP debe reproducir los resultados de referencia."""

    @pytest.mark.parametrize("alcance,mecanismo,particion_esperada,perdida_esperada", CASOS_K2)
    def test_k2_n10(self, manager_n10, tpm_n10, alcance, mecanismo, particion_esperada, perdida_esperada):
        sol = KGeoMIP(manager_n10, k=2).aplicar_estrategia(
            "1111111111", alcance, mecanismo, tpm_n10
        )
        assert sol.particion.replace("\r\n", "\n") == particion_esperada
        assert abs(float(sol.perdida) - perdida_esperada) < TOL

    @pytest.mark.parametrize("alcance,mecanismo,particion_esperada,perdida_esperada", CASOS_K3)
    def test_k3_n10(self, manager_n10, tpm_n10, alcance, mecanismo, particion_esperada, perdida_esperada):
        sol = KGeoMIP(manager_n10, k=3).aplicar_estrategia(
            "1111111111", alcance, mecanismo, tpm_n10
        )
        assert sol.particion.replace("\r\n", "\n") == particion_esperada
        assert abs(float(sol.perdida) - perdida_esperada) < TOL
