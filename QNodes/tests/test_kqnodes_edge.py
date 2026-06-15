"""
Robustez de KQNodes ante inputs extremos y parámetros inválidos.

Responsabilidad única: casos borde (k=|V|, subsistema mínimo) y validación de
parámetros (k fuera de rango, método desconocido).
"""

import math

import pytest

from src.strategies.KQNodes import KQNodes
from tests._kqnodes_helpers import build_kq, vertices


class TestKQNodesEdgeCases:

    def test_k_igual_num_vertices(self, tpm_n3):
        """k = |V| debe funcionar (cada vértice en su propio bloque)."""
        bits, alc, mec = 3, "111", "111"
        twoN = len(vertices(build_kq(tpm_n3, bits, alc, mec)))
        sol = KQNodes(tpm_n3).aplicar_estrategia(
            "100", "111", alc, mec, twoN, metodo="exacto"
        )
        assert math.isfinite(float(sol.perdida))

    def test_k_igual_2_subsistema_minimo(self, tpm_n3):
        """Subsistema con 1 futuro + 1 presente, k=2 debe resolver sin inf."""
        sol = KQNodes(tpm_n3).aplicar_estrategia("100", "111", "100", "100", 2)
        assert math.isfinite(float(sol.perdida))

    def test_parametros_invalidos_k_fuera_rango(self, tpm_n3):
        """k=1 y k > |V| deben lanzar ValueError."""
        with pytest.raises(ValueError):
            KQNodes(tpm_n3).aplicar_estrategia("100", "111", "111", "111", 1)
        with pytest.raises(ValueError):
            KQNodes(tpm_n3).aplicar_estrategia("100", "111", "111", "111", 99)

    def test_metodo_desconocido_lanza_error(self, tpm_n3):
        """Un método no soportado debe lanzar ValueError."""
        with pytest.raises(ValueError):
            KQNodes(tpm_n3).aplicar_estrategia("100", "111", "111", "111", 2, metodo="xyz")
