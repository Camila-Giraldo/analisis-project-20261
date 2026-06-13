"""
Suite pytest formal para KQNodes.

Cubre los criterios de validación obligatorios del proyecto:
  - Evaluación correcta de k-particiones (exacto vs kbruteforce).
  - Consistencia con BruteForce para k=2.
  - Robustez ante distintos inputs y casos edge.
  - Heurísticas no superan al exacto; recocido alcanza ≥ 80 % de óptimos.
  - generate_k_partitions produce S(n, k) particiones válidas.

Ejecución:
    pytest tests/test_kqnodes.py -v
"""

import math
import pytest

from tests.fase0_kloss import kbruteforce, construir_subsistema, vertices_de, stirling2
from src.controllers.manager import Manager
from src.strategies.KQNodes import KQNodes
from src.strategies.force import BruteForce

TOL = 1e-5  # tolerancia float32


# ---------------------------------------------------------------------------
# Helpers locales
# ---------------------------------------------------------------------------

def _kq(tpm, bits, alc, mec, k, metodo="exacto", **kw):
    """Crea KQNodes y ejecuta aplicar_estrategia; devuelve la pérdida."""
    estado = "1" + "0" * (bits - 1)
    cond = "1" * bits
    return float(KQNodes(tpm).aplicar_estrategia(estado, cond, alc, mec, k, metodo=metodo, **kw).perdida)


def _ref(bits, alc, mec, k):
    """Pérdida exacta de referencia via kbruteforce."""
    sub, base = construir_subsistema(bits, alc, mec)
    d, _ = kbruteforce(sub, base, k)
    return d


def _bf(tpm, bits, alc, mec):
    """Pérdida BruteForce (bipartición exacta)."""
    estado = "1" + "0" * (bits - 1)
    cond = "1" * bits
    return float(BruteForce(tpm).aplicar_estrategia(estado, cond, alc, mec).perdida)


# ---------------------------------------------------------------------------
# TestKQNodesExacto — validación exacta vs kbruteforce
# ---------------------------------------------------------------------------

class TestKQNodesExacto:
    """El método exacto debe coincidir con kbruteforce hasta la precisión float32."""

    @pytest.mark.parametrize("k", [2, 3])
    def test_coincide_con_kbruteforce_n3(self, tpm_n3, k):
        bits, alc, mec = 3, "111", "111"
        assert abs(_kq(tpm_n3, bits, alc, mec, k) - _ref(bits, alc, mec, k)) < TOL

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_coincide_con_kbruteforce_n4(self, tpm_n4, k):
        bits, alc, mec = 4, "1111", "1111"
        assert abs(_kq(tpm_n4, bits, alc, mec, k) - _ref(bits, alc, mec, k)) < TOL

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_coincide_con_kbruteforce_n5(self, tpm_n5, k):
        bits, alc, mec = 5, "11111", "11111"
        sub, base = construir_subsistema(bits, alc, mec)
        twoN = len(vertices_de(sub))
        if k > twoN:
            pytest.skip(f"k={k} > |V|={twoN}")
        assert abs(_kq(tpm_n5, bits, alc, mec, k) - _ref(bits, alc, mec, k)) < TOL

    # Validación obligatoria del spec: k=2 debe coincidir con BruteForce exacto.
    @pytest.mark.parametrize("bits,alc,mec", [
        (3, "111", "111"),
        (4, "1111", "1111"),
        (5, "11111", "11111"),
    ])
    def test_k2_coincide_con_bruteforce_biparticion(self, bits, alc, mec):
        tpm = Manager("1" + "0" * (bits - 1)).cargar_red()
        assert abs(_kq(tpm, bits, alc, mec, 2) - _bf(tpm, bits, alc, mec)) < TOL


# ---------------------------------------------------------------------------
# TestKQNodesHeuristicas — heurísticas no superan al exacto; recocido ≥ 80 %
# ---------------------------------------------------------------------------

class TestKQNodesHeuristicas:
    """Las heurísticas nunca deben dar pérdida menor que el exacto."""

    PRUEBAS = [
        (4, "1111", "1111"),
        (5, "11111", "11111"),
        (6, "111111", "111111"),
        (6, "111100", "110111"),
    ]

    @pytest.mark.parametrize("bits,alc,mec", PRUEBAS)
    @pytest.mark.parametrize("k", [2, 3])
    def test_voraz_no_peor_que_exacto(self, bits, alc, mec, k):
        tpm = Manager("1" + "0" * (bits - 1)).cargar_red()
        d_ex = _kq(tpm, bits, alc, mec, k, metodo="exacto")
        d_vz = _kq(tpm, bits, alc, mec, k, metodo="voraz")
        # δ_voraz ≥ δ_exacto (heurística nunca puede ser más exacta)
        assert d_vz >= d_ex - TOL

    @pytest.mark.parametrize("bits,alc,mec", PRUEBAS)
    @pytest.mark.parametrize("k", [2, 3])
    def test_recocido_no_peor_que_exacto(self, bits, alc, mec, k):
        tpm = Manager("1" + "0" * (bits - 1)).cargar_red()
        d_ex = _kq(tpm, bits, alc, mec, k, metodo="exacto")
        d_rc = _kq(tpm, bits, alc, mec, k, metodo="recocido", iteraciones=2000)
        assert d_rc >= d_ex - TOL

    def test_recocido_acierto_minimo_80pct(self):
        """El recocido debe alcanzar el óptimo exacto en ≥ 80 % de los casos."""
        pruebas = [(4, "1111", "1111"), (5, "11111", "11111"), (6, "111111", "111111")]
        ok = total = 0
        for bits, alc, mec in pruebas:
            tpm = Manager("1" + "0" * (bits - 1)).cargar_red()
            for k in [2, 3]:
                d_ex = _kq(tpm, bits, alc, mec, k, metodo="exacto")
                d_rc = _kq(tpm, bits, alc, mec, k, metodo="recocido", iteraciones=2000)
                ok += int(abs(d_rc - d_ex) < TOL)
                total += 1
        assert ok / total >= 0.80, f"acierto recocido: {ok}/{total} = {ok/total:.0%} < 80 %"


# ---------------------------------------------------------------------------
# TestKQNodesEdgeCases — robustez ante inputs extremos
# ---------------------------------------------------------------------------

class TestKQNodesEdgeCases:

    def test_k_igual_num_vertices(self, tpm_n3):
        """k = |V| debe funcionar sin error (cada vértice en su propio bloque)."""
        bits, alc, mec = 3, "111", "111"
        sub, _ = construir_subsistema(bits, alc, mec)
        twoN = len(vertices_de(sub))
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


# ---------------------------------------------------------------------------
# TestGenerateKPartitions — API pública generate_k_partitions
# ---------------------------------------------------------------------------

class TestGenerateKPartitions:

    def test_cantidad_particiones_es_stirling(self):
        """generate_k_partitions(k) debe producir exactamente S(n, k) particiones."""
        bits = 4
        estado, cond, alc, mec = "1000", "1111", "1111", "1111"
        tpm = Manager(estado).cargar_red()
        kq = KQNodes(tpm)
        kq.sia_preparar_subsistema(estado, cond, alc, mec)
        n = len(list(kq.sia_subsistema.dims_ncubos))
        for k in range(1, n + 1):
            particiones = list(kq.generate_k_partitions(k))
            esperado = stirling2(n, k)
            assert len(particiones) == esperado, (
                f"S({n},{k})={esperado} pero generate_k_partitions produjo {len(particiones)}"
            )

    def test_particiones_son_disjuntas_y_cubren(self):
        """Cada partición generada debe cubrir todos los presentes sin solape."""
        bits = 3
        estado, cond, alc, mec = "100", "111", "111", "111"
        tpm = Manager(estado).cargar_red()
        kq = KQNodes(tpm)
        kq.sia_preparar_subsistema(estado, cond, alc, mec)
        presentes = frozenset(int(i) for i in kq.sia_subsistema.dims_ncubos)
        for k in range(1, len(presentes) + 1):
            for bloques in kq.generate_k_partitions(k):
                union = frozenset(e for b in bloques for e in b)
                pares = [(i, j) for i, b1 in enumerate(bloques) for j, b2 in enumerate(bloques)
                         if i < j for _ in [set(b1) & set(b2)] if set(b1) & set(b2)]
                assert union == presentes, "los bloques no cubren todos los presentes"
                assert not pares, "hay bloques solapados"
