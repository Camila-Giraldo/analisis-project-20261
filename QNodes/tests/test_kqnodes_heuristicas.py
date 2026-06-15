"""
Validación de las HEURÍSTICAS de KQNodes (voraz y recocido).

Responsabilidad única: una heurística nunca puede dar pérdida menor que el
óptimo exacto, y el recocido debe alcanzar el óptimo en ≥ 80 % de los casos.
"""

import pytest

from tests._kqnodes_helpers import TOL, kq_loss, nueva_red

PRUEBAS = [
    (4, "1111", "1111"),
    (5, "11111", "11111"),
    (6, "111111", "111111"),
    (6, "111100", "110111"),
]


class TestKQNodesHeuristicas:
    """Las heurísticas nunca deben dar pérdida menor que el exacto."""

    @pytest.mark.parametrize("bits,alc,mec", PRUEBAS)
    @pytest.mark.parametrize("k", [2, 3])
    def test_voraz_no_peor_que_exacto(self, bits, alc, mec, k):
        tpm = nueva_red(bits)
        d_ex = kq_loss(tpm, bits, alc, mec, k, metodo="exacto")
        d_vz = kq_loss(tpm, bits, alc, mec, k, metodo="voraz")
        # δ_voraz ≥ δ_exacto (una heurística no puede ser más exacta que el óptimo).
        assert d_vz >= d_ex - TOL

    @pytest.mark.parametrize("bits,alc,mec", PRUEBAS)
    @pytest.mark.parametrize("k", [2, 3])
    def test_recocido_no_peor_que_exacto(self, bits, alc, mec, k):
        tpm = nueva_red(bits)
        d_ex = kq_loss(tpm, bits, alc, mec, k, metodo="exacto")
        d_rc = kq_loss(tpm, bits, alc, mec, k, metodo="recocido", iteraciones=2000)
        assert d_rc >= d_ex - TOL

    def test_recocido_acierto_minimo_80pct(self):
        """El recocido debe alcanzar el óptimo exacto en ≥ 80 % de los casos."""
        pruebas = [(4, "1111", "1111"), (5, "11111", "11111"), (6, "111111", "111111")]
        ok = total = 0
        for bits, alc, mec in pruebas:
            tpm = nueva_red(bits)
            for k in [2, 3]:
                d_ex = kq_loss(tpm, bits, alc, mec, k, metodo="exacto")
                d_rc = kq_loss(tpm, bits, alc, mec, k, metodo="recocido", iteraciones=2000)
                ok += int(abs(d_rc - d_ex) < TOL)
                total += 1
        assert ok / total >= 0.80, f"acierto recocido: {ok}/{total} = {ok/total:.0%} < 80 %"
