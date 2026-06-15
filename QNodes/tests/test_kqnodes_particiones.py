"""
API pública ``generate_k_partitions``.

Responsabilidad única: la generación de k-particiones de los presentes debe
producir exactamente S(n, k) particiones, todas disjuntas y que cubren el
conjunto completo de presentes.
"""

from tests._kqnodes_helpers import build_kq, stirling2


class TestGenerateKPartitions:

    def test_cantidad_particiones_es_stirling(self, tpm_n4):
        """generate_k_partitions(k) debe producir exactamente S(n, k) particiones."""
        kq = build_kq(tpm_n4, 4, "1111", "1111")
        n = len(list(kq.sia_subsistema.dims_ncubos))
        for k in range(1, n + 1):
            particiones = list(kq.generate_k_partitions(k))
            esperado = stirling2(n, k)
            assert len(particiones) == esperado, (
                f"S({n},{k})={esperado} pero generate_k_partitions dio {len(particiones)}"
            )

    def test_particiones_son_disjuntas_y_cubren(self, tpm_n3):
        """Cada partición generada cubre todos los presentes sin solape."""
        kq = build_kq(tpm_n3, 3, "111", "111")
        presentes = frozenset(int(i) for i in kq.sia_subsistema.dims_ncubos)
        for k in range(1, len(presentes) + 1):
            for bloques in kq.generate_k_partitions(k):
                union = frozenset(e for b in bloques for e in b)
                assert union == presentes, "los bloques no cubren todos los presentes"
                vistos: set = set()
                for b in bloques:
                    s = set(b)
                    assert not (s & vistos), "hay bloques solapados"
                    vistos |= s
