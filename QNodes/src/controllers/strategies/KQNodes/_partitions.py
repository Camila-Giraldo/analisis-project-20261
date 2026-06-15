"""
Generación de k-particiones de conjuntos (número de Stirling de 2.ª especie)
y constantes compartidas de evaluación.
"""

#: Conjunto de presentes vacío reutilizable (clave de coste de un bloque sin
#: presentes). Se define una sola vez para no reconstruir ``frozenset()`` en
#: cada evaluación del lazo caliente.
VACIO: frozenset = frozenset()


def particiones_en_k(elementos: list, k: int):
    """
    Genera todas las particiones de `elementos` en EXACTAMENTE k bloques no vacíos
    (particiones de conjunto sin etiqueta, número de Stirling de 2.ª especie).

    Args:
        elementos: lista de elementos a particionar.
        k: número exacto de bloques no vacíos.

    Yields:
        list[list]: una partición como lista de k bloques (listas no vacías).
    """
    n = len(elementos)
    if k < 1 or k > n:
        return
    if k == 1:
        yield [list(elementos)]
        return
    if k == n:
        yield [[e] for e in elementos]
        return
    primero, resto = elementos[0], elementos[1:]
    for parte in particiones_en_k(resto, k - 1):
        yield [[primero]] + parte
    for parte in particiones_en_k(resto, k):
        for i in range(len(parte)):
            yield parte[:i] + [[primero] + parte[i]] + parte[i + 1:]
