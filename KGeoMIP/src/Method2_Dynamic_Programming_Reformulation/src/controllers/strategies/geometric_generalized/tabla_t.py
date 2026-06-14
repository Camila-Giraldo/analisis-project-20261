import numpy as np
from typing import Dict, Tuple
from numpy.typing import NDArray
from .utils import hamming


class TablaT:
    """
    Tabla T del documento GeoMIP con cálculo adaptativo según n.

    Implementa ec. 3.1:  t_x(i,j) = γ · (|X[i]-X[j]| + Σ_{k∈N(i,j)} t_x(k,j))
    donde γ = 2^(-d(i,j)).

    Para n ≤ 15 calcula T exacta; para n > 15 usa umbral de exactitud
    y aproxima pares distantes (el error es mínimo porque γ → 0 para d grande).
    """

    def __init__(self, tpm_estado_nodo: NDArray[np.float64], n: int):
        self.n = n
        self.tpm = tpm_estado_nodo
        self.num_estados = 2 ** n
        self.umbral_exacto = self._calcular_umbral()
        self._cache: Dict[Tuple[int, int, int], float] = {}
        self._cache_cruzados: Dict[Tuple[int, int], float] = {}

    def _calcular_umbral(self) -> int:
        if self.n <= 15:
            return self.n
        elif self.n <= 20:
            return self.n // 2
        elif self.n <= 22:
            return self.n // 3
        else:
            return 3

    def get(self, variable: int, i: int, j: int) -> float:
        if i == j:
            return 0.0
        clave = (variable, i, j)
        if clave in self._cache:
            return self._cache[clave]

        d = hamming(i, j)
        resultado = (
            self._exacto(variable, i, j, d)
            if d <= self.umbral_exacto
            else self._aproximar(variable, i, j, d)
        )
        self._cache[clave] = resultado
        return resultado

    def costos_cruzados(self, var_i: int, var_j: int) -> float:
        """
        Costo cruzado promedio entre dos variables.
        Bajo → independencia causal; alto → dependencia causal.
        """
        clave = (min(var_i, var_j), max(var_i, var_j))
        if clave in self._cache_cruzados:
            return self._cache_cruzados[clave]

        total, conteo = 0.0, 0
        for i in range(self.num_estados):
            for bit in range(self.n):
                j = i ^ (1 << bit)
                if j < self.num_estados:
                    total += abs(self.get(var_i, i, j) - self.get(var_j, i, j))
                    conteo += 1

        resultado = total / conteo if conteo > 0 else 0.0
        self._cache_cruzados[clave] = resultado
        return resultado

    def _exacto(self, variable: int, i: int, j: int, d: int) -> float:
        gamma = 2.0 ** (-d)
        diferencia = abs(self.tpm[i, variable] - self.tpm[j, variable])
        if d == 1:
            return gamma * diferencia
        vecinos = self._vecinos_optimos(i, j, d)
        suma = sum(self._exacto(variable, k, j, d - 1) for k in vecinos)
        return gamma * (diferencia + suma)

    def _aproximar(self, variable: int, i: int, j: int, d: int) -> float:
        gamma = 2.0 ** (-d)
        diferencia = abs(self.tpm[i, variable] - self.tpm[j, variable])
        vecinos = self._vecinos_en_umbral(i, j)
        suma = sum(self.get(variable, k, j) for k in vecinos)
        return gamma * (diferencia + suma)

    def _vecinos_optimos(self, i: int, j: int, d: int) -> list:
        return [
            i ^ (1 << bit)
            for bit in range(self.n)
            if 0 <= i ^ (1 << bit) < self.num_estados
            and hamming(i ^ (1 << bit), j) == d - 1
        ]

    def _vecinos_en_umbral(self, i: int, j: int) -> list:
        frontera = {i}
        for _ in range(self.umbral_exacto):
            nueva = set()
            for estado in frontera:
                d_e = hamming(estado, j)
                if d_e > 0:
                    for v in self._vecinos_optimos(estado, j, d_e):
                        nueva.add(v)
            frontera = nueva if nueva else frontera
        return list(frontera)
