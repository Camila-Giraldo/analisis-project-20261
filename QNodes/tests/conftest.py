"""
Fixtures compartidas de la suite KQNodes.

``conftest.py`` es descubierto automáticamente por pytest: las fixtures aquí
definidas están disponibles en todos los ``test_kqnodes_*.py`` sin importación
explícita.

Responsabilidad única: preparación del entorno de pruebas (página de red de
muestra + profiling desactivado) y generación de las TPM de referencia.
"""

import pytest

from src.controllers.manager import Manager
from src.models.base.application import aplicacion
from src.middlewares.profile import gestor_perfilado


@pytest.fixture(autouse=True, scope="module")
def _entorno_pruebas():
    """Fija la página de red de muestra "A" (como hace exec.py) y DESACTIVA el
    profiling, que de otro modo escribiría HTML en review/profiling/ por cada
    llamada a los algoritmos de búsqueda."""
    aplicacion.set_pagina_red_muestra("A")
    aplicacion.desactivar_profiling()
    gestor_perfilado.enabled = False
    yield


@pytest.fixture(scope="module")
def tpm_n3():
    return Manager("100").cargar_red()


@pytest.fixture(scope="module")
def tpm_n4():
    return Manager("1000").cargar_red()


@pytest.fixture(scope="module")
def tpm_n5():
    return Manager("10000").cargar_red()
