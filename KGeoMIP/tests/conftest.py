"""
Fixtures compartidas de la suite KGeoMIP.

Responsabilidad única: preparar el entorno de pruebas (deshabilitar
profiling y síntesis de voz) y proveer la TPM de referencia para N=10.
"""

import numpy as np
import pytest
from pathlib import Path

from src.controllers.manager import Manager
from src.middlewares.profile import profiler_manager
from src.models.base.application import aplicacion


@pytest.fixture(autouse=True, scope="session")
def _deshabilitar_profiling():
    """Desactiva el profiler y la síntesis de voz para toda la sesión de tests."""
    aplicacion.profiler_habilitado = False
    profiler_manager.enabled = False
    yield


@pytest.fixture(scope="module")
def tpm_n10():
    path = Path(__file__).parent.parent / "src" / ".samples" / "N10A.npy"
    return np.load(str(path), mmap_mode="r")


@pytest.fixture(scope="module")
def manager_n10():
    return Manager("1000000000")
