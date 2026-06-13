"""
Fixtures compartidos para la suite pytest de KQNodes.

Uso:
    pytest tests/test_kqnodes.py -v
"""

import logging
import sys
import os

import pytest

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Silenciar SafeLogger antes de cualquier importación del proyecto.
from src.middlewares import slogger  # noqa: E402

_SILENT = logging.getLogger("pytest_silent")
_SILENT.addHandler(logging.NullHandler())
_SILENT.setLevel(logging.CRITICAL + 1)
slogger.SafeLogger.__init__ = lambda self, *a, **kw: setattr(self, "_logger", _SILENT)

from src.models.base.application import aplicacion  # noqa: E402

aplicacion.profiler_habilitado = False

from src.controllers.manager import Manager  # noqa: E402
from src.strategies.KQNodes import KQNodes  # noqa: E402
from src.strategies.q_nodes import QNodes  # noqa: E402


def _tpm(bits: int):
    """Carga la TPM de N{bits} con estado inicial '1' + '0'*(bits-1)."""
    estado = "1" + "0" * (bits - 1)
    return Manager(estado).cargar_red()


@pytest.fixture(scope="session")
def tpm_n3():
    return _tpm(3)


@pytest.fixture(scope="session")
def tpm_n4():
    return _tpm(4)


@pytest.fixture(scope="session")
def tpm_n5():
    return _tpm(5)


def _subsistema(bits: int, alcance: str, mecanismo: str):
    """Devuelve (subsistema, dists_marginales) para el subsistema completo de N{bits}."""
    estado = "1" + "0" * (bits - 1)
    cond = "1" * bits
    q = QNodes(_tpm(bits))
    q.sia_preparar_subsistema(estado, cond, alcance, mecanismo)
    return q.sia_subsistema, q.sia_dists_marginales


@pytest.fixture(scope="session")
def subsistema_n3():
    return _subsistema(3, "111", "111")


@pytest.fixture(scope="session")
def subsistema_n4():
    return _subsistema(4, "1111", "1111")


@pytest.fixture(scope="session")
def subsistema_n5():
    return _subsistema(5, "11111", "11111")
