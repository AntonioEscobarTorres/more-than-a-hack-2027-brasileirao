"""Configuracao compartilhada do pytest para a suite de testes de pre-processamento."""
from __future__ import annotations

import os
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Registra a opcao --master para validar tanto o mock quanto o master_features.parquet real."""
    parser.addoption(
        "--master",
        action="store",
        default=None,
        help="Caminho para um arquivo Parquet master (mock ou real) a ser validado pelos testes de schema.",
    )
