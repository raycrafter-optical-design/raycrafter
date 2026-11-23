"""Shared test fixtures."""

import pytest

from raycrafter.core import OpticalModel


@pytest.fixture
def focused_model() -> OpticalModel:
    """A bench with one ACT508-200-A (f = 200 mm), a 10 mm collimated beam, autofocused."""
    model = OpticalModel()
    model.autofocus()
    return model
