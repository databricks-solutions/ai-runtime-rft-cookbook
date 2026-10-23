from __future__ import annotations

import pytest
from support import ROW


@pytest.fixture
def row():
    return dict(ROW)
