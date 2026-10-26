from __future__ import annotations

from contextlib import contextmanager

import pytest

from app.lib.contracts import IntegrityError


@contextmanager
def raises_integrity(fragment: str):
    """Assert an integrity failure without exposing its detail to visitors."""

    with pytest.raises(IntegrityError) as caught:
        yield
    assert fragment in caught.value.detail
