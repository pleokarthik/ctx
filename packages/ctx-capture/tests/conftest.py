import logging

import pytest


@pytest.fixture(autouse=True)
def ctx_home(tmp_path, monkeypatch):
    ctx_dir = tmp_path / ".ctx"
    monkeypatch.setattr("ctx_capture.store._ctx_dir", lambda: ctx_dir)
    logger = logging.getLogger("ctx-capture")
    logger.handlers.clear()
    yield tmp_path
    logger.handlers.clear()
