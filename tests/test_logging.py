import json
import logging

from shelfsight.logging_config import setup_logging


def test_json_format_includes_extra_fields(tmp_path):
    log_file = tmp_path / "logs" / "t.log"
    logger = setup_logging("INFO", json_format=True, log_file=log_file)
    logger.info("hello", extra={"frame": 7})
    for h in logger.handlers:
        h.flush()
    record = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["message"] == "hello"
    assert record["level"] == "INFO"
    assert record["frame"] == 7


def test_setup_is_idempotent():
    setup_logging("INFO")
    logger = setup_logging("DEBUG")
    ours = [h for h in logger.handlers if getattr(h, "_shelfsight_handler", False)]
    assert len(ours) == 1
    assert logger.level == logging.DEBUG
