import logging

from hpo.lunar_lander.logging import configure_file_logging


def test_configure_file_logging_writes_hpo_logs_only_to_file(tmp_path) -> None:
    logger = logging.getLogger("hpo")
    old_handlers = logger.handlers[:]
    old_level = logger.level
    old_propagate = logger.propagate

    try:
        logger.handlers.clear()
        configure_file_logging(tmp_path)

        logger.info("test message")
        handler = logger.handlers[0]
        handler.flush()

        line = (tmp_path / "hpo.log").read_text().strip()
        assert line.endswith(" INFO hpo: test message")
        assert "," not in line
        assert logger.propagate is False
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers[:] = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate
