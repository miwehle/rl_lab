import logging

from hpo.lunar_lander.logging import configure_file_logging, log_call


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
        assert " INFO     hpo:" in line
        assert line.endswith(" test message")
        assert "," not in line
        assert logger.propagate is False
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers[:] = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_log_call_uses_function_definition_line(tmp_path) -> None:
    logger = logging.getLogger("hpo.test")

    def example_function() -> None:
        pass

    example_function.__module__ = "hpo.test"
    example = log_call(example_function)

    configure_file_logging(tmp_path)
    example()
    handler = logging.getLogger("hpo").handlers[0]
    handler.flush()

    lines = (tmp_path / "hpo.log").read_text().splitlines()
    source = f"hpo.test:{example.__wrapped__.__code__.co_firstlineno}"
    assert source in lines[0]
    assert lines[0].endswith("-> example_function")
    assert source in lines[1]
    assert lines[1].endswith("<- example_function")

def test_log_call_indents_nested_and_regular_logs(tmp_path) -> None:
    logger = logging.getLogger("hpo.test")

    def inner_function() -> None:
        logger.info("regular log")

    inner_function.__module__ = "hpo.test"
    inner = log_call(inner_function)

    def outer_function() -> None:
        inner()

    outer_function.__module__ = "hpo.test"
    outer = log_call(outer_function)

    configure_file_logging(tmp_path)
    outer()
    handler = logging.getLogger("hpo").handlers[0]
    handler.flush()

    lines = (tmp_path / "hpo.log").read_text().splitlines()
    assert lines[0].endswith("-> outer_function")
    assert lines[1].endswith("   -> inner_function")
    assert lines[2].endswith("      regular log")
    assert lines[3].endswith("   <- inner_function")
    assert lines[4].endswith("<- outer_function")
