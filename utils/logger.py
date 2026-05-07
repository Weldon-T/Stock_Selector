import logging
import sys
from pathlib import Path

_logger: logging.Logger | None = None


def setup_logger(
    log_dir: str = "./logs",
    log_file: str = "run.log",
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    global _logger

    _logger = logging.getLogger("stock_selector")
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _logger.handlers:
        _logger.handlers.clear()

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        Path(log_dir) / log_file, encoding="utf-8"
    )
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(logging.DEBUG)
    _logger.addHandler(file_handler)

    if console:
        console_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_fmt)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        _logger.addHandler(console_handler)

    return _logger


def get_logger() -> logging.Logger:
    if _logger is None:
        return setup_logger()
    return _logger
