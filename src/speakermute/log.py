"""Shared rich console and logger."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()
logger = logging.getLogger("speakermute")


def setup_logging(verbose: bool = False) -> None:
    """Our logger at INFO (DEBUG with ``verbose``); third-party loggers at WARNING (INFO)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
        force=True,
    )
    logger.setLevel(level)
