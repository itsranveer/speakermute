"""Shared rich console and logger, plus taming of third-party log noise."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()
logger = logging.getLogger("speakermute")

# Libraries that raise their own loggers to INFO (and attach their own handlers)
# when they are imported. Imports happen lazily, after :func:`setup_logging`, so
# :func:`quiet_third_party_loggers` has to be called again once they are loaded.
THIRD_PARTY_LOGGERS = ("lightning", "lightning.pytorch", "lightning.fabric", "pyannote")

_third_party_level = logging.WARNING


class ThirdPartyFilter(logging.Filter):
    """Drop records from other packages below the third-party level.

    Our own records always pass. This catches libraries that set their loggers to
    INFO after our setup ran and then propagate to the root handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == logger.name or record.name.startswith(logger.name + "."):
            return True
        return record.levelno >= _third_party_level


def setup_logging(verbose: bool = False) -> None:
    """Our logger at INFO (DEBUG with ``verbose``); third-party loggers at WARNING (INFO)."""
    global _third_party_level
    _third_party_level = logging.INFO if verbose else logging.WARNING
    handler = RichHandler(console=console, show_path=False, rich_tracebacks=True)
    handler.addFilter(ThirdPartyFilter())
    logging.basicConfig(
        level=_third_party_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def third_party_level() -> int:
    """The level third-party loggers should run at (WARNING, or INFO with ``--verbose``)."""
    return _third_party_level


def quiet_third_party_loggers() -> None:
    """Re-apply the third-party level to loggers that libraries reconfigure on import."""
    for name in THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(_third_party_level)
