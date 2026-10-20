import logging

import pytest

from speakermute import log


@pytest.fixture(autouse=True)
def _restore_logging():
    yield
    log.setup_logging(False)


def _record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, "msg", None, None)


def test_default_hides_third_party_info_but_keeps_ours_and_their_warnings():
    log.setup_logging(False)
    flt = log.ThirdPartyFilter()
    assert flt.filter(_record("speakermute", logging.INFO))
    assert flt.filter(_record("speakermute.ffmpeg", logging.DEBUG))
    assert not flt.filter(_record("lightning.pytorch.utilities.migration", logging.INFO))
    assert flt.filter(_record("lightning.pytorch", logging.WARNING))


def test_verbose_lets_third_party_info_through():
    log.setup_logging(True)
    assert log.ThirdPartyFilter().filter(_record("pyannote.audio", logging.INFO))
    assert log.third_party_level() == logging.INFO


def test_root_handler_carries_the_filter():
    log.setup_logging(False)
    (handler,) = logging.getLogger().handlers
    assert any(isinstance(f, log.ThirdPartyFilter) for f in handler.filters)


def test_quiet_third_party_loggers_overrides_import_time_levels():
    log.setup_logging(False)
    for name in log.THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)  # what lightning does on import
    log.quiet_third_party_loggers()
    assert all(logging.getLogger(n).level == logging.WARNING for n in log.THIRD_PARTY_LOGGERS)
