"""
Stdlib logging configured for CLI-friendly output.

Why a tiny wrapper around logging?
----------------------------------
Every script in this repo wants the same log format and the same
``LOG_LEVEL`` env override. Inlining ``logging.basicConfig`` in each
script breaks the moment two scripts run in the same process
(``basicConfig`` is no-op after the first call). This helper keeps
the format consistent and the configuration idempotent.

Output format: ``YYYY-MM-DDTHH:MM:SS LEVEL name :: message``. The
``::`` separator is unusual enough that grepping logs is easy.
"""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = "thoughttracker_ml") -> logging.Logger:
    """Return a configured logger.

    Idempotent: if the logger already has handlers (e.g. because
    another module called ``get_logger`` with the same name first),
    we return it as-is instead of double-attaching handlers — that
    would cause each log line to print twice.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # pragma: no cover  (only exercised on second call in a fresh process)

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s :: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
