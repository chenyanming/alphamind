"""Control noisy native model output without hiding Python exceptions."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

_NATIVE_STDERR_LOCK = Lock()


@contextmanager
def native_stderr(*, visible: bool) -> Iterator[None]:
    """Temporarily hide native stderr, restoring it before errors propagate."""
    if visible:
        yield
        return
    with _NATIVE_STDERR_LOCK:
        saved_fd: int | None = None
        null_fd: int | None = None
        try:
            sys.stderr.flush()
            stderr_fd = sys.stderr.fileno()
            saved_fd = os.dup(stderr_fd)
            null_fd = os.open(os.devnull, os.O_WRONLY)
        except (AttributeError, OSError):
            if saved_fd is not None:
                os.close(saved_fd)
            if null_fd is not None:
                os.close(null_fd)
            yield
            return
        try:
            os.dup2(null_fd, stderr_fd)
            yield
        finally:
            try:
                sys.stderr.flush()
            finally:
                os.dup2(saved_fd, stderr_fd)
                os.close(saved_fd)
                os.close(null_fd)
