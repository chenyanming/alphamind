from __future__ import annotations

import os
import sys
import unittest
from collections.abc import Callable

from native_output import native_stderr


def _capture_native_stderr(callback: Callable[[], None]) -> bytes:
    stderr_fd = sys.stderr.fileno()
    read_fd, write_fd = os.pipe()
    saved_fd = os.dup(stderr_fd)
    try:
        os.dup2(write_fd, stderr_fd)
        callback()
        sys.stderr.flush()
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(write_fd)
    try:
        return os.read(read_fd, 4_096)
    finally:
        os.close(read_fd)


class NativeOutputTests(unittest.TestCase):
    def test_product_mode_hides_native_stderr(self) -> None:
        def emit() -> None:
            with native_stderr(visible=False):
                os.write(sys.stderr.fileno(), b"ggml_metal_init: allocating\n")

        self.assertEqual(_capture_native_stderr(emit), b"")

    def test_debug_mode_preserves_native_stderr(self) -> None:
        def emit() -> None:
            with native_stderr(visible=True):
                os.write(sys.stderr.fileno(), b"ggml_metal_init: allocating\n")

        self.assertEqual(
            _capture_native_stderr(emit),
            b"ggml_metal_init: allocating\n",
        )

    def test_exceptions_escape_after_native_stderr_is_restored(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            with native_stderr(visible=False):
                raise RuntimeError("model failed")

        self.assertEqual(
            _capture_native_stderr(
                lambda: os.write(sys.stderr.fileno(), b"restored\n")
            ),
            b"restored\n",
        )


if __name__ == "__main__":
    unittest.main()
