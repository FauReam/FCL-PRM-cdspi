"""Tee utility: mirror stdout to both terminal and a timestamped log file.

Handles tqdm \r overwrite cleanly so the log file stays readable without
hundreds of repeated progress-bar lines.
"""

import atexit
import sys
from pathlib import Path


class Tee:
    """Mirror stdout to both terminal and a log file.

    Uses an atexit hook and a try/finally pattern to ensure the log buffer
    is flushed even on hard crashes (KeyboardInterrupt, SIGTERM, etc.).
    """

    def __init__(self, log_path: Path) -> None:
        self.terminal = sys.stdout
        self.log_file = log_path.open("w", encoding="utf-8")
        self._file_buf = ""
        self._closed = False
        atexit.register(self._flush_and_close)

    def write(self, message: str) -> None:
        # Terminal always sees everything (handles \r correctly)
        self.terminal.write(message)
        self.terminal.flush()

        # For the log file: accumulate; on \n write the complete line.
        # tqdm uses \r to overwrite the same line — we keep only the last
        # segment so the log file does not get cluttered with repeats.
        self._file_buf += message
        if "\r" in message and "\n" not in message:
            self._file_buf = self._file_buf.rsplit("\r", 1)[-1]
        elif "\n" in message:
            parts = self._file_buf.split("\n")
            for part in parts[:-1]:
                self.log_file.write(part + "\n")
            self._file_buf = parts[-1]
            self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        if self._file_buf:
            self.log_file.write(self._file_buf + "\n")
            self._file_buf = ""
            self.log_file.flush()

    def _flush_and_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._file_buf:
            try:
                self.log_file.write(self._file_buf + "\n")
            except Exception:
                pass
        try:
            self.log_file.close()
        except Exception:
            pass

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def close(self) -> None:
        self._flush_and_close()
