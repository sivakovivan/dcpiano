from __future__ import annotations

import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TextIO


class LogLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class Logger:
    """Simple console and file logger for dcpiano processing sessions."""

    _COLORS = {
        LogLevel.INFO: "\033[36m",
        LogLevel.WARNING: "\033[33m",
        LogLevel.ERROR: "\033[31m",
        LogLevel.SUCCESS: "\033[32m",
    }

    _RESET = "\033[0m"

    def __init__(
        self,
        name: str = "dcpiano",
        *,
        use_colors: bool = True,
        show_timestamp: bool = True,
        log_filename: str = "process.log",
    ) -> None:
        self.name = name
        self.use_colors = use_colors
        self.show_timestamp = show_timestamp
        self.log_filename = log_filename

        self._log_file: Path | None = None
        self._write_lock = Lock()

    @property
    def log_file(self) -> Path | None:
        """Return the current processing session log file."""
        return self._log_file

    def start_session(
        self,
        output_directory: str | Path,
        *,
        filename: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Start logging to a processing session's output directory.

        The output directory is created when it does not already exist.
        """
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        log_filename = filename or self.log_filename
        self._log_file = output_path / log_filename

        if overwrite:
            self._log_file.write_text("", encoding="utf-8")

        self.info(f"Processing session started: {output_path}")

        return self._log_file

    def end_session(self) -> None:
        """Log the session completion and detach the current log file."""
        if self._log_file is None:
            return

        self.info("Processing session ended")
        self._log_file = None

    def info(self, message: object) -> None:
        self._log(LogLevel.INFO, message)

    def warning(self, message: object) -> None:
        self._log(LogLevel.WARNING, message)

    def error(self, message: object) -> None:
        self._log(LogLevel.ERROR, message, stream=sys.stderr)

    def success(self, message: object) -> None:
        self._log(LogLevel.SUCCESS, message)

    def _log(
        self,
        level: LogLevel,
        message: object,
        *,
        stream: TextIO = sys.stdout,
    ) -> None:
        formatted_message = self._format_message(level, message)

        self._write_to_console(
            level=level,
            message=formatted_message,
            stream=stream,
        )
        self._write_to_file(formatted_message)

    def _format_message(
        self,
        level: LogLevel,
        message: object,
    ) -> str:
        parts: list[str] = []

        if self.show_timestamp:
            timestamp = datetime.now().astimezone().isoformat(
                sep=" ",
                timespec="seconds",
            )
            parts.append(timestamp)

        parts.extend((self.name, level.value))

        prefix = " | ".join(parts)
        return f"[{prefix}] {message}"

    def _write_to_console(
        self,
        *,
        level: LogLevel,
        message: str,
        stream: TextIO,
    ) -> None:
        console_message = message

        if self.use_colors and stream.isatty():
            console_message = (
                f"{self._COLORS[level]}"
                f"{message}"
                f"{self._RESET}"
            )

        print(console_message, file=stream, flush=True)

    def _write_to_file(self, message: str) -> None:
        if self._log_file is None:
            return

        with self._write_lock:
            with self._log_file.open("a", encoding="utf-8") as file:
                file.write(f"{message}\n")


logger = Logger()