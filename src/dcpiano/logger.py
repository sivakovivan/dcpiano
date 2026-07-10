from __future__ import annotations

import sys
from datetime import datetime
from enum import Enum
from typing import TextIO


class LogLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class Logger:
    """Simple console logger for the dcpiano application."""

    _COLORS = {
        LogLevel.INFO: "\033[36m",      # Cyan
        LogLevel.WARNING: "\033[33m",   # Yellow
        LogLevel.ERROR: "\033[31m",     # Red
        LogLevel.SUCCESS: "\033[32m",   # Green
    }

    _RESET = "\033[0m"

    def __init__(
        self,
        name: str = "dcpiano",
        *,
        use_colors: bool = True,
        show_timestamp: bool = True,
    ) -> None:
        self.name = name
        self.use_colors = use_colors
        self.show_timestamp = show_timestamp

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
        parts: list[str] = []

        if self.show_timestamp:
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
            parts.append(timestamp)

        parts.extend((self.name, level.value))

        prefix = " | ".join(parts)
        output = f"[{prefix}] {message}"

        if self.use_colors and stream.isatty():
            color = self._COLORS[level]
            output = f"{color}{output}{self._RESET}"

        print(output, file=stream, flush=True)


logger = Logger()