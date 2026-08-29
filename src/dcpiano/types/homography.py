from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """A detector point in frame-relative coordinates.

    ``x`` and ``y`` are fractions of the frame width and height.  ``z`` uses
    the detector's x/frame-width scale (the convention used by MediaPipe).
    """

    x: float
    y: float
    z: float | None = None


@dataclass(frozen=True, slots=True)
class KeyboardPoint:
    """A point in millimetres from the keyboard's bottom-left corner."""

    x: float
    y: float
    z: float | None = None


@dataclass(frozen=True, slots=True, eq=False)
class KeyboardHomography:
    """Precomputed projective transforms between a frame and a keyboard.

    The matrices map pixel coordinates to millimetres and back.  Keeping both
    avoids computing an inverse for every point.
    """

    frame_width: int
    frame_height: int
    keyboard_width_mm: float
    keyboard_height_mm: float
    frame_to_keyboard: np.ndarray
    keyboard_to_frame: np.ndarray

