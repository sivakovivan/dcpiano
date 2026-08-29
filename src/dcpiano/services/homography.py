from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np

from dcpiano.types.calibration import KeyboardCalibration
from dcpiano.types.config import Config
from dcpiano.types.homography import KeyboardHomography, KeyboardPoint, LandmarkPoint


class HomographyService:
    """Convert frame-relative landmark points to keyboard millimetres and back.

    The keyboard coordinate system has its origin at the bottom-left corner,
    +x toward the bottom-right, and +y toward the top-left.  Coordinates are
    deliberately not clipped, so points outside the calibrated keyboard remain
    meaningful.
    """

    def __init__(
        self,
        calibration: KeyboardCalibration,
        config: Config,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.homography = self.create_keyboard_homography(
            calibration, config, frame_width, frame_height
        )

    @staticmethod
    def create_keyboard_homography(
        calibration: KeyboardCalibration,
        config: Config,
        frame_width: int,
        frame_height: int,
    ) -> KeyboardHomography:
        """Build and validate a reusable homography from calibration data."""
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_width and frame_height must be positive integers.")

        width = float(config.keyboard_width_mm)
        height = float(config.keyboard_height_mm)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("config.keyboard_width_mm must be positive and finite.")
        if not math.isfinite(height) or height <= 0:
            raise ValueError("config.keyboard_height_mm must be positive and finite.")

        # Keep the same winding order on both planes: TL, TR, BR, BL.
        source = np.asarray(
            [
                (calibration.top_left_corner_px.x, calibration.top_left_corner_px.y),
                (calibration.top_right_corner_px.x, calibration.top_right_corner_px.y),
                (calibration.bottom_right_corner_px.x, calibration.bottom_right_corner_px.y),
                (calibration.bottom_left_corner_px.x, calibration.bottom_left_corner_px.y),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(source)):
            raise ValueError("Calibration pixel coordinates must be finite.")

        destination = np.asarray(
            [(0.0, height), (width, height), (width, 0.0), (0.0, 0.0)],
            dtype=np.float64,
        )
        matrix = cv2.getPerspectiveTransform(source.astype(np.float32), destination.astype(np.float32))
        determinant = float(np.linalg.det(matrix))
        if not np.all(np.isfinite(matrix)) or abs(determinant) < 1e-12:
            raise ValueError("Calibration corners do not define a valid homography.")

        inverse = np.linalg.inv(matrix)
        matrix.setflags(write=False)
        inverse.setflags(write=False)
        return KeyboardHomography(
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            keyboard_width_mm=width,
            keyboard_height_mm=height,
            frame_to_keyboard=matrix,
            keyboard_to_frame=inverse,
        )

    def landmark_to_keyboard(
        self,
        point_or_x: LandmarkPoint | float,
        y: float | None = None,
        z: float | None = None,
    ) -> KeyboardPoint:
        """Convert one normalized detector point to keyboard millimetres."""
        if isinstance(point_or_x, LandmarkPoint):
            x, y_value, z_value = point_or_x.x, point_or_x.y, point_or_x.z
        else:
            if y is None:
                raise TypeError("y is required when x is supplied separately.")
            x, y_value, z_value = float(point_or_x), float(y), z
        self._require_finite(x, y_value, z_value)

        homography = self.homography
        pixel_x = x * homography.frame_width
        pixel_y = y_value * homography.frame_height
        keyboard_x, keyboard_y = self._project(
            homography.frame_to_keyboard, pixel_x, pixel_y
        )
        keyboard_z = None
        if z_value is not None:
            keyboard_z = z_value * self._x_metric_scale(pixel_x, pixel_y)
        return KeyboardPoint(keyboard_x, keyboard_y, keyboard_z)

    def keyboard_to_landmark(
        self,
        point_or_x: KeyboardPoint | float,
        y: float | None = None,
        z: float | None = None,
    ) -> LandmarkPoint:
        """Convert keyboard millimetres to normalized detector coordinates."""
        if isinstance(point_or_x, KeyboardPoint):
            x, y_value, z_value = point_or_x.x, point_or_x.y, point_or_x.z
        else:
            if y is None:
                raise TypeError("y is required when x is supplied separately.")
            x, y_value, z_value = float(point_or_x), float(y), z
        self._require_finite(x, y_value, z_value)

        homography = self.homography
        pixel_x, pixel_y = self._project(homography.keyboard_to_frame, x, y_value)
        landmark_z = None
        if z_value is not None:
            landmark_z = z_value / self._x_metric_scale(pixel_x, pixel_y)
        return LandmarkPoint(
            pixel_x / homography.frame_width,
            pixel_y / homography.frame_height,
            landmark_z,
        )

    def landmarks_to_keyboard(
        self, points: Iterable[LandmarkPoint]
    ) -> list[KeyboardPoint]:
        """Convert a sequence while reusing the precomputed matrices."""
        return [self.landmark_to_keyboard(point) for point in points]

    def keyboard_to_landmarks(
        self, points: Iterable[KeyboardPoint]
    ) -> list[LandmarkPoint]:
        """Convert a sequence while reusing the precomputed matrices."""
        return [self.keyboard_to_landmark(point) for point in points]

    # Concise aliases for callers that already establish the service's direction.
    to_keyboard = landmark_to_keyboard
    to_landmark = keyboard_to_landmark

    @staticmethod
    def _project(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
        denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        if abs(float(denominator)) < 1e-12:
            raise ValueError("Point projects to infinity under this homography.")
        return (
            float((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator),
            float((matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator),
        )

    def _x_metric_scale(self, pixel_x: float, pixel_y: float) -> float:
        """Millimetres per one frame-relative x unit at a projected point."""
        matrix = self.homography.frame_to_keyboard
        keyboard_x, keyboard_y = self._project(matrix, pixel_x, pixel_y)
        denominator = matrix[2, 0] * pixel_x + matrix[2, 1] * pixel_y + matrix[2, 2]
        dx_du = (matrix[0, 0] - matrix[2, 0] * keyboard_x) / denominator
        dy_du = (matrix[1, 0] - matrix[2, 0] * keyboard_y) / denominator
        scale = math.hypot(float(dx_du), float(dy_du)) * self.homography.frame_width
        if not math.isfinite(scale) or scale <= 1e-12:
            raise ValueError("Cannot determine depth scale at this point.")
        return scale

    @staticmethod
    def _require_finite(x: float, y: float, z: float | None) -> None:
        values = (x, y) if z is None else (x, y, z)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Point coordinates must be finite.")
