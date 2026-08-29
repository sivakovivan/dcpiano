import pytest

from dcpiano.services.homography import HomographyService
from dcpiano.types.calibration import KeyboardCalibration
from dcpiano.types.config import Config
from dcpiano.types.core import Coordinate
from dcpiano.types.homography import KeyboardPoint, LandmarkPoint


def calibration() -> KeyboardCalibration:
    return KeyboardCalibration(
        top_left_corner_px=Coordinate(100, 100),
        top_right_corner_px=Coordinate(900, 120),
        bottom_right_corner_px=Coordinate(950, 600),
        bottom_left_corner_px=Coordinate(50, 620),
        top_left_corner_mm=Coordinate(0, 150),
        top_right_corner_mm=Coordinate(1200, 150),
        bottom_right_corner_mm=Coordinate(1200, 0),
        bottom_left_corner_mm=Coordinate(0, 0),
    )


def test_calibration_corners_use_bottom_left_keyboard_origin() -> None:
    service = HomographyService(calibration(), Config(1200, 150), 1000, 800)

    bottom_left = service.to_keyboard(0.05, 0.775)
    top_right = service.to_keyboard(0.9, 0.15)
    assert (bottom_left.x, bottom_left.y) == pytest.approx((0, 0), abs=1e-10)
    assert (top_right.x, top_right.y) == pytest.approx((1200, 150), abs=1e-10)


def test_xyz_conversion_round_trips_through_perspective() -> None:
    service = HomographyService(calibration(), Config(1200, 150), 1000, 800)
    original = LandmarkPoint(0.43, 0.47, -0.08)

    real_world = service.landmark_to_keyboard(original)
    restored = service.keyboard_to_landmark(real_world)

    assert (restored.x, restored.y, restored.z) == pytest.approx(
        (original.x, original.y, original.z), abs=1e-10
    )
    assert isinstance(real_world, KeyboardPoint)


def test_points_outside_keyboard_are_not_clipped() -> None:
    service = HomographyService(calibration(), Config(1200, 150), 1000, 800)

    assert service.to_keyboard(-0.1, 0.5).x < 0


def test_rejects_degenerate_calibration() -> None:
    invalid = calibration()
    invalid.top_left_corner_px = Coordinate(0, 0)
    invalid.top_right_corner_px = Coordinate(1, 1)
    invalid.bottom_right_corner_px = Coordinate(2, 2)
    invalid.bottom_left_corner_px = Coordinate(3, 3)

    with pytest.raises(ValueError, match="valid homography"):
        HomographyService(invalid, Config(1200, 150), 1000, 800)
