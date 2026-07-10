from dataclasses import dataclass

from dcpiano.types.core import Coordinate


@dataclass
class KeyboardCalibration:
    top_right_corner_px: Coordinate
    bottom_right_corner_px: Coordinate
    bottom_left_corner_px: Coordinate
    top_left_corner_px: Coordinate
    
    top_right_corner_mm: Coordinate
    bottom_right_corner_mm: Coordinate
    bottom_left_corner_mm: Coordinate
    top_left_corner_mm: Coordinate