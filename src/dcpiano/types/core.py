from dataclasses import dataclass

number = float | int

@dataclass
class Coordinate:
    x: number
    y: number