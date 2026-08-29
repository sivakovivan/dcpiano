from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from dcpiano.kinematics.core import KinematicCalculator, KinematicContext
from dcpiano.types.kinematic import KinematicFrame, KinematicValue
from dcpiano.types.landmark import LandmarkFrame


class KinematicService:
    """Generate a dependency-aware kinematic frame for every landmark frame."""

    def __init__(self, calculators: Sequence[KinematicCalculator]) -> None:
        self.calculators = tuple(calculators)
        self._validate_registry()

    def generate_kinematic_frames(
        self,
        landmark_frames: Sequence[LandmarkFrame],
        output_dir: str | Path | None = None,
    ) -> list[KinematicFrame]:
        self._validate_frame_order(landmark_frames)
        generated: list[KinematicFrame] = []
        for index, landmark_frame in enumerate(landmark_frames):
            frame = KinematicFrame(landmark_frame.metadata, {})
            generated.append(frame)
            context = KinematicContext(landmark_frames[: index + 1], generated)
            for calculator in self.calculators:
                value = None
                if self._inputs_valid(calculator, landmark_frame, frame):
                    value = calculator.calculate(context)
                frame.values[calculator.path] = KinematicValue(value)

        if output_dir is not None:
            self.save_kinematic_frames(generated, output_dir)
        return generated

    @staticmethod
    def save_kinematic_frames(frames: Sequence[KinematicFrame], output_dir: str | Path) -> Path:
        output_path = Path(output_dir) / "kinematic_frames.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump([asdict(frame) for frame in frames], output_file, indent=4)
        return output_path

    @staticmethod
    def _inputs_valid(calculator: KinematicCalculator, landmark_frame: LandmarkFrame, frame: KinematicFrame) -> bool:
        if any(landmark_id not in landmark_frame.landmarks for landmark_id in calculator.landmark_inputs):
            return False
        return all(frame.values[dependency].valid for dependency in calculator.prerequisites)

    def _validate_registry(self) -> None:
        registered: set[str] = set()
        for calculator in self.calculators:
            if calculator.path in registered:
                raise ValueError(f"Duplicate kinematic calculator path: {calculator.path}")
            missing = set(calculator.prerequisites) - registered
            if missing:
                raise ValueError(
                    f"Calculator {calculator.path!r} must be registered after prerequisites: {sorted(missing)}"
                )
            registered.add(calculator.path)

    @staticmethod
    def _validate_frame_order(frames: Sequence[LandmarkFrame]) -> None:
        for previous, current in zip(frames, frames[1:]):
            if current.metadata.index <= previous.metadata.index or current.metadata.timestamp < previous.metadata.timestamp:
                raise ValueError("LandmarkFrames must be ordered by increasing index and non-decreasing timestamp.")
