from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from dcpiano.types.kinematic import KinematicData, KinematicFrame
from dcpiano.types.landmark import LandmarkFrame, RawLandmark


@dataclass(frozen=True, slots=True)
class KinematicContext:
    """Read-only history made available to metric calculators."""

    landmark_frames: Sequence[LandmarkFrame]
    kinematic_frames: Sequence[KinematicFrame]

    @property
    def current_landmark_frame(self) -> LandmarkFrame:
        return self.landmark_frames[-1]

    def landmark(self, landmark_id: str) -> RawLandmark | None:
        return self.current_landmark_frame.landmarks.get(landmark_id)


class KinematicCalculator(ABC):
    """Registration contract for one value in a kinematic frame."""

    path: str
    landmark_inputs: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()

    @abstractmethod
    def calculate(self, context: KinematicContext) -> KinematicData | None:
        ...
