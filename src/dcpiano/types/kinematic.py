from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from dcpiano.types.video import FrameMetadata


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float


KinematicData: TypeAlias = float | Vector3


@dataclass(frozen=True, slots=True)
class KinematicValue:
    """One registered metric. ``value=None`` explicitly denotes invalid data."""

    value: KinematicData | None

    @property
    def valid(self) -> bool:
        return self.value is not None


@dataclass(slots=True)
class KinematicFrame:
    """All registered kinematic values corresponding to one landmark frame."""

    metadata: FrameMetadata
    values: dict[str, KinematicValue]

    @property
    def timestamp(self) -> float:
        return self.metadata.timestamp

    def get(self, path: str) -> KinematicData | None:
        metric = self.values.get(path)
        return None if metric is None else metric.value
