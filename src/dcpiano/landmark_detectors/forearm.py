from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from dcpiano.landmark_detectors.core import LandmarkDetector
from dcpiano.types.landmark import RawLandmark
from dcpiano.types.render import Connection
from dcpiano.types.video import VideoFrame


class ForearmLandmarks(StrEnum):
    """Estimated points on the visible forearm centerline (not joints)."""

    FOREARM_MIDPOINT = "FOREARM_MIDPOINT"
    FOREARM_FAR = "FOREARM_FAR"


@dataclass(slots=True)
class ForearmLandmarkConfig:
    """Geometric boundary detection and short-term tracking parameters."""

    roi_length_palms: float = 4.0
    roi_half_width_palms: float = 1.65
    hand_exclusion_padding_palms: float = 0.12
    cross_section_spacing_pixels: int = 4  # Retained for config compatibility.
    cross_section_count: int = 10
    min_cross_sections: int = 5
    midpoint_fraction: float = 0.50
    far_fraction: float = 0.82
    max_boundary_angle_degrees: float = 58.0
    max_pair_angle_degrees: float = 28.0
    min_boundary_length_palms: float = 0.65
    max_tracking_gap: int = 3
    propagation_decay: float = 0.65

    def __post_init__(self) -> None:
        if self.roi_length_palms <= 0 or self.roi_half_width_palms <= 0:
            raise ValueError("Forearm ROI dimensions must be positive")
        if self.cross_section_spacing_pixels <= 0 or self.min_cross_sections < 2:
            raise ValueError("Invalid forearm cross-section settings")
        if not 8 <= self.cross_section_count <= 12:
            raise ValueError("cross_section_count must be between 8 and 12")
        if not 0 < self.midpoint_fraction < self.far_fraction < 1:
            raise ValueError("Landmark fractions must satisfy 0 < midpoint < far < 1")
        if not 20 <= self.max_boundary_angle_degrees < 90:
            raise ValueError("max_boundary_angle_degrees must be in [20, 90)")
        if not 0 < self.max_pair_angle_degrees < 90:
            raise ValueError("max_pair_angle_degrees must be between 0 and 90")
        if self.min_boundary_length_palms <= 0 or self.max_tracking_gap < 0:
            raise ValueError("Invalid boundary length or tracking gap")
        if not 0 < self.propagation_decay < 1:
            raise ValueError("propagation_decay must be between 0 and 1")


@dataclass(slots=True)
class _Boundary:
    slope: float
    intercept: float
    start_y: float
    end_y: float
    length: float
    strength: float

    def x_at(self, y: float) -> float:
        return self.slope * y + self.intercept


@dataclass(slots=True)
class _Track:
    wrist: NDArray[np.float64]
    midpoint: NDArray[np.float64]
    farpoint: NDArray[np.float64]
    direction: NDArray[np.float64]
    width: float
    confidence: float
    missed_frames: int = 0


RENDER_CONNECTIONS: tuple[Connection, ...] = (
    ("FOREARM_MIDPOINT", "FOREARM_FAR"),
)


class ForearmLandmarkDetector(LandmarkDetector):
    """Fit paired image boundaries behind each MediaPipe-detected wrist."""

    name = "ForearmLandmarkDetector"
    render_connections = RENDER_CONNECTIONS
    _HAND_SOURCE = "HandLandmarkDetector"
    _PALM_NAMES = (
        "INDEX_FINGER_MCP",
        "MIDDLE_FINGER_MCP",
        "RING_FINGER_MCP",
        "PINKY_MCP",
    )

    def __init__(self, config: ForearmLandmarkConfig | None = None) -> None:
        self.config = config or ForearmLandmarkConfig()
        self.landmarks = tuple(ForearmLandmarks)
        self._tracks: dict[tuple[int, str], _Track] = {}

    def detect(
        self, frame: VideoFrame, existing_landmarks: dict[str, RawLandmark]
    ) -> dict[str, RawLandmark]:
        image = frame.frame
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            return {}

        hands: dict[int, dict[str, RawLandmark]] = {}
        for landmark in existing_landmarks.values():
            if landmark.source == self._HAND_SOURCE:
                hands.setdefault(landmark.instance_id, {})[landmark.name] = landmark

        detected: dict[str, RawLandmark] = {}
        active_keys: set[tuple[int, str]] = set()
        for instance_id, hand in hands.items():
            wrist_landmark = hand.get("WRIST")
            if wrist_landmark is None:
                continue
            key = (instance_id, wrist_landmark.side.value)
            active_keys.add(key)
            estimate = self._estimate_hand(image[..., :3], hand, self._tracks.get(key))
            if estimate is None:
                estimate = self._propagate(key, wrist_landmark, image.shape[:2])
            if estimate is None:
                continue

            midpoint, farpoint, width, confidence, propagated = estimate
            if not propagated:
                direction = farpoint - midpoint
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-6:
                    continue
                wrist = self._to_pixel(wrist_landmark, image.shape[:2])
                self._tracks[key] = _Track(
                    wrist=wrist,
                    midpoint=midpoint.copy(),
                    farpoint=farpoint.copy(),
                    direction=direction / norm,
                    width=width,
                    confidence=confidence,
                )

            for name, point in (
                (ForearmLandmarks.FOREARM_MIDPOINT, midpoint),
                (ForearmLandmarks.FOREARM_FAR, farpoint),
            ):
                raw = RawLandmark(
                    name=name.value,
                    source=self.name,
                    instance_id=instance_id,
                    side=wrist_landmark.side,
                    x=float(point[0] / image.shape[1]),
                    y=float(point[1] / image.shape[0]),
                    z=wrist_landmark.z,
                    confidence=confidence,
                )
                landmark_id = raw.generate_landmark_id()
                if landmark_id in detected:
                    raise ValueError(f"Duplicate raw landmark id: {landmark_id}")
                detected[landmark_id] = raw

        # A missing MediaPipe hand cannot be associated safely with an old track.
        for key in self._tracks.keys() - active_keys:
            self._tracks[key].missed_frames += 1
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if track.missed_frames <= self.config.max_tracking_gap
        }
        return detected

    def _estimate_hand(
        self,
        image: NDArray[np.uint8],
        hand: dict[str, RawLandmark],
        previous: _Track | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, bool] | None:
        required = ("WRIST", *self._PALM_NAMES)
        if any(name not in hand for name in required):
            return None
        height, width = image.shape[:2]
        points = {
            name: self._to_pixel(landmark, (height, width))
            for name, landmark in hand.items()
            if np.isfinite(landmark.x) and np.isfinite(landmark.y)
        }
        if any(name not in points for name in required):
            return None

        wrist = points["WRIST"]
        palm_center = np.mean([points[name] for name in self._PALM_NAMES], axis=0)
        palm_vector = palm_center - wrist
        palm_size = float(np.linalg.norm(palm_vector))
        if palm_size < 4.0:
            return None
        axis = -palm_vector / palm_size
        normal = np.array([-axis[1], axis[0]])
        half_width = self.config.roi_half_width_palms * palm_size
        origin_s = -0.10 * palm_size
        crop_width = max(3, int(np.ceil(2 * half_width)) + 1)
        crop_height = max(
            3, int(np.ceil(self.config.roi_length_palms * palm_size - origin_s)) + 1
        )
        local_to_frame = np.array(
            [
                [normal[0], axis[0], wrist[0] - normal[0] * half_width + axis[0] * origin_s],
                [normal[1], axis[1], wrist[1] - normal[1] * half_width + axis[1] * origin_s],
            ],
            dtype=np.float32,
        )
        crop = cv2.warpAffine(
            image,
            local_to_frame,
            (crop_width, crop_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        valid_mask = self._valid_crop_mask(
            width, height, wrist, normal, axis, half_width, origin_s, crop.shape[:2]
        )
        hand_mask = self._local_hand_mask(
            points, wrist, normal, axis, half_width, origin_s, crop.shape[:2], palm_size
        )
        wrist_row = -origin_s
        edges, gradient = self._detect_edges(crop, valid_mask, hand_mask)
        boundaries = self._boundary_candidates(
            edges, gradient, wrist_row, half_width, palm_size
        )
        result = self._best_boundary_pair(
            boundaries, wrist_row, half_width, palm_size, local_to_frame, wrist, previous
        )
        if result is None:
            return None
        midpoint, farpoint, detected_width, evidence_confidence = result

        wrist_confidence = hand["WRIST"].confidence
        confidence = evidence_confidence
        if wrist_confidence is not None and np.isfinite(wrist_confidence):
            confidence *= float(np.clip(wrist_confidence, 0.0, 1.0))

        if previous is not None:
            predicted_mid = previous.midpoint + (wrist - previous.wrist)
            jump = float(np.linalg.norm(midpoint - predicted_mid)) / palm_size
            new_direction = farpoint - midpoint
            new_direction /= max(float(np.linalg.norm(new_direction)), 1e-6)
            angle = np.degrees(
                np.arccos(np.clip(np.dot(new_direction, previous.direction), -1.0, 1.0))
            )
            # Weak evidence cannot cause an abrupt identity/edge switch.
            if (jump > 1.6 or angle > 65.0) and evidence_confidence < 0.72:
                return None
            temporal = np.exp(-0.55 * jump) * np.exp(-angle / 100.0)
            confidence *= float(0.55 + 0.45 * temporal)
            blend = float(np.clip(0.55 + 0.35 * evidence_confidence, 0.55, 0.9))
            midpoint = blend * midpoint + (1 - blend) * predicted_mid
            predicted_far = previous.farpoint + (wrist - previous.wrist)
            farpoint = blend * farpoint + (1 - blend) * predicted_far
            detected_width = blend * detected_width + (1 - blend) * previous.width

        return midpoint, farpoint, detected_width, float(np.clip(confidence, 0, 1)), False

    @staticmethod
    def _to_pixel(landmark: RawLandmark, shape: tuple[int, int]) -> NDArray[np.float64]:
        height, width = shape
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float64)

    @staticmethod
    def _valid_crop_mask(
        width: int,
        height: int,
        wrist: NDArray[np.float64],
        normal: NDArray[np.float64],
        axis: NDArray[np.float64],
        half_width: float,
        origin_s: float,
        crop_shape: tuple[int, int],
    ) -> NDArray[np.uint8]:
        corners = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float64,
        )
        offsets = corners - wrist
        local = np.column_stack(
            (offsets @ normal + half_width, offsets @ axis - origin_s)
        )
        mask = np.zeros(crop_shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(local).astype(np.int32), 1)
        return mask

    def _local_hand_mask(
        self,
        points: dict[str, NDArray[np.float64]],
        wrist: NDArray[np.float64],
        normal: NDArray[np.float64],
        axis: NDArray[np.float64],
        half_width: float,
        origin_s: float,
        crop_shape: tuple[int, int],
        palm_size: float,
    ) -> NDArray[np.uint8]:
        offsets = np.stack(list(points.values())) - wrist
        local = np.column_stack(
            (offsets @ normal + half_width, offsets @ axis - origin_s)
        )
        mask = np.zeros(crop_shape, dtype=np.uint8)
        if len(local) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(local).astype(np.int32)), 255)
            padding = max(1, round(self.config.hand_exclusion_padding_palms * palm_size))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding + 1,) * 2)
            mask = cv2.dilate(mask, kernel)
        return mask

    @staticmethod
    def _detect_edges(
        crop: NDArray[np.uint8],
        valid_mask: NDArray[np.uint8],
        hand_mask: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        median = float(np.median(gray[valid_mask != 0]))
        low = int(np.clip(0.66 * median, 20, 160))
        high = int(np.clip(1.33 * median, low + 20, 240))
        edges = cv2.Canny(gray, low, high, L2gradient=True)
        allowed = (valid_mask != 0) & (hand_mask == 0)
        edges[~allowed] = 0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return edges, cv2.magnitude(gx, gy)

    def _boundary_candidates(
        self,
        edges: NDArray[np.uint8],
        gradient: NDArray[np.float32],
        wrist_row: float,
        center_x: float,
        palm_size: float,
    ) -> list[_Boundary]:
        min_length = max(8, round(self.config.min_boundary_length_palms * palm_size))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(10, round(0.28 * palm_size)),
            minLineLength=min_length,
            maxLineGap=max(4, round(0.28 * palm_size)),
        )
        if lines is None:
            return []
        edge_y, edge_x = np.nonzero(edges)
        edge_points = np.column_stack((edge_x, edge_y)).astype(np.float32)
        candidates: list[_Boundary] = []
        max_angle = np.radians(self.config.max_boundary_angle_degrees)
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            vector = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            length = float(np.linalg.norm(vector))
            if (
                length < min_length
                or np.arctan2(abs(vector[0]), abs(vector[1])) > max_angle
            ):
                continue
            unit = vector / length
            relative = edge_points - np.array([x1, y1], dtype=np.float32)
            along = relative @ unit
            perpendicular = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
            nearby = (along >= -3) & (along <= length + 3) & (perpendicular <= 2.5)
            support = edge_points[nearby]
            if len(support) < max(6, min_length // 3):
                continue
            vx, vy, x0, y0 = (
                float(v)
                for v in cv2.fitLine(support, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            )
            if (
                abs(vy) < 1e-5
                or np.arctan2(abs(vx), abs(vy)) > max_angle
            ):
                continue
            slope = vx / vy
            intercept = x0 - slope * y0
            start_y, end_y = float(np.min(support[:, 1])), float(np.max(support[:, 1]))
            if start_y > wrist_row + 0.9 * palm_size or end_y <= wrist_row:
                continue
            strengths = gradient[support[:, 1].astype(int), support[:, 0].astype(int)]
            strength = float(np.clip(np.mean(strengths) / 180.0, 0.0, 1.0))
            candidate = _Boundary(slope, intercept, start_y, end_y, length, strength)
            # Only keep structures plausibly reachable from either side of the wrist.
            wrist_x = candidate.x_at(wrist_row)
            if abs(wrist_x - center_x) <= 1.55 * palm_size:
                candidates.append(candidate)
        return candidates

    def _best_boundary_pair(
        self,
        boundaries: list[_Boundary],
        wrist_row: float,
        center_x: float,
        palm_size: float,
        local_to_frame: NDArray[np.float32],
        wrist: NDArray[np.float64],
        previous: _Track | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float] | None:
        best: tuple[float, NDArray[np.float64], NDArray[np.float64], float] | None = None
        transform = local_to_frame[:, :2].astype(np.float64)
        translation = local_to_frame[:, 2].astype(np.float64)
        for first_index, first in enumerate(boundaries):
            for second in boundaries[first_index + 1 :]:
                near_y = max(wrist_row + 0.08 * palm_size, first.start_y, second.start_y)
                far_y = min(first.end_y, second.end_y)
                extent = far_y - near_y
                if extent < self.config.min_boundary_length_palms * palm_size:
                    continue
                first_x = first.x_at(near_y)
                second_x = second.x_at(near_y)
                left, right = (first, second) if first_x < second_x else (second, first)
                if not left.x_at(wrist_row) < center_x < right.x_at(wrist_row):
                    continue
                widths = np.array(
                    [right.x_at(y) - left.x_at(y) for y in np.linspace(near_y, far_y, 6)]
                )
                mean_width = float(np.mean(widths))
                if (
                    np.any(widths <= 0)
                    or not 0.22 * palm_size <= mean_width <= 2.6 * palm_size
                    or np.ptp(widths) > 0.65 * mean_width
                ):
                    continue
                angle_difference = abs(np.arctan(left.slope) - np.arctan(right.slope))
                if angle_difference > np.radians(self.config.max_pair_angle_degrees):
                    continue

                rows = np.linspace(near_y, far_y, self.config.cross_section_count)
                centers = np.column_stack(
                    ((np.array([left.x_at(y) for y in rows])
                      + np.array([right.x_at(y) for y in rows])) / 2, rows)
                ).astype(np.float32)
                vx, vy, x0, y0 = (
                    float(v)
                    for v in cv2.fitLine(centers, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
                )
                if abs(vy) < 1e-6:
                    continue

                def local_point(fraction: float) -> NDArray[np.float64]:
                    y = near_y + fraction * extent
                    return np.array([x0 + (y - y0) * vx / vy, y], dtype=np.float64)

                midpoint = transform @ local_point(self.config.midpoint_fraction) + translation
                farpoint = transform @ local_point(self.config.far_fraction) + translation
                parallelism = float(np.exp(-angle_difference / np.radians(18.0)))
                width_consistency = float(np.exp(-np.std(widths) / max(mean_width, 1.0)))
                origin_distance = (
                    abs(left.x_at(wrist_row) - (center_x - mean_width / 2))
                    + abs(right.x_at(wrist_row) - (center_x + mean_width / 2))
                ) / (2 * palm_size)
                origin_score = float(np.exp(-origin_distance))
                length_score = float(np.clip(extent / (2.5 * palm_size), 0, 1))
                edge_score = (left.strength + right.strength) / 2
                score = (
                    0.25 * edge_score
                    + 0.20 * length_score
                    + 0.20 * parallelism
                    + 0.15 * width_consistency
                    + 0.10 * origin_score
                )
                if previous is not None:
                    predicted_mid = previous.midpoint + (wrist - previous.wrist)
                    distance = np.linalg.norm(midpoint - predicted_mid) / palm_size
                    direction = farpoint - midpoint
                    direction /= max(float(np.linalg.norm(direction)), 1e-6)
                    direction_score = (1 + np.clip(np.dot(direction, previous.direction), -1, 1)) / 2
                    width_score = np.exp(-abs(mean_width - previous.width) / max(previous.width, 1.0))
                    temporal_score = np.exp(-distance) * direction_score * width_score
                    score += 0.10 * float(temporal_score)
                if best is None or score > best[0]:
                    best = (score, midpoint, farpoint, mean_width)
        if best is None:
            return None
        score, midpoint, farpoint, width = best
        return midpoint, farpoint, width, float(np.clip(score, 0, 1))

    def _propagate(
        self,
        key: tuple[int, str],
        wrist_landmark: RawLandmark,
        frame_shape: tuple[int, int],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, bool] | None:
        track = self._tracks.get(key)
        if track is None or track.missed_frames >= self.config.max_tracking_gap:
            return None
        wrist = self._to_pixel(wrist_landmark, frame_shape)
        displacement = wrist - track.wrist
        midpoint = track.midpoint + displacement
        farpoint = track.farpoint + displacement
        height, width = frame_shape
        if not (
            0 <= midpoint[0] < width
            and 0 <= midpoint[1] < height
            and 0 <= farpoint[0] < width
            and 0 <= farpoint[1] < height
        ):
            return None
        track.wrist = wrist
        track.midpoint = midpoint
        track.farpoint = farpoint
        track.missed_frames += 1
        track.confidence *= self.config.propagation_decay
        return midpoint, farpoint, track.width, track.confidence, True

    def reset_video(self) -> None:
        self._tracks.clear()

    def close(self) -> None:
        self._tracks.clear()
