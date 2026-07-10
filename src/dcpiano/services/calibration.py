from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from dcpiano.types.calibration import KeyboardCalibration
from dcpiano.types.config import Config
from dcpiano.types.core import Coordinate
from dcpiano.types.video import VideoFrame


class CalibrationService:
    """Interactive four-corner keyboard calibration using an OpenCV window.

    Controls
    --------
    Left click
        Place the next corner.
    Left drag
        Move an already placed corner.
    Right click / Backspace
        Remove the most recently placed corner.
    Enter / Space
        Finish once all four corners have been placed.
    R
        Reset all points.
    Escape / Q
        Cancel calibration and raise ``RuntimeError``.

    The corner order is top-left, top-right, bottom-right, bottom-left.  Pixel
    coordinates are stored in the original frame coordinate system, regardless
    of window size or image letterboxing.
    """

    _WINDOW_NAME: Final[str] = "Piano Keyboard Calibration"
    _CORNER_KEYS: Final[tuple[str, ...]] = (
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    )
    _CORNER_LABELS: Final[tuple[str, ...]] = (
        "Top left",
        "Top right",
        "Bottom right",
        "Bottom left",
    )

    @staticmethod
    def run_manual_keyboard_calibration(
        calibration_frame: VideoFrame,
        config: Config,
        output_dir: Path,
    ) -> KeyboardCalibration:
        """Open an interactive calibration UI and return the selected corners.

        The physical keyboard coordinate system follows the Cartesian convention:
        the origin is the keyboard's bottom-left corner, +x runs toward the
        bottom-right corner, and +y runs toward the top-left corner. Therefore the
        returned millimetre coordinates are ``(0, height)``, ``(width, height)``,
        ``(width, 0)``, and ``(0, 0)`` for TL, TR, BR, and BL respectively.
        """
        image = CalibrationService._normalise_frame(calibration_frame.frame)
        image_height, image_width = image.shape[:2]
        CalibrationService._validate_inputs(image_width, image_height, config)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state: dict[str, object] = {
            "points": [],  # points in original image pixels: list[tuple[float, float]]
            "drag_index": None,
            "hover": None,
            "canvas_size": (0, 0),
            "image_rect": (0, 0, image_width, image_height),
            "panel_width": 350,
            "panel_height": 245,
            "panel_dragging": False,
        }

        cv2.namedWindow(CalibrationService._WINDOW_NAME, cv2.WINDOW_NORMAL)
        initial_width, initial_height = CalibrationService._comfortable_window_size(
            image_width, image_height
        )
        cv2.resizeWindow(CalibrationService._WINDOW_NAME, initial_width, initial_height)

        def display_to_image(x: int, y: int) -> tuple[float, float] | None:
            rx, ry, rw, rh = state["image_rect"]  # type: ignore[misc]
            if rw <= 0 or rh <= 0 or not (rx <= x <= rx + rw and ry <= y <= ry + rh):
                return None
            px = (x - rx) * image_width / rw
            py = (y - ry) * image_height / rh
            return (
                float(np.clip(px, 0.0, image_width - 1.0)),
                float(np.clip(py, 0.0, image_height - 1.0)),
            )

        def image_to_display(point: tuple[float, float]) -> tuple[int, int]:
            rx, ry, rw, rh = state["image_rect"]  # type: ignore[misc]
            px, py = point
            return (
                int(round(rx + px * rw / image_width)),
                int(round(ry + py * rh / image_height)),
            )

        def panel_rect() -> tuple[int, int, int, int]:
            canvas_width, _ = state["canvas_size"]  # type: ignore[misc]
            margin = 18
            panel_width = min(int(state["panel_width"]), max(280, canvas_width - 2 * margin))
            panel_height = int(state["panel_height"])
            return canvas_width - margin - panel_width, margin, panel_width, panel_height

        def hit_test_point(x: int, y: int) -> int | None:
            points: list[tuple[float, float]] = state["points"]  # type: ignore[assignment]
            radius = max(12, int(round(min(state["canvas_size"]) * 0.018)))  # type: ignore[arg-type]
            for index in range(len(points) - 1, -1, -1):
                dx, dy = image_to_display(points[index])
                if (x - dx) ** 2 + (y - dy) ** 2 <= radius**2:
                    return index
            return None

        def mouse_callback(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            points: list[tuple[float, float]] = state["points"]  # type: ignore[assignment]
            state["hover"] = (x, y)

            px, py, pw, ph = panel_rect()
            handle_size = 22
            handle_hit = (
                px - handle_size <= x <= px + handle_size
                and py + ph - handle_size <= y <= py + ph + handle_size
            )

            if event == cv2.EVENT_LBUTTONDOWN:
                if handle_hit:
                    state["panel_dragging"] = True
                    return

                hit = hit_test_point(x, y)
                if hit is not None:
                    state["drag_index"] = hit
                    return

                image_point = display_to_image(x, y)
                if image_point is not None and len(points) < 4:
                    points.append(image_point)
                    state["drag_index"] = len(points) - 1

            elif event == cv2.EVENT_MOUSEMOVE:
                if bool(state["panel_dragging"]):
                    canvas_width, canvas_height = state["canvas_size"]  # type: ignore[misc]
                    right_edge = canvas_width - 18
                    top_edge = 18
                    new_width = right_edge - x
                    new_height = y - top_edge
                    state["panel_width"] = int(np.clip(new_width, 280, max(280, canvas_width - 36)))
                    state["panel_height"] = int(np.clip(new_height, 205, max(205, canvas_height - 36)))
                    return

                drag_index = state["drag_index"]
                if isinstance(drag_index, int):
                    image_point = display_to_image(x, y)
                    if image_point is not None:
                        points[drag_index] = image_point

            elif event == cv2.EVENT_LBUTTONUP:
                state["drag_index"] = None
                state["panel_dragging"] = False

            elif event == cv2.EVENT_RBUTTONDOWN:
                state["drag_index"] = None
                if points:
                    points.pop()

        cv2.setMouseCallback(CalibrationService._WINDOW_NAME, mouse_callback)

        try:
            while True:
                if cv2.getWindowProperty(
                    CalibrationService._WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    raise RuntimeError("Keyboard calibration was cancelled: window closed.")

                try:
                    _wx, _wy, canvas_width, canvas_height = cv2.getWindowImageRect(
                        CalibrationService._WINDOW_NAME
                    )
                except (cv2.error, AttributeError):
                    canvas_width, canvas_height = initial_width, initial_height

                canvas_width = max(480, int(canvas_width))
                canvas_height = max(360, int(canvas_height))
                state["canvas_size"] = (canvas_width, canvas_height)

                canvas, image_rect = CalibrationService._build_canvas(
                    image, canvas_width, canvas_height
                )
                state["image_rect"] = image_rect
                CalibrationService._draw_calibration_overlay(
                    canvas=canvas,
                    state=state,
                    image_size=(image_width, image_height),
                    image_to_display=image_to_display,
                    panel_rect=panel_rect,
                )
                cv2.imshow(CalibrationService._WINDOW_NAME, canvas)

                key = cv2.waitKeyEx(16)
                if key in (27, ord("q"), ord("Q")):
                    raise RuntimeError("Keyboard calibration was cancelled by the user.")
                if key in (8, 127):  # Backspace / Delete
                    points = state["points"]  # type: ignore[assignment]
                    if points:
                        points.pop()
                elif key in (ord("r"), ord("R")):
                    state["points"] = []
                    state["drag_index"] = None
                elif key in (10, 13, 32):  # Enter / Return / Space
                    points = state["points"]  # type: ignore[assignment]
                    if len(points) == 4:
                        if not CalibrationService._is_valid_quadrilateral(points):
                            continue
                        result = CalibrationService._make_result(points, config)
                        CalibrationService._save_outputs(
                            output_dir=output_dir,
                            image=image,
                            points=points,
                            result=result,
                            frame=calibration_frame,
                        )
                        return result
        finally:
            try:
                cv2.destroyWindow(CalibrationService._WINDOW_NAME)
            except cv2.error:
                pass

    @staticmethod
    def _normalise_frame(frame: np.ndarray) -> np.ndarray:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("calibration_frame.frame must be a non-empty numpy array.")

        if frame.ndim == 2:
            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            image = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            image = frame.copy()
        elif frame.ndim == 3 and frame.shape[2] == 4:
            image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(
                "calibration_frame.frame must be grayscale, BGR, or BGRA image data."
            )

        if image.dtype != np.uint8:
            finite = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            if np.issubdtype(finite.dtype, np.floating) and float(np.max(finite)) <= 1.0:
                finite = finite * 255.0
            image = np.clip(finite, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    @staticmethod
    def _validate_inputs(image_width: int, image_height: int, config: Config) -> None:
        if image_width < 2 or image_height < 2:
            raise ValueError("The calibration frame is too small to calibrate.")
        if not math.isfinite(float(config.keyboard_width_mm)) or config.keyboard_width_mm <= 0:
            raise ValueError("config.keyboard_width_mm must be a positive finite number.")
        if not math.isfinite(float(config.keyboard_height_mm)) or config.keyboard_height_mm <= 0:
            raise ValueError("config.keyboard_height_mm must be a positive finite number.")

    @staticmethod
    def _comfortable_window_size(image_width: int, image_height: int) -> tuple[int, int]:
        # A good default for laptop/desktop displays without requiring a GUI toolkit
        # solely to query monitor dimensions. The window remains fully resizable.
        max_width, max_height = 1440, 900
        min_width, min_height = 800, 560
        scale = min(max_width / image_width, max_height / image_height, 1.25)
        width = int(np.clip(round(image_width * scale), min_width, max_width))
        height = int(np.clip(round(image_height * scale), min_height, max_height))
        return width, height

    @staticmethod
    def _build_canvas(
        image: np.ndarray, canvas_width: int, canvas_height: int
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        image_height, image_width = image.shape[:2]
        scale = min(canvas_width / image_width, canvas_height / image_height)
        render_width = max(1, int(round(image_width * scale)))
        render_height = max(1, int(round(image_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        rendered = cv2.resize(image, (render_width, render_height), interpolation=interpolation)

        canvas = np.full((canvas_height, canvas_width, 3), 26, dtype=np.uint8)
        x0 = (canvas_width - render_width) // 2
        y0 = (canvas_height - render_height) // 2
        canvas[y0 : y0 + render_height, x0 : x0 + render_width] = rendered
        return canvas, (x0, y0, render_width, render_height)

    @staticmethod
    def _draw_calibration_overlay(
        *,
        canvas: np.ndarray,
        state: dict[str, object],
        image_size: tuple[int, int],
        image_to_display,
        panel_rect,
    ) -> None:
        points: list[tuple[float, float]] = state["points"]  # type: ignore[assignment]
        canvas_height, canvas_width = canvas.shape[:2]
        scale = min(canvas_width, canvas_height) / 900.0
        marker_radius = max(8, int(round(11 * scale)))
        line_width = max(2, int(round(3 * scale)))

        display_points = [image_to_display(point) for point in points]
        if len(display_points) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(display_points, dtype=np.int32)],
                isClosed=len(display_points) == 4,
                color=(50, 220, 255),
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )

        if len(display_points) == 4:
            fill = canvas.copy()
            cv2.fillPoly(fill, [np.asarray(display_points, dtype=np.int32)], (50, 220, 255))
            cv2.addWeighted(fill, 0.12, canvas, 0.88, 0.0, canvas)

        for index, (x, y) in enumerate(display_points):
            active = index == len(points) - 1 and len(points) < 4
            outer = marker_radius + (4 if active else 2)
            cv2.circle(canvas, (x, y), outer, (16, 16, 16), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), marker_radius, (50, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), marker_radius, (255, 255, 255), 2, cv2.LINE_AA)
            label = str(index + 1)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
            cv2.putText(
                canvas,
                label,
                (x - text_size[0] // 2, y + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        CalibrationService._draw_panel(
            canvas=canvas,
            panel=panel_rect(),
            points=points,
            image_size=image_size,
        )

    @staticmethod
    def _draw_panel(
        *,
        canvas: np.ndarray,
        panel: tuple[int, int, int, int],
        points: list[tuple[float, float]],
        image_size: tuple[int, int],
    ) -> None:
        x, y, width, height = panel
        x = max(0, x)
        y = max(0, y)
        width = min(width, canvas.shape[1] - x)
        height = min(height, canvas.shape[0] - y)
        if width <= 0 or height <= 0:
            return

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0, canvas)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (215, 225, 235), 1, cv2.LINE_AA)

        pad = 18
        cursor_y = y + 29
        cv2.putText(
            canvas,
            "Keyboard calibration",
            (x + pad, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cursor_y += 26
        next_index = len(points)
        status = (
            f"Next: {CalibrationService._CORNER_LABELS[next_index]}"
            if next_index < 4
            else "All corners placed - press Enter to finish"
        )
        cv2.putText(
            canvas,
            status,
            (x + pad, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 235, 255) if next_index < 4 else (130, 245, 165),
            1,
            cv2.LINE_AA,
        )

        diagram_top = cursor_y + 18
        diagram_height = max(58, min(94, height - 145))
        diagram_left = x + pad
        diagram_right = x + width - pad
        diagram_bottom = diagram_top + diagram_height
        cv2.rectangle(
            canvas,
            (diagram_left, diagram_top),
            (diagram_right, diagram_bottom),
            (205, 215, 225),
            2,
            cv2.LINE_AA,
        )

        corners = (
            (diagram_left, diagram_top),
            (diagram_right, diagram_top),
            (diagram_right, diagram_bottom),
            (diagram_left, diagram_bottom),
        )
        short_labels = ("TL", "TR", "BR", "BL")
        offsets = ((7, 17), (-31, 17), (-34, -8), (7, -8))
        for index, ((cx, cy), label, (ox, oy)) in enumerate(zip(corners, short_labels, offsets)):
            completed = index < len(points)
            is_next = index == len(points)
            color = (130, 245, 165) if completed else (210, 220, 230)
            cv2.circle(canvas, (cx, cy), 6, color, -1, cv2.LINE_AA)
            if is_next:
                cv2.circle(canvas, (cx, cy), 13, (50, 220, 255), 3, cv2.LINE_AA)
            cv2.putText(
                canvas,
                label,
                (cx + ox, cy + oy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        text_y = diagram_bottom + 25
        remaining_height = y + height - text_y - 12
        help_lines = [
            "Click to place  •  Drag to refine",
            "Right-click/Backspace undo  •  R reset",
        ]
        if remaining_height >= 38:
            help_lines.append("Enter finish  •  Esc cancel")
        for line in help_lines:
            if text_y > y + height - 10:
                break
            cv2.putText(
                canvas,
                line,
                (x + pad, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (215, 222, 230),
                1,
                cv2.LINE_AA,
            )
            text_y += 20

        # Bottom-left resize handle requested by the UI specification.
        handle = 17
        cv2.line(canvas, (x, y + height), (x + handle, y + height), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (x, y + height), (x, y + height - handle), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (x + 5, y + height - 5), (x + handle, y + height - handle), (170, 180, 190), 1, cv2.LINE_AA)

    @staticmethod
    def _is_valid_quadrilateral(points: list[tuple[float, float]]) -> bool:
        if len(points) != 4:
            return False
        contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        area = abs(float(cv2.contourArea(contour)))
        if area < 25.0 or not cv2.isContourConvex(contour.astype(np.int32)):
            return False

        # Prevent zero-length edges and obvious duplicate clicks.
        for index in range(4):
            p1 = np.asarray(points[index], dtype=np.float64)
            p2 = np.asarray(points[(index + 1) % 4], dtype=np.float64)
            if float(np.linalg.norm(p2 - p1)) < 3.0:
                return False
        return True

    @staticmethod
    def _make_result(
        points: list[tuple[float, float]], config: Config
    ) -> KeyboardCalibration:
        top_left, top_right, bottom_right, bottom_left = points

        width_mm = float(config.keyboard_width_mm)
        height_mm = float(config.keyboard_height_mm)

        # Use exact physical corner coordinates instead of transforming the source
        # corners back through a homography. This avoids tiny floating-point noise
        # such as 1.3350574172652646e-13 in the saved JSON.
        tl_mm = (0.0, height_mm)
        tr_mm = (width_mm, height_mm)
        br_mm = (width_mm, 0.0)
        bl_mm = (0.0, 0.0)

        def clean(value: float, decimals: int = 6) -> float:
            rounded = round(float(value), decimals)
            return 0.0 if abs(rounded) < 10 ** (-decimals) else rounded

        def coordinate(point: tuple[float, float]) -> Coordinate:
            return Coordinate(x=clean(point[0]), y=clean(point[1]))

        return KeyboardCalibration(
            top_right_corner_px=coordinate(top_right),
            bottom_right_corner_px=coordinate(bottom_right),
            bottom_left_corner_px=coordinate(bottom_left),
            top_left_corner_px=coordinate(top_left),
            top_right_corner_mm=coordinate(tr_mm),
            bottom_right_corner_mm=coordinate(br_mm),
            bottom_left_corner_mm=coordinate(bl_mm),
            top_left_corner_mm=coordinate(tl_mm),
        )

    @staticmethod
    def _save_outputs(
        *,
        output_dir: Path,
        image: np.ndarray,
        points: list[tuple[float, float]],
        result: KeyboardCalibration,
        frame: VideoFrame,
    ) -> None:
        annotated = image.copy()
        integer_points = np.rint(np.asarray(points)).astype(np.int32)
        cv2.polylines(
            annotated,
            [integer_points],
            isClosed=True,
            color=(50, 220, 255),
            thickness=max(2, round(min(image.shape[:2]) / 350)),
            lineType=cv2.LINE_AA,
        )
        for index, point in enumerate(integer_points):
            cv2.circle(annotated, tuple(point), 8, (50, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                str(index + 1),
                (int(point[0]) + 10, int(point[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(output_dir / "keyboard_calibration.png"), annotated)
        payload = {
            "frame_index": int(frame.metadata.index),
            "frame_timestamp": float(frame.metadata.timestamp),
            "calibration": asdict(result),
        }
        (output_dir / "keyboard_calibration.json").write_text(
            json.dumps(payload, indent=4), encoding="utf-8"
        )