# DCPiano

DCPiano processes a piano-performance video with MediaPipe. It guides you through keyboard calibration, detects hand landmarks in each frame, and produces calibration files, landmark data, a processing log, and a video with the detected hands overlaid.

## Installation

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.13+, and a desktop environment for the calibration window.

```bash
uv sync
```

Download a MediaPipe Hand Landmarker model and save it as `models/hand_landmarker.task`.

## Running

1. Put the source video at `data/input/sample.mp4`.
2. From the project root, run:

   ```bash
   uv run python main.py
   ```

3. In the calibration window, select the keyboard corners in this order: top-left, top-right, bottom-right, bottom-left. Press Enter to continue.

Results are written to `data/output/sample/`, including `rendered.mp4`, `landmark_frames.json`, `kinematic_frames.json`, keyboard calibration files, and `process.log`. `kinematic_frames.json` contains the registered position, derivative, axis, plane, joint-angle, and keyboard-height metrics for each frame; unavailable values are explicit `null` entries. To use different paths, edit the `input_video` and `output_dir` values in `main.py`.

Pipeline behavior is configured in `config/default.yaml`. Set `tracking_only: true` to run landmark tracking without calibration or kinematic calculation. Set `create_rendered_video: false` to skip creating `rendered.mp4`; this setting also applies in tracking-only mode. Keyboard dimensions can be changed in the same file.
