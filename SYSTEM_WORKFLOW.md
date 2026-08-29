# DCPiano System Workflow

DCPiano is currently a batch video-processing pipeline. It reads a piano-performance video, detects hand and forearm landmarks for every frame, reshapes those detections into time-series data, and writes an annotated copy of the video.

## End-to-end flow

```text
main.py
  -> create output directory and process.log
  -> load config/default.yaml and save effective_config.yaml
  -> read video metadata and decode the complete video into memory
  -> optionally calibrate the keyboard from the first frame
  -> process each frame through the ordered detector chain
       HandLandmarkDetector (MediaPipe)
         -> ForearmLandmarkDetector (ArmTrace, using hand landmarks)
  -> save frame-centric landmarks
  -> build and save continuous landmark streams
  -> read the source video again and render landmark overlays
  -> save rendered.mp4
```

The entry point is `main.py`. Its input and output paths are currently hard-coded to:

- Input: `data/input/sample.mp4`
- Output directory: `data/output/sample/`

Run it from the repository root with:

```bash
uv run python main.py
```

## Processing stages

1. **Session setup and configuration**
   
   `DCPPipeline.main()` creates the output directory, starts console/file logging, loads `config/default.yaml`, and copies the resolved settings to `effective_config.yaml` for reproducibility.

2. **Video ingestion**
   
   `VideoService` reads width, height, FPS, and duration with OpenCV. It then decodes the entire video into a list of `VideoFrame` objects. Each frame contains its BGR image, sequential frame index, and timestamp in seconds.

3. **Optional keyboard calibration**
   
   When `skip_calibration` is `false`, the first video frame opens in an interactive OpenCV window. The user selects the keyboard corners in this order: top-left, top-right, bottom-right, bottom-left. Calibration data and a preview are written to the output directory.

   The current default is `skip_calibration: true`, so this stage does not run by default. The homography service can convert normalized image landmarks to keyboard millimetres, but it is not yet connected to the main pipeline.

4. **Ordered landmark detection**
   
   `LandmarkService` sends every frame through the detector list in order. Detector order is significant because later detectors receive landmarks produced earlier in the same frame.

   - `HandLandmarkDetector` uses MediaPipe Hand Landmarker in VIDEO mode, with tracking state retained between consecutive frames. It produces 21 normalized landmarks per detected hand, plus handedness and confidence.
   - `ForearmLandmarkDetector` consumes the wrist and four finger-base landmarks for each detected hand. ArmTrace then estimates `FOREARM_MIDPOINT` and `FOREARM_FAR`. If the required hand points are absent or tracing fails, no forearm points are emitted for that hand and frame.

   Landmark IDs combine detector source, landmark name, and known side, for example `HandLandmarkDetector:WRIST:left`. The detector instance is still stored in each record, but is not part of the ID.

5. **Frame-centric output**
   
   All detections are saved to `landmark_frames.json`. This representation is organized by video frame and contains only landmarks that existed in that frame.

6. **Landmark streams**
   
   `LandmarkStreamService` transposes the frame-centric data into `landmark_streams.json`. Each stable landmark ID gets one entry for every processed frame. Missing detections are explicit entries with `exists: false`, which makes downstream time-series processing easier.

7. **Rendering**
   
   `RenderService` opens the source video again, matches detections by frame index, and draws configured detector connections and joints. Left, right, and unknown-side landmarks use different colors; joint size varies with depth. The result is saved as `rendered.mp4` using the `mp4v` codec.

## Output artifacts

For the default sample run, `data/output/sample/` contains:

| Artifact | Purpose |
| --- | --- |
| `process.log` | Timestamped processing log; overwritten at the start of each run |
| `effective_config.yaml` | Exact configuration used by the run |
| `landmark_frames.json` | Detected landmarks grouped by frame |
| `landmark_streams.json` | Continuous timelines grouped by landmark ID |
| `rendered.mp4` | Source video with hand and forearm overlays |
| `keyboard_calibration.json` | Corner coordinates; present only when calibration is enabled |
| `keyboard_calibration.png` | Annotated calibration preview; present only when calibration is enabled |

## Current operational notes

- The MediaPipe model must exist at `models/hand_landmarker.task`.
- Frames and all detection results are held in memory, so memory use grows with video length and resolution.
- Frame indices must strictly increase and timestamps must not decrease. MediaPipe timestamps are adjusted by at least one millisecond when rounding would create duplicates.
- Detector resources are closed in a `finally` block if landmark processing fails.
- Calibration and homography support exist, but keyboard-space coordinates are not currently included in landmark outputs or rendering.
