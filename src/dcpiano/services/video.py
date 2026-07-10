import cv2
from pathlib import Path
from dcpiano.types.video import FrameMetadata, VideoMetadata, VideoFrame


class VideoService:
    @staticmethod
    def get_video_metadata(path: Path) -> VideoMetadata:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return VideoMetadata(width=width, height=height, path=path, fps=fps, duration=duration)
    
    @staticmethod
    def get_video_frames(path: Path) -> list[VideoFrame]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        frames = []
        index = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = index / fps if fps > 0 else 0
            frames.append(VideoFrame(metadata=FrameMetadata(index=index, timestamp=timestamp), frame=frame))
            index += 1

        cap.release()
        return frames