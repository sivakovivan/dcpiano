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
    def _validate_frame(frame: VideoFrame) -> None:
        if frame.metadata.timestamp < 0:
            raise ValueError(
                f"Frame {frame.metadata.index} has a negative timestamp: "
                f"{frame.metadata.timestamp}"
            )
            
        if frame.frame.size == 0:
            raise ValueError(f"Frame {frame.metadata.index} has no pixel data.")
        
        if frame.frame.ndim != 3 or frame.frame.shape[2] != 3:
            raise ValueError(f"Frame {frame.metadata.index} is not a valid color image (expected 3 channels).")
    
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
            
        VideoService._validate_frame(frames[0])

        cap.release()
        return frames