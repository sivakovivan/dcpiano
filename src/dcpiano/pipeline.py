import os
from pathlib import Path

from dcpiano.landmark_detectors.hand import HandLandmarkDetector
from dcpiano.logger import logger


from dcpiano.services.config import ConfigService
from dcpiano.services.landmark import LandmarkService
from dcpiano.services.calibration import CalibrationService
from dcpiano.services.video import VideoService


def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


class DCPPipeline:
    @staticmethod
    def main(input_video: Path, output_dir: Path):
        create_output_directory(output_dir)
        
        log_file = logger.start_session(output_dir, overwrite=True)
        
        logger.info(f"Saving logs to: {log_file}")
        logger.info("Starting DCPiano pipeline...")
        logger.info(f"Input video: {input_video}")
        logger.info(f"Output directory: {output_dir}")
        
        config = ConfigService.load_config()
        logger.info(f"Loaded configuration: {config}")
        
        video_metadata = VideoService.get_video_metadata(input_video)
        logger.info(f"Input video metadata: {video_metadata}")
        
        frames = VideoService.get_video_frames(input_video)
        logger.info(f"Extracted {len(frames)} frames from the video.")
        
        calibration_frame = frames[0]
        
        logger.info("Starting manual calibration process...")
        
        keyboard_calibration = CalibrationService.run_manual_keyboard_calibration(calibration_frame, config, output_dir)
    
        logger.info(f"Keyboard calibration result: {keyboard_calibration}")
        
        detectors = [
            HandLandmarkDetector()
        ]
        
        logger.info(f"Initialized landmark detectors: {[detector.name for detector in detectors]}")
        
        landmark_service = LandmarkService(detectors)
        
        try:
            landmark_frames = landmark_service.generate_landmark_frames(frames, output_dir, reset_detectors=False)
        finally:
            landmark_service.close()
            
        logger.info(
            f"Generated {len(landmark_frames)} landmark frames."
        )