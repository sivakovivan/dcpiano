import os
from pathlib import Path

from dcpiano.logger import logger





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