from pathlib import Path

from dcpiano.pipeline import DCPPipeline

if __name__ == "__main__":
    DCPPipeline.main(
        input_video=Path("data") / "input" / "sample.mp4",
        output_dir=Path("data") / "output" / "sample",
    )