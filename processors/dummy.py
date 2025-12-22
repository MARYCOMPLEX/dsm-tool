"""
Dummy Processor Module

Example implementation of a Processor for testing purposes.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from processors.base import Processor


class DummyProcessor(Processor):
    """
    Dummy processor that simply copies the first image as output.

    This is a placeholder implementation for testing the framework.
    """

    @property
    def name(self) -> str:
        """Return the display name."""
        return "Dummy Processor (Copy Image)"

    @property
    def description(self) -> str:
        """Return the description."""
        return "Copies the first input image as output (for testing purposes)"

    def process(
        self,
        image1_path: str,
        image2_path: str,
        rpc1: Dict[str, str],
        rpc2: Dict[str, str],
        output_dir: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """
        Process images by copying the first image.

        Args:
            image1_path: Path to the first image file
            image2_path: Path to the second image file
            rpc1: RPC metadata dictionary for the first image
            rpc2: RPC metadata dictionary for the second image
            output_dir: Directory where output files should be saved
            progress_callback: Optional progress reporting callback

        Returns:
            Path to the output file
        """
        if not self.validate_inputs(image1_path, image2_path, rpc1, rpc2, output_dir):
            raise ValueError("Invalid inputs for processing")

        def report(pct: int) -> None:
            if progress_callback is not None:
                progress_callback(pct)

        report(5)
        time.sleep(0.1)

        # Create output filename
        input_name = Path(image1_path).stem
        output_path = os.path.join(output_dir, f"{input_name}_output.tif")

        report(40)
        time.sleep(0.1)

        # Copy the first image
        shutil.copy2(image1_path, output_path)

        report(90)
        time.sleep(0.05)

        return output_path

