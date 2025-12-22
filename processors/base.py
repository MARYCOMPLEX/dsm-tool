"""
Processor Base Module

Defines the abstract base class for all image processing algorithms.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional


class Processor(ABC):
    """
    Abstract base class for image processing algorithms.

    All processors must inherit from this class and implement the process method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Display name of the processor.

        Returns:
            Human-readable name for the processor
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Description of what the processor does.

        Returns:
            Human-readable description
        """
        pass

    @abstractmethod
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
        Process two images with their RPC metadata.

        Args:
            image1_path: Path to the first image file
            image2_path: Path to the second image file
            rpc1: RPC metadata dictionary for the first image
            rpc2: RPC metadata dictionary for the second image
            output_dir: Directory where output files should be saved
            progress_callback: Optional callback used by long-running
                algorithms to report progress in percent [0-100].

        Returns:
            Path to the output file

        Raises:
            Exception: If processing fails
        """
        pass

    def validate_inputs(
        self,
        image1_path: str,
        image2_path: str,
        rpc1: Dict[str, str],
        rpc2: Dict[str, str],
        output_dir: str,
    ) -> bool:
        """
        Validate inputs before processing.

        Args:
            image1_path: Path to the first image file
            image2_path: Path to the second image file
            rpc1: RPC metadata dictionary for the first image
            rpc2: RPC metadata dictionary for the second image
            output_dir: Directory where output files should be saved

        Returns:
            True if inputs are valid, False otherwise
        """
        import os

        return (
            os.path.exists(image1_path)
            and os.path.exists(image2_path)
            and os.path.isdir(output_dir)
            and len(rpc1) > 0
            and len(rpc2) > 0
        )

