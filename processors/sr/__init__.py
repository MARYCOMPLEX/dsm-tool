# processors/sr/__init__.py
from pathlib import Path
from typing import Dict, Optional, Callable

from processors.base import Processor
from .sr.run import run_sr
import os


class SuperResolutionProcessor(Processor):
    @property
    def name(self) -> str:
        return "Super Resolution"

    @property
    def description(self) -> str:
        return "Super Resolution"

    def process(
            self,
            image1_path: str,
            image2_path: str,
            rpc1: Dict[str, str],
            rpc2: Dict[str, str],
            output_dir: str,
            progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        # 使用 progress_callback 报告进度（可选）
        if progress_callback:
            progress_callback(5)

        # TODO: 读取 image1/image2，根据output_dir和image1/image2的文件名，生成输出文件名
        # 
        output_path1 = os.path.join(output_dir, Path(image1_path).stem + "_sr.tif")
        output_path2 = os.path.join(output_dir, Path(image2_path).stem + "_sr.tif")
        run_sr(
            left_input=image1_path,
            left_output=output_path1,
            right_input=image2_path,
            right_output=output_path2,
            RPC=True
        )
        if progress_callback:
            progress_callback(100)
        return str(output_path1)
