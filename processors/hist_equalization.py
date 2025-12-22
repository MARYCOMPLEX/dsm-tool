"""
Histogram Equalization Processor

Example processor that applies per-band histogram equalization
to the first input image and writes a new GeoTIFF as output.
"""

import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from osgeo import gdal

from processors.base import Processor


class HistogramEqualizationProcessor(Processor):
    """
    Apply histogram equalization to all bands of the first image.

    This processor is intended as a simple example of how to plug
    new algorithms into the framework.
    """

    @property
    def name(self) -> str:
        """Display name shown in the algorithm combo box."""
        return "Histogram Equalization (Image 1)"

    @property
    def description(self) -> str:
        """Short description of the algorithm."""
        return "Applies per-band histogram equalization to the first image and saves a new GeoTIFF."

    def process(
        self,
        image1_path: str,
        image2_path: str,  # unused but kept for interface compatibility
        rpc1: Dict[str, str],
        rpc2: Dict[str, str],  # unused but kept for interface compatibility
        output_dir: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """
        Run histogram equalization on the first image.

        Args:
            image1_path: Path to the first image file
            image2_path: Path to the second image file (unused)
            rpc1: RPC metadata for the first image
            rpc2: RPC metadata for the second image (unused)
            output_dir: Directory where output files should be saved

        Returns:
            Path to the output GeoTIFF file.
        """
        if not self.validate_inputs(image1_path, image2_path, rpc1, rpc2, output_dir):
            raise ValueError("Invalid inputs for histogram equalization processor.")

        def report(pct: int) -> None:
            """Helper to safely report progress."""
            if progress_callback is not None:
                try:
                    progress_callback(int(pct))
                except Exception:
                    # Never let UI errors break the algorithm
                    pass

        gdal.UseExceptions()

        report(5)

        ds = gdal.Open(image1_path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Failed to open image: {image1_path}")

        try:
            width = ds.RasterXSize
            height = ds.RasterYSize
            bands = ds.RasterCount

            # Simulate some startup delay for debugging
            time.sleep(0.2)
            report(10)

            driver = gdal.GetDriverByName("GTiff")
            if driver is None:
                raise RuntimeError("GDAL GTiff driver not available.")

            input_stem = Path(image1_path).stem
            output_path = os.path.join(output_dir, f"{input_stem}_histeq.tif")

            # Create output dataset: 8-bit, same size / band count
            out_ds = driver.Create(
                output_path,
                width,
                height,
                bands,
                gdal.GDT_Byte,
            )
            if out_ds is None:
                raise RuntimeError(f"Failed to create output dataset: {output_path}")

            report(20)

            # Copy georeferencing and RPC metadata
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            if geotransform:
                out_ds.SetGeoTransform(geotransform)
            if projection:
                out_ds.SetProjection(projection)

            rpc_md = ds.GetMetadata("RPC")
            if rpc_md:
                out_ds.SetMetadata(rpc_md, "RPC")

            # Process each band independently
            for band_idx in range(1, bands + 1):
                # Simulate per-band work delay for debugging
                time.sleep(0.3)

                in_band = ds.GetRasterBand(band_idx)
                if in_band is None:
                    continue

                data = in_band.ReadAsArray()
                if data is None:
                    continue

                # Ensure float for processing
                data = data.astype(np.float32)

                # Simple contrast stretching: clip 2% and 98% percentiles
                p2, p98 = np.percentile(data, [2, 98])
                if p98 > p2:
                    data = np.clip(data, p2, p98)
                    # Normalize to [0, 255]
                    norm = ((data - p2) / (p98 - p2) * 255.0).astype(np.uint8)
                else:
                    norm = np.zeros_like(data, dtype=np.uint8)

                # Histogram equalization on 0-255 image
                hist, _ = np.histogram(norm.flatten(), bins=256, range=(0, 255))
                cdf = hist.cumsum()

                # Mask zero values to avoid division by zero
                cdf_masked = np.ma.masked_equal(cdf, 0)
                if cdf_masked.max() == cdf_masked.min():
                    # Flat histogram, nothing to equalize
                    equalized = norm
                else:
                    cdf_scaled = (
                        (cdf_masked - cdf_masked.min())
                        * 255
                        / (cdf_masked.max() - cdf_masked.min())
                    )
                    cdf_final = np.ma.filled(cdf_scaled, 0).astype("uint8")
                    equalized = cdf_final[norm]

                out_band = out_ds.GetRasterBand(band_idx)
                out_band.WriteArray(equalized)

                # Only set NoData when it is defined; GDAL expects a double
                nodata = in_band.GetNoDataValue()
                if nodata is not None:
                    out_band.SetNoDataValue(float(nodata))

            out_ds.FlushCache()
            out_ds = None

            report(100)

            return output_path

        finally:
            ds = None


