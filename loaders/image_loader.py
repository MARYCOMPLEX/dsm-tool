"""
Image Loader Module

Handles loading satellite imagery and RPC metadata using GDAL.
Directory layout assumption (per user requirement):
    - Exactly 4 files in the folder are relevant:
        * 2 GeoTIFF images:  *.tif / *.tiff
        * 2 RPC files:      same basename as each image, with .RPB / .RPC / .rpb / .rpc
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal
from PyQt6.QtGui import QImage
from PyQt6.QtCore import QSize


# Suppress GDAL warnings
gdal.UseExceptions()


def find_image_files(directory: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Find image files and their corresponding RPC files in a directory.
    
    Args:
        directory: Path to the directory containing images
        
    Returns:
        Tuple of (image_paths, rpc_mapping) where:
        - image_paths: List of image file paths
        - rpc_mapping: Dict mapping image path to RPC file path
        
    Raises:
        ValueError: If not exactly 2 images found or RPC files missing
    """
    directory_path = Path(directory)
    image_extensions = {".tif", ".tiff", ".TIF", ".TIFF"}

    # Find all image files (only by extension)
    image_files = [
        p for p in directory_path.iterdir() if p.suffix in image_extensions
    ]
    
    if len(image_files) != 2:
        raise ValueError(
            f"Expected exactly 2 image files, found {len(image_files)}. "
            f"Please ensure the directory contains exactly 2 GeoTIFF files."
        )
    
    # Sort for deterministic order
    image_files = sorted(image_files)
    image_paths = [str(f) for f in image_files]
    rpc_mapping = {}
    
    # Find corresponding RPC files: must be same stem with .rpb / .rpc (any case)
    for img_path in image_paths:
        img_path_obj = Path(img_path)
        base_name = img_path_obj.stem
        
        # Try different RPC file naming conventions (same basename)
        rpc_candidates = [
            img_path_obj.parent / f"{base_name}.rpb",
            img_path_obj.parent / f"{base_name}.rpc",
            img_path_obj.parent / f"{base_name}.RPB",
            img_path_obj.parent / f"{base_name}.RPC",
        ]
        
        rpc_found = None
        for rpc_candidate in rpc_candidates:
            if rpc_candidate.exists():
                rpc_found = str(rpc_candidate)
                break
        
        if rpc_found is None:
            raise ValueError(
                f"RPC file not found for image: {img_path}\n"
                f"Expected one of: {[str(c) for c in rpc_candidates]}"
            )
        
        rpc_mapping[img_path] = rpc_found
    
    return image_paths, rpc_mapping


def load_rpc_metadata(image_path: str, rpc_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load RPC metadata from image file or external RPC file.
    
    Args:
        image_path: Path to the image file
        rpc_file: Optional path to external RPC file (if None, checks embedded)
        
    Returns:
        Dictionary containing RPC metadata
        
    Raises:
        RuntimeError: If RPC metadata cannot be loaded
    """
    # GDAL 会根据文件名自动关联外部 RPB/RPC 文件，这里只需打开图像
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {image_path}")
    
    try:
        # 直接从 GDAL 元数据中读取 RPC；如果有外部 RPB/RPC，GDAL 会自动使用
        rpc_metadata = ds.GetMetadata("RPC")

        if not rpc_metadata:
            raise RuntimeError(
                f"No RPC metadata found for image: {image_path}\n"
                f"RPC file (expected): {rpc_file}"
            )

        return dict(rpc_metadata)
    
    finally:
        ds = None


def get_image_info(image_path: str) -> Dict:
    """
    Get basic information about the image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information (width, height, bands, etc.)
    """
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {image_path}")
    
    try:
        info = {
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'bands': ds.RasterCount,
            'projection': ds.GetProjection(),
            'geotransform': ds.GetGeoTransform(),
        }
        return info
    finally:
        ds = None


def load_image_as_qimage(
    image_path: str,
    max_size: Optional[QSize] = None,
    rgb_bands: Optional[Tuple[int, int, int]] = None
) -> QImage:
    """
    Load image as QImage for display, using overviews if available.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size for the loaded image (for performance)
        rgb_bands: Tuple of (R, G, B) band indices (1-based). If None, auto-detect.
        
    Returns:
        QImage object ready for display
    """
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {image_path}")
    
    try:
        num_bands = ds.RasterCount
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        # Determine which bands to use for RGB
        if rgb_bands is None:
            if num_bands >= 3:
                # Default: use bands 1, 2, 3 (or 4, 3, 2 for common satellite)
                rgb_bands = (1, 2, 3) if num_bands == 3 else (4, 3, 2)
            else:
                # Grayscale: use first band for all channels
                rgb_bands = (1, 1, 1)
        
        # Check if we need to use overviews
        use_overview = False
        overview_level = 0
        
        if max_size:
            scale_x = width / max_size.width()
            scale_y = height / max_size.height()
            max_scale = max(scale_x, scale_y)
            
            if max_scale > 1.0:
                # Find appropriate overview level
                band = ds.GetRasterBand(1)
                num_overviews = band.GetOverviewCount()
                
                if num_overviews > 0:
                    for i in range(num_overviews):
                        ovr_band = band.GetOverview(i)
                        ovr_width = ovr_band.XSize
                        ovr_height = ovr_band.YSize
                        ovr_scale = max(width / ovr_width, height / ovr_height)
                        
                        if ovr_scale <= max_scale * 1.5:  # Use overview if close enough
                            use_overview = True
                            overview_level = i
                            width = ovr_width
                            height = ovr_height
                            break
        
        # Read RGB bands
        r_band_idx, g_band_idx, b_band_idx = rgb_bands
        
        def read_band(band_idx: int) -> np.ndarray:
            """Read a band, using overview if needed."""
            band = ds.GetRasterBand(band_idx)
            if use_overview:
                band = band.GetOverview(overview_level)
            data = band.ReadAsArray(0, 0, width, height)
            return data
        
        r_data = read_band(r_band_idx)
        g_data = read_band(g_band_idx)
        b_data = read_band(b_band_idx)
        
        # Normalize to 0-255 range with automatic contrast stretching
        def normalize_band(data: np.ndarray) -> np.ndarray:
            """Normalize band data to 0-255 with contrast stretching."""
            if data.dtype != np.uint8:
                # Remove outliers (2% and 98% percentiles)
                p2, p98 = np.percentile(data, [2, 98])
                data = np.clip(data, p2, p98)
                # Normalize to 0-255
                if p98 > p2:
                    data = ((data - p2) / (p98 - p2) * 255).astype(np.uint8)
                else:
                    data = np.zeros_like(data, dtype=np.uint8)
            return data
        
        r_norm = normalize_band(r_data)
        g_norm = normalize_band(g_data)
        b_norm = normalize_band(b_data)
        
        # Stack into RGB array
        rgb_array = np.dstack([r_norm, g_norm, b_norm])
        
        # Convert to QImage
        height, width, channels = rgb_array.shape
        bytes_per_line = channels * width
        qimage = QImage(
            rgb_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Make a copy since the numpy array might be deallocated
        return qimage.copy()
    
    finally:
        ds = None


def load_image_as_array(
    image_path: str,
    rgb_bands: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Load full resolution image as numpy array.
    
    Args:
        image_path: Path to the image file
        rgb_bands: Tuple of (R, G, B) band indices (1-based)
        
    Returns:
        numpy array of shape (height, width, 3) with RGB data
    """
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {image_path}")
    
    try:
        num_bands = ds.RasterCount
        
        if rgb_bands is None:
            if num_bands >= 3:
                rgb_bands = (1, 2, 3) if num_bands == 3 else (4, 3, 2)
            else:
                rgb_bands = (1, 1, 1)
        
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        r_data = ds.GetRasterBand(rgb_bands[0]).ReadAsArray()
        g_data = ds.GetRasterBand(rgb_bands[1]).ReadAsArray()
        b_data = ds.GetRasterBand(rgb_bands[2]).ReadAsArray()
        
        return np.dstack([r_data, g_data, b_data])
    
    finally:
        ds = None

