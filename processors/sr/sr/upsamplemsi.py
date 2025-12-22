import numpy as np
import rasterio
from scipy.ndimage import zoom
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import rasterio.windows
#pan+msi
def upsamplemsi(input_path, output_path, scale_factor):
    with rasterio.open(input_path) as src:
        # Calculate new dimensions
        new_height = int(src.height * scale_factor)
        new_width = int(src.width * scale_factor)

        # Create the new transform
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # Write the new image
        with rasterio.open(output_path, 'w',
                           driver='GTiff',
                           height=new_height,
                           width=new_width,
                           count=src.count,
                           dtype=src.dtypes[0],  # Use original data type
                           crs=src.crs,
                           transform=transform,
                           nodata=src.nodata) as dst:

            # Loop through the data in chunks
            for i in range(src.count):
                # Read the data in manageable chunks
                for j in range(0, src.height, 1000):  # Adjust the chunk size as needed
                    # Read a chunk of the data
                    window_height = min(1000, src.height - j)
                    data = src.read(
                        i + 1,
                        window=rasterio.windows.Window(
                            0, j, src.width, window_height
                        )
                    )

                    # Resample the data
                    resampled_data = zoom(data, (scale_factor, scale_factor), order=1)  # Bilinear resampling

                    # Write the resampled chunk to the output
                    dst.write(
                        resampled_data,
                        indexes=i + 1,
                        window=rasterio.windows.Window(0, j * scale_factor, new_width, resampled_data.shape[0])
                    )

