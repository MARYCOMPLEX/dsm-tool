import os
import numpy as np
import rasterio
from rasterio.merge import merge
from skimage import exposure
from scipy import ndimage

def mosaic_raster_imagesmsi(input_folder, output_file):
    # 获取文件夹下所有的tif文件
    raster_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]

    if not raster_files:
        print("没有找到tif文件")
        return

    # 读取第一个文件以获取元数据
    with rasterio.open(raster_files[0]) as src:
        meta = src.meta.copy()

    # 使用merge函数拼接所有文件
    src_files_to_mosaic = []
    for fp in raster_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    # 确保拼接后的数据类型为uint8，并且波段数为3
    mosaic = mosaic.astype('uint8')

    # 对每个波段进行色彩均衡和插值
    for band in range(mosaic.shape[0]):
        band_data = mosaic[band, :, :]

        # 局部直方图均衡化
        band_data = exposure.equalize_adapthist(band_data, clip_limit=0.03)

        # 双线性插值
        band_data = ndimage.zoom(band_data, zoom=1.0, order=1)

        # 将处理后的数据放回原数组
        mosaic[band, :, :] = (band_data * 255).astype('uint8')

    # 更新元数据
    meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": 3,  # 三波段（RGB）
        "dtype": 'uint8'  # 数据类型为8位无符号整数
    })

    # 保存输出文件
    with rasterio.open(output_file, 'w', **meta) as dest:
        dest.write(mosaic)

    # 关闭所有打开的栅格文件
    for src in src_files_to_mosaic:
        src.close()




