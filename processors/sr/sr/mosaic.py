import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
import glob
import os


def mosaic_raster_images(input_files, output_file):
    # 读取栅格图像
    src_files = [rasterio.open(f) for f in input_files]

    # 合并图像
    mosaic, out_transform = merge(src_files)

    # 获取图像的波段数
    num_bands = src_files[0].count

    # 保存合并后的图像
    with rasterio.open(output_file, 'w',
                       driver='GTiff',
                       height=mosaic.shape[1],
                       width=mosaic.shape[2],
                       count=num_bands,
                       dtype=mosaic.dtype,
                       crs=src_files[0].crs,
                       transform=out_transform) as dst:
        for i in range(num_bands):
            dst.write(mosaic[i], i + 1)  # 修正索引，适应单波段或多波段情况

    # 关闭打开的文件
    for src in src_files:
        src.close()


def color_balance_edges(image_array, smoothing_sigma=2):
    # 对图像进行高斯模糊处理，以平滑边缘
    return gaussian_filter(image_array, sigma=smoothing_sigma)


def process_mosaic(input_folder, output_file):
    # 获取输入文件夹中的所有 TIFF 文件
    input_files = glob.glob(os.path.join(input_folder, '*.tif'))

    if not input_files:
        print("没有找到 TIFF 文件。")
        return

    # 拼接栅格图像
    temp_file = os.path.join(input_folder, 'temp_mosaic.tif')
    mosaic_raster_images(input_files, temp_file)

    # 读取合并后的图像
    with rasterio.open(temp_file) as src:
        num_bands = src.count
        image_array = src.read(1)  # 读取第一个波段
        transform = src.transform
        crs = src.crs

    # 色彩均衡处理
    balanced_array = color_balance_edges(image_array)

    # 保存色彩均衡后的图像
    with rasterio.open(output_file, 'w',
                       driver='GTiff',
                       height=balanced_array.shape[0],
                       width=balanced_array.shape[1],
                       count=1,
                       dtype=balanced_array.dtype,
                       crs=crs,
                       transform=transform) as dst:
        dst.write(balanced_array, 1)

    # 删除临时文件
    #os.remove(temp_file)


# 使用示例
"""input_folder = 'G:/new_computer/software project/BWDLR/middle0/REF/'  # 指定输入的文件夹路径
output_file = '3857.tif'  # 输出文件路径

process_mosaic(input_folder, output_file)"""