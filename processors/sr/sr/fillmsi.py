import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
#增加
def pad_to_multiple_of(array, multiple):
    # 获取图像的尺寸
    height, width = array.shape

    # 计算目标尺寸
    new_width = (width // multiple + 1) * multiple
    new_height = (height // multiple + 1) * multiple

    # 创建填充后的数组，填充颜色设置为0（黑色）
    padded_array = np.full((new_height, new_width), 255, dtype=array.dtype)

    # 将原图像数据拷贝到填充后的数组中
    padded_array[:height, :width] = array

    return padded_array

def fillmsi(input_path, output_path):
    with rasterio.open(input_path) as src:
        # 读取图像数据和元数据
        arrays = src.read()  # 读取所有波段
        dtype = arrays[0].dtype
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        width = src.width
        height = src.height

        # 对每个波段进行填充
        padded_arrays = [pad_to_multiple_of(array, 512) for array in arrays]

        # 创建一个临时的内存文件
        with MemoryFile() as memfile:
            with memfile.open(
                    driver='GTiff',
                    height=padded_arrays[0].shape[0],
                    width=padded_arrays[0].shape[1],
                    count=len(padded_arrays),  # 设置波段数
                    dtype=dtype,
                    crs=crs,
                    transform=transform,  # 保持原有的变换矩阵
                    nodata=nodata
            ) as dst:
                for i, padded_array in enumerate(padded_arrays):
                    dst.write(padded_array, i + 1)  # 写入每个波段

            # 将内存文件写入输出路径
            with open(output_path, 'wb') as f:
                f.write(memfile.read())
