import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile

def pad_to_multiple_of(array, multiple):
    # 获取图像的尺寸
    height, width = array.shape

    # 计算目标尺寸
    new_width = (width // multiple + 1) * multiple
    new_height = (height // multiple + 1) * multiple

    # 创建填充后的数组，填充颜色设置为0（黑色）
    padded_array = np.zeros((new_height, new_width), dtype=array.dtype)

    # 将原图像数据拷贝到填充后的数组中
    padded_array[:height, :width] = array

    return padded_array

def fill(input_path, output_path):
    with rasterio.open(input_path) as src:
        # 读取图像数据和元数据
        array = src.read(1)  # 读取第一个波段
        dtype = array.dtype
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        width = src.width
        height = src.height

        # 对图像数据进行填充
        padded_array = pad_to_multiple_of(array, 512)

        # 更新图像的元数据
        new_transform = from_origin(transform[2], transform[5], transform[0], transform[4])

        # 创建一个临时的内存文件
        with MemoryFile() as memfile:
            with memfile.open(
                    driver='GTiff',
                    height=padded_array.shape[0],
                    width=padded_array.shape[1],
                    count=1,
                    dtype=padded_array.dtype,
                    crs=crs,
                    transform=new_transform,
                    nodata=nodata
            ) as dst:
                dst.write(padded_array, 1)

            # 将内存文件写入输出路径
            with open(output_path, 'wb') as f:
                f.write(memfile.read())


# 调用函数
#input_image_path = 'E:/stereosr/geoeye/2/3D/3857LR.tif'  # 输入图像路径
#output_image_path = 'padded_image0.tif'  # 输出图像路径

#fill(input_image_path, output_image_path)