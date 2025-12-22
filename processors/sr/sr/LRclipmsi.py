import os
import rasterio
from rasterio.windows import Window


def crop_raster_imagemsi(image_path, output_folder, lr_size):
    os.makedirs(output_folder, exist_ok=True)

    # 打开栅格图像
    with rasterio.open(image_path) as src:
        # 获取图像的宽度、高度和波段数量
        width = src.width
        height = src.height
        num_bands = src.count

        # 确定小块的宽度和高度
        crop_width, crop_height = lr_size

        # 计算需要裁剪的小块数量
        num_cols = width // crop_width
        num_rows = height // crop_height

        # 逐行逐列进行裁剪和保存
        for row in range(num_rows):
            for col in range(num_cols):
                # 计算当前小块的窗口坐标
                window = Window(col * crop_width, row * crop_height, crop_width, crop_height)

                # 读取小块数据
                cropped_data = src.read(window=window)

                # 获取小块的元数据
                transform = rasterio.windows.transform(window, src.transform)
                profile = src.profile
                profile.update({
                    'width': crop_width,
                    'height': crop_height,
                    'transform': transform
                })

                # 构造保存路径和文件名
                output_file = f"{output_folder}/tile_{row}_{col}.tif"

                # 保存小块
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(cropped_data)