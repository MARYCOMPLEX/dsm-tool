import os
from osgeo import gdal, gdalconst

def crop_refimagemsi(input_image, output_folder, ref_size):
    os.makedirs(output_folder, exist_ok=True)
    # 打开影像
    dataset = gdal.Open(input_image, gdalconst.GA_ReadOnly)

    # 获取影像的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 计算切割后小块的行数和列数
    num_rows = height // ref_size[0]
    num_cols = width // ref_size[1]

    # 获取影像的地理转换参数
    geo_transform = dataset.GetGeoTransform()

    # 获取波段数量
    num_bands = dataset.RasterCount

    # 切割小块
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算裁剪的起始点
            start_x = col * ref_size[1]
            start_y = row * ref_size[0]

            # 构建输出文件路径
            output_path = os.path.join(output_folder, f"tile_{row}_{col}.tif")

            # 根据裁剪的起始点计算裁剪后的地理坐标范围
            xmin = geo_transform[0] + start_x * geo_transform[1]
            ymax = geo_transform[3] + start_y * geo_transform[5]
            xmax = xmin + ref_size[1] * geo_transform[1]
            ymin = ymax + ref_size[0] * geo_transform[5]

            # 创建输出数据集
            output_driver = gdal.GetDriverByName('GTiff')
            output_dataset = output_driver.Create(output_path, ref_size[1], ref_size[0], num_bands, gdalconst.GDT_Float32)

            # 设置输出数据集的投影信息和地理转换参数
            output_dataset.SetProjection(dataset.GetProjection())
            output_dataset.SetGeoTransform(
                (xmin, geo_transform[1], geo_transform[2], ymax, geo_transform[4], geo_transform[5]))

            # 进行裁剪操作
            for i in range(num_bands):
                band = dataset.GetRasterBand(i + 1)
                data = band.ReadAsArray(start_x, start_y, ref_size[1], ref_size[0])
                output_band = output_dataset.GetRasterBand(i + 1)
                output_band.WriteArray(data)
                output_band.FlushCache()

            # 关闭数据集
            output_dataset = None

    # 关闭输入数据集
    dataset = None
