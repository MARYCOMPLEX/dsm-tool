from osgeo import gdal, osr
import os

# 设置已有坐标系栅格影像文件夹和无坐标系栅格影像文件夹
def set_georeference(reference_folder, target_folder):
    """
    将无坐标系的栅格影像文件夹中的影像赋予已有坐标系栅格影像文件夹中的坐标系信息。

    :param reference_folder: 包含已有坐标系栅格影像的文件夹路径
    :param target_folder: 包含无坐标系栅格影像的文件夹路径
    """
    # 获取已有坐标系栅格影像文件夹下的所有 TIFF 文件
    files = [file for file in os.listdir(reference_folder) if file.endswith('.tif')]

    # 创建输出文件夹目录
    os.makedirs(target_folder, exist_ok=True)

    # 遍历每个文件
    for file in files:
        # 构造已有坐标系栅格影像的路径
        reference_path = os.path.join(reference_folder, file)

        # 构造无坐标系栅格影像的路径
        target_path = os.path.join(target_folder, file)

        # 打开已有坐标系栅格影像
        reference_dataset = gdal.Open(reference_path)

        if reference_dataset is None:
            print(f"无法打开参考文件: {reference_path}")
            continue

        # 获取已有坐标系栅格影像的四个角点坐标
        transform = reference_dataset.GetGeoTransform()
        xmin = transform[0]
        ymax = transform[3]
        xres = transform[1]
        yres = transform[5]

        xmax = xmin + reference_dataset.RasterXSize * xres
        ymin = ymax + reference_dataset.RasterYSize * yres

        # 构造地理坐标系的投影
        target_projection = reference_dataset.GetProjection()

        # 打开无坐标系栅格影像
        target_dataset = gdal.Open(target_path, gdal.GA_Update)

        if target_dataset is None:
            print(f"无法打开目标文件: {target_path}")
            continue

        # 设置地理坐标系的投影和变换参数
        target_dataset.SetProjection(target_projection)
        target_dataset.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))

        # 刷新缓存，确保写入到文件
        target_dataset.FlushCache()

        # 关闭数据集
        reference_dataset = None
        target_dataset = None

    print("所有文件处理完毕。")

# 示例调用
if __name__ == "__main__":
    reference_folder = "./REF0/"
    target_folder = "SR0/"
    set_georeference(reference_folder, target_folder)