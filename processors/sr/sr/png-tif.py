from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None

# 设置原始图片文件夹和目标图片文件夹

input_folder = "E:/software project/TransENet-master/SR1"
output_folder= "E:/software project/TransENet-master/SR1SRTIF"
os.makedirs(output_folder, exist_ok=True)

# 获取原始图片文件夹下的所有文件
files = os.listdir(input_folder)

# 遍历每个文件
for file in files:
    # 仅处理 PNG 格式的图片
    if file.endswith('.png'):
        # 打开 RGB PNG 图像
        rgb_image = Image.open(os.path.join(input_folder, file))

        # 转换为单通道的灰度图像
        gray_image = rgb_image.convert('L')

        # 获取灰度图像的像素值
        gray_pixels = gray_image.getdata()

        # 创建新的单波段 TIFF 图像
        tiff_image = Image.new('I', gray_image.size)


        gray_pixels_scaled = [p*1 for p in gray_pixels]
        tiff_image.putdata(gray_pixels_scaled)

        # 保存 TIFF 图像到目标文件夹
        tiff_image.save(os.path.join(output_folder, file.replace('.png', '.tif')))