from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
def crop_lrimage(image_path, output_folder,lr_size):
    os.makedirs(output_folder, exist_ok=True)
    # 打开图片
    image = Image.open(image_path)

    # 获取图片的宽度和高度
    width, height = image.size

    # 确定小块的宽度和高度
    (crop_width, crop_height) = lr_size

    # 计算需要裁剪的小块数量
    num_cols = width // crop_width
    num_rows = height // crop_height

    # 逐行逐列进行裁剪和保存
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前小块的左上角和右下角坐标
            left = col * crop_width
            top = row * crop_height
            right = left + crop_width
            bottom = top + crop_height

            # 裁剪小块
            cropped_image = image.crop((left, top, right, bottom))

            # 构造保存路径和文件名
            output_file = f"{output_folder}/tile_{row}_{col}.tif"

            # 保存小块
            cropped_image.save(output_file)



# 设置输入图片路径和输出文件夹路径
#image_path ="ldr_image0.tif"    # 输入图片路径
#lr_size = (512, 512)
#output_folder = "LR0"   # 输出文件夹路径

# 调用函数进行图片裁剪
#crop_lrimage(image_path, output_folder,lr_size)