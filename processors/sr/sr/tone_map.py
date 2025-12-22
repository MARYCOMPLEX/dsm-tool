import numpy as np
import imageio
import os

def tone_map(hdr_img, ldr_img, gamma=2.2, exposure_adjustment=1.0, contrast_adjustment=0.9):
    # 从 hdr_img 中读取图像并将其转换为浮点数类型的 NumPy 数组
    im = imageio.imread(hdr_img).astype(dtype=np.float64)

    # 伽马校正
    im = np.power(im, 1.0 / gamma)

    # 曝光调整
    im *= exposure_adjustment

    # 裁剪小值
    below_thres = np.percentile(im.reshape((-1, 1)), 0.5)
    im[im < below_thres] = below_thres
    # 裁剪大值
    above_thres = np.percentile(im.reshape((-1, 1)), 99.5)
    im[im > above_thres] = above_thres

    # 对比度调整
    im = (im - np.mean(im)) * contrast_adjustment + np.mean(im)

    # 将亮度范围映射到 0-255，以适应 8 位图像
    im = 255 * (im - below_thres) / (above_thres - below_thres)

    # 确保处理后的图像值在 0-255 范围内
    im = np.clip(im, 0, 255)

    # 如果 ldr_img 存在，则删除它
    if os.path.exists(ldr_img):
        os.remove(ldr_img)


    imageio.imwrite(ldr_img, im.astype(dtype=np.uint8))

# 调用 tone_map 函数
#hdr_image_path = "padded_image0.tif"  # HDR 图像的路径
#ldr_image_path = 'ldr_image0.tif'  # 处理后图像的保存路径

#tone_map(hdr_image_path, ldr_image_path, gamma=2.2, exposure_adjustment=1.0, contrast_adjustment=0.9)


