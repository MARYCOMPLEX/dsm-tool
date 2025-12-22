
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import cv2
import numpy as np

def convert_to_hdr_opencv(ldr_img_path, hdr_img_path, gamma=2.2, exposure_adjustment=1.0, contrast_adjustment=0.9):
    # 读取图像
    im = cv2.imread(ldr_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 确保图像值在 0-255 范围内
    im = np.clip(im, 0, 255) / 255.0

    # 打印调试信息
    print("LDR Image - Min:", im.min())
    print("LDR Image - Max:", im.max())
    print("LDR Image - dtype:", im.dtype)

    # 反向对比度调整
    mean = np.mean(im)
    im = (im - mean) / contrast_adjustment + mean

    # 反向曝光调整
    im /= exposure_adjustment

    # 反向伽马校正
    im = np.power(im, gamma)

    # 将亮度范围映射回原始范围
    im = np.clip(im, 0, 1)

    # 打印调试信息
    print("HDR Image - Min:", im.min())
    print("HDR Image - Max:", im.max())
    print("HDR Image - dtype:", im.dtype)

    # 将图像数据映射到 16 位范围
    im = (im * 65535.0).astype(np.uint16)

    # 保存 HDR 图像
    cv2.imwrite(hdr_img_path, im)


# 调用 convert_to_hdr 函数
#ldr_image_path = "3857.tif"  # LDR 图像的路径
#hdr_image_path = '3857HDR.tif'  # 处理后图像的保存路径

#convert_to_hdr_opencv(ldr_image_path, hdr_image_path)