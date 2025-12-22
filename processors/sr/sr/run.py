import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import rasterio
from . import model
from .utils import use_checkpoint
from .data import common
from .refclip import crop_refimage
from .refclipmsi import crop_refimagemsi
import torch
import numpy as np
import os
import glob
import cv2
from .mosaic import process_mosaic
from .mosaicmsi import mosaic_raster_imagesmsi
import os
import time
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .LRclipmsi import crop_raster_imagemsi
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import os
from .fill import pad_to_multiple_of, fill
from .fillmsi import fillmsi
from .tone_map import tone_map
from .LRclip import crop_lrimage
from .upsample import upsample
from .upsamplemsi import upsamplemsi
from .demo_deploy import deploy
from .deploymsi import deploymsi
from .rectification import set_georeference
from .rpc_change import modify_rpb_file
import os
import shutil  # 导入 shutil 模块来删除目录
import threading
from multiprocessing import Process


def process_image_pipeline(input_file, output_file, RPC, lr_size=(256, 256), pre_train="model_best.pt", scale=4):
    """
    处理图像的整个流程，包括 RPC 再计算、边缘填充、上采样、裁剪、伽马校正、超分、配准和拼接。

    :param input_file: 输入图像路径
    :param output_file: 输出图像文件路径
    :param lr_size: 低分辨率图像的大小
    :param pre_train_model: 预训练模型路径
    :param pre_train_model: 预训练模型路径
    :param scale: 缩放因子
    """
    ## 获取输入文件的父文件夹
    parent_dir = os.path.dirname(input_file)
    c = parent_dir
    # 获得文件名
    file_name = os.path.basename(input_file)
    print('filename', file_name)

    middle_dir = f'{parent_dir}/{file_name.split(".")[0]}/middle0'

    if not os.path.exists(middle_dir):
        os.makedirs(middle_dir)

    start_time = time.time()  # 记录总开始时间
    """
        读取图像并返回波段数。

        :param image_path: 图像文件的路径
        :return: 波段数
        """
    with rasterio.open(input_file) as src:
        band = src.count
        print(f"图像的波段数为: {band}")

    # 生成输入 .RPB 文件路径
    input_rpb_file = os.path.splitext(input_file)[0] + '.RPB'
    output_rpb_file = os.path.splitext(output_file)[0] + '.RPB'

    # RPC再计算
    if RPC:
        start = time.time()
        modify_rpb_file(input_rpb_file, output_rpb_file, scale)
        rpc_time = time.time() - start
        print(f"RPC再计算完成，耗时: {rpc_time:.2f}秒")
    else:
        print("RPC=false，跳过 RPC 再计算步骤")

    # 边缘填充
    start = time.time()
    fill_output = f'{middle_dir}/padded_image.tif'

    if band == 1:
        fill(input_file, fill_output)
    else:
        fillmsi(input_file, fill_output)
    fill_time = time.time() - start
    print(f"边缘填充完成，耗时: {fill_time:.2f}秒")

    # 参考图像上采样
    start = time.time()
    input_image_path = fill_output
    upsample_output = f'{middle_dir}/upscaled_image.tif'
    if band == 1:
        upsample(input_image_path, upsample_output, scale)
    else:
        upsamplemsi(input_image_path, upsample_output, scale)

    upsample_time = time.time() - start
    print(f"参考图像上采样完成，耗时: {upsample_time:.2f}秒")

    # 裁剪参考图像
    start = time.time()
    refclip_output = f"{middle_dir}/REF"
    ref_size = [lr_size[0] * scale, lr_size[1] * scale]
    if band == 1:
        crop_refimage(upsample_output, refclip_output, ref_size)
    else:
        crop_refimagemsi(upsample_output, refclip_output, ref_size)

    crop_ref_time = time.time() - start
    print(f"参考图像裁剪完成，耗时: {crop_ref_time:.2f}秒")

    # 伽马校正
    start = time.time()
    hdr_image_path = fill_output
    ldr_image_path = f'{middle_dir}/ldr_image.tif'
    if band == 1:
        tone_map(hdr_image_path, ldr_image_path, gamma=2.2, exposure_adjustment=1.0, contrast_adjustment=0.9)
    else:
        print(f"伽马校正跳过")

    gamma_time = time.time() - start
    print(f"伽马校正完成，耗时: {gamma_time:.2f}秒")

    # LR裁剪
    start = time.time()
    lrclip_output = f"{middle_dir}/LR"
    if band == 1:
        crop_lrimage(ldr_image_path, lrclip_output, lr_size)
    else:
        crop_raster_imagemsi(hdr_image_path, lrclip_output, lr_size)
    lr_crop_time = time.time() - start
    print(f"LR图像裁剪完成，耗时: {lr_crop_time:.2f}秒")

    # 超分
    start = time.time()
    dir_data = lrclip_output
    dir_out = f'{middle_dir}/SR'
    print("超分图像准备完成")

    checkpoint = use_checkpoint()
    sr_model = model.Model(pre_train, checkpoint)
    sr_model.eval()
    if band == 1:
        deploy(dir_data, dir_out, sr_model)
    else:
        deploymsi(dir_data, dir_out, sr_model)
    super_resolution_time = time.time() - start
    print(f"超分完成，耗时: {super_resolution_time:.2f}秒")

    # 配准
    start = time.time()
    reference_folder = refclip_output
    target_folder = dir_out
    set_georeference(reference_folder, target_folder)
    registration_time = time.time() - start
    print(f"配准完成，耗时: {registration_time:.2f}秒")

    # 拼接
    start = time.time()
    input_folder = target_folder
    if band == 1:
        process_mosaic(input_folder, output_file)
    else:
        mosaic_raster_imagesmsi(input_folder, output_file)

    mosaic_time = time.time() - start
    print(f"图像拼接完成，耗时: {mosaic_time:.2f}秒")

    # # 删除中间结果
    # if os.path.exists(c):
    #     shutil.rmtree(c)
    #     print("中间结果已删除")

    total_time = time.time() - start_time  # 计算总耗时
    print(f"处理完成，总耗时: {total_time:.2f}秒")


def run_sr(**kwargs):
    processes = []
    left_input = kwargs.get('left_input', None)
    left_output = kwargs.get('left_output', None)
    right_input = kwargs.get('right_input', None)
    right_output = kwargs.get('right_output', None)
    RPC = kwargs.get('RPC', None)

    # 如果单个文件传入的rightinput和right_output为None
    # 启动左侧进程
    if right_output and right_input:
        process_left = Process(target=process_image_pipeline, args=(left_input, left_output, RPC))
        processes.append(process_left)
        process_left.start()

        # 启动右侧进程
        process_right = Process(target=process_image_pipeline, args=(right_input, right_output, RPC))
        processes.append(process_right)
        process_right.start()
    else:
        process_left = Process(target=process_image_pipeline, args=(left_input, left_output, RPC))
        processes.append(process_left)
        process_left.start()
    # 等待所有进程完成
    for process in processes:
        process.join()

    # 在这里进行后续步骤
    print("所有进程已完成，继续进行后续操作。")


if __name__ == '__main__':
    # 调用
    # left_input = './little/LR/BWDLR.tif'  # 输入图像路径
    # left_output = "./little/SR/BWDSR.tif"
    # right_input = './little/LR/FWDLR.tif'  # 输入图像路径
    # right_output = "./little/SR/FWDSR.tif"
    # # process_image_pipeline(input_file, output_file)
    # processes = []
    # process = Process(target=process_image_pipeline, args=(left_input, left_output))
    # processes.append(process)
    # process.start()
    #
    # process = Process(target=process_image_pipeline, args=(right_input, right_output))
    # processes.append(process)
    # process.start()
    left_input = './little/LR/11.18RSLR.tif.tif'  # 输入图像路径
    left_output = "./little/SR/11.18RSSR.tif"
    right_input = None  # 输入图像路径
    right_output = None
    RPC = False
    run_sr(left_input, left_output, right_input, right_output)
