from fontTools.ttx import process

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from . import model
import utils
from .data import common


import torch
import numpy as np
import os
import glob
import cv2

device=torch.device("cuda"if torch.cuda.is_available()else"cpu")

from PIL import Image

import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image

from multiprocessing import cpu_count, Process, Queue

# 假设 utils 和 common 模块已经定义
from . import utils



def deploy_single( dir_out,sr_model, img_lists, process_id, queue):
    with torch.no_grad():
        for i, img_path in enumerate(img_lists):

            lr_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
            lr_np = cv2.cvtColor(lr_np, cv2.COLOR_BGR2RGB)
            cubic_input = False
            rgb_range=1

            if cubic_input:
                lr_np = cv2.resize(lr_np, (lr_np.shape[0] * 4, lr_np.shape[1] * 4),
                                   interpolation=cv2.INTER_CUBIC)

            lr = common.np2Tensor([lr_np], 1)[0].unsqueeze(0)

            test_block= True

            if test_block:
                # test block-by-block
                b, c, h, w = lr.shape
                factor = 4
                tp = 128

                if not cubic_input:
                    ip = tp // factor
                else:
                    ip = tp

                assert h >= ip and w >= ip, 'LR input must be larger than the training inputs'
                if not cubic_input:
                    sr = torch.zeros((b, c, h * factor, w * factor))
                else:
                    sr = torch.zeros((b, c, h, w))

                for iy in range(0, h, ip):
                    if iy + ip > h:
                        iy = h - ip
                    ty = factor * iy

                    for ix in range(0, w, ip):
                        if ix + ip > w:
                            ix = w - ip
                        tx = factor * ix

                        # forward-pass
                        lr_p = lr[:, :, iy:iy + ip, ix:ix + ip]
                        lr_p = lr_p.to(device)
                        sr_p = sr_model(lr_p)
                        sr[:, :, ty:ty + tp, tx:tx + tp] = sr_p

            else:
                lr = lr.to(device)
                sr = sr_model(lr)

            sr_np = np.array(sr.cpu().detach())
            sr_np = sr_np[0, :].transpose([1, 2, 0])
            lr_np = lr_np * 1 / 255.
            img_ext = '.tif'

            # Again back projection for the final fused result
            for bp_iter in range(10):
                sr_np = utils.back_projection(sr_np, lr_np, down_kernel='cubic',
                                              up_kernel='cubic', sf=4, range=1)
            if rgb_range == 1:
                final_sr = np.clip(sr_np * 255, 0, 1* 255)
            else:
                final_sr = np.clip(sr_np, 0, 1)

            final_sr = final_sr.astype(np.uint8)
            final_sr = cv2.cvtColor(final_sr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dir_out, os.path.split(img_path)[-1]), final_sr)

            # Convert to grayscale (single band) if necessary
            # Assuming we want the average of the RGB channels for single band



    queue.put(process_id)  # 通知主进程该子进程已完成

def deploymsi(dir_data,dir_out, sr_model):
    img_ext = '.tif'
    img_lists = glob.glob(os.path.join(dir_data, '*'+img_ext))

    if len(img_lists) == 0:
        print("Error: there are no images in given folder!")
        return

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)




#x=1 160.29s /x=2 98.16s x=3  83.86s / x=4 78.53s / x=5 79.86s x=6 80.78s
    x = 4
    batch = len(img_lists) // x
    num_processes = len(img_lists) // batch + (1 if len(img_lists) % batch != 0 else 0)
    num_processes = min(cpu_count(), num_processes)  # 限制进程数为 CPU 核心数

    img_lists_split = [img_lists[i * batch:(i + 1) * batch] for i in range(num_processes)]

    processes = []
    queue = Queue()

    for i in range(num_processes):
        process = Process(target=deploy_single, args=(dir_out,sr_model, img_lists_split[i], i + 1, queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    while not queue.empty():
        process_id = queue.get()





if __name__ == '__main__':



    # args parameter setting
    pre_train = "model_best.pt"
    dir_data = 'G:/new_computer/software_project/共享到群里的项目/FWDLR/middle0/LR'
    dir_out = 'G:/new_computer/software_project/共享到群里的项目/FWDLR/middle0/SR'

    checkpoint = utils.checkpoint()
    sr_model = model.Model(pre_train,checkpoint)
    sr_model.eval()

    # # analyse the params of the load model
    # pytorch_total_params = sum(p.numel() for p in sr_model.parameters())
    # print(pytorch_total_params)
    # pytorch_total_params2 = sum(p.numel() for p in sr_model.parameters() if p.requires_grad)
    # print(pytorch_total_params2)
    #
    # for name, p in sr_model.named_parameters():
    #     print(name)
    #     print(p.numel())
    #     print('========')

