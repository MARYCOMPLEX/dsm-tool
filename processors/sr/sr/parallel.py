import os
os.environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from option import args
import model
import utils
import data.common as common

import torch
import numpy as np
import os
import glob
import cv2

device = torch.device('cpu' if args.cpu else 'cuda')

from PIL import Image

from option import args
import model
import utils
import data.common as common

import torch
import numpy as np
import os
import glob
import cv2
from PIL import Image
from concurrent import futures

device = torch.device('cpu' if args.cpu else 'cuda')


def process_block(sr_model, lr_block, factor, tp, args):
    # Perform the forward-pass on the block
    lr_block = lr_block.to(device)
    sr_block = sr_model(lr_block)
    sr_block_np = np.array(sr_block.cpu().detach())[0, :].transpose([1, 2, 0])

    # Back projection for final fused result
    for bp_iter in range(args.back_projection_iters):
        lr_np = lr_block.cpu().numpy().squeeze(0).transpose(1, 2, 0) * args.rgb_range / 255.
        sr_block_np = utils.back_projection(sr_block_np, lr_np, down_kernel='cubic',
                                            up_kernel='cubic', sf=factor, range=args.rgb_range)

    if args.rgb_range == 1:
        final_sr = np.clip(sr_block_np * 255, 0, args.rgb_range * 255)
    else:
        final_sr = np.clip(sr_block_np, 0, args.rgb_range)

    return final_sr.astype(np.uint8)


def deploy(args, sr_model):
    img_ext = '.tif'
    img_lists = glob.glob(os.path.join(args.dir_data, '*' + img_ext))

    if len(img_lists) == 0:
        print("Error: there are no images in given folder!")

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    with torch.no_grad():
        for i, img_path in enumerate(img_lists):
            print("[%d/%d] %s" % (i + 1, len(img_lists), img_path))
            lr_np = cv2.imread(img_path, cv2.IMREAD_COLOR)
            lr_np = cv2.cvtColor(lr_np, cv2.COLOR_BGR2RGB)

            if args.cubic_input:
                lr_np = cv2.resize(lr_np, (lr_np.shape[0] * args.scale[0], lr_np.shape[1] * args.scale[0]),
                                   interpolation=cv2.INTER_CUBIC)

            lr = common.np2Tensor([lr_np], args.rgb_range)[0].unsqueeze(0)

            b, c, h, w = lr.shape
            factor = args.scale[0]
            tp = args.patch_size
            if not args.cubic_input:
                ip = tp // factor
            else:
                ip = tp

            assert h >= ip and w >= ip, 'LR input must be larger than the training inputs'
            sr = torch.zeros((b, c, h * factor, w * factor)) if not args.cubic_input else torch.zeros((b, c, h, w))

            futures_list = []
            with futures.ThreadPoolExecutor() as executor:
                for iy in range(0, h, ip):
                    if iy + ip > h:
                        iy = h - ip
                    ty = factor * iy

                    for ix in range(0, w, ip):
                        if ix + ip > w:
                            ix = w - ip
                        tx = factor * ix

                        lr_block = lr[:, :, iy:iy + ip, ix:ix + ip]
                        futures_list.append(executor.submit(process_block, sr_model, lr_block, factor, tp, args))

            # Collect results
            for future, iy in zip(futures.as_completed(futures_list), range(len(futures_list))):
                sr_block = future.result()
                # Determine the original indices of the block
                iy = (iy // (w // ip)) * ip
                ix = (iy % (w // ip)) * ip
                sr[:, :, ty:ty + tp, tx:tx + tp] = sr_block

            sr_np = np.array(sr.cpu().detach())[0, :].transpose([1, 2, 0])

            # Again back projection for the final fused result
            lr_np = lr_np * args.rgb_range / 255.
            for bp_iter in range(args.back_projection_iters):
                sr_np = utils.back_projection(sr_np, lr_np, down_kernel='cubic',
                                              up_kernel='cubic', sf=args.scale[0], range=args.rgb_range)

            if args.rgb_range == 1:
                final_sr = np.clip(sr_np * 255, 0, args.rgb_range * 255)
            else:
                final_sr = np.clip(sr_np, 0, args.rgb_range)

            final_sr = final_sr.astype(np.uint8)

            # Convert to grayscale (single band) if necessary
            if final_sr.shape[2] == 3:  # If RGB
                final_sr_gray = np.mean(final_sr, axis=2).astype(np.uint8)
                img = Image.fromarray(final_sr_gray)
                img.save(os.path.join(args.dir_out, os.path.split(img_path)[-1].replace(img_ext, '.tif')))
            else:
                # Save as RGB TIF (for reference)
                final_sr = cv2.cvtColor(final_sr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.dir_out, os.path.split(img_path)[-1]), final_sr)


if __name__ == '__main__':
    # args parameter setting
    args.pre_train = "model_best.pt"
    args.dir_data = './GF7'
    args.dir_out = 'SR'

    checkpoint = utils.checkpoint(args)
    sr_model = model.Model(args, checkpoint)
    sr_model.eval()

    deploy(args, sr_model)