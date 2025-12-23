# pre_compile.py
import os, sys

# 根据命令行参数决定引入哪个包
target = sys.argv[1] if len(sys.argv) > 1 else ""

if target == "torch":
    import torch, torchvision
elif target == "qt":
    from PyQt6 import QtWidgets, QtCore, QtGui
elif target == "sci":
    import numpy, rasterio, scipy, skimage
elif target == "osgeo":
    import osgeo.gdal, osgeo.osr
elif target == "misc":
    import matplotlib, einops, tifffile, cv2

print(f"Pre-compile trigger for {target} done.")