# pre_compile.py
import os, sys

# 改为从环境变量 PRE_TARGET 获取参数
target = os.environ.get("PRE_TARGET", "")

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