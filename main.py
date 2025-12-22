import sys
import os
import multiprocessing
from pathlib import Path

# 1. 必须在所有逻辑之前调用，处理多进程打包后的递归启动问题
if __name__ == '__main__':
    multiprocessing.freeze_support()

# 2. 检测是否处于 Nuitka 编译后的环境
IS_FROZEN = "__compiled__" in globals()

if IS_FROZEN:
    # 获取 exe 所在目录
    base_path = Path(sys.argv[0]).parent

    # 强制指定 GDAL 和 PROJ 数据路径 (对应打包命令中的目标文件夹)
    os.environ['GDAL_DATA'] = str(base_path / "gdal-data")
    os.environ['PROJ_LIB'] = str(base_path / "proj")

    # 将 exe 目录加入 DLL 搜索路径 (针对 Python 3.8+ 处理 GDAL whl)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(str(base_path))

from PyQt6.QtWidgets import QApplication
# 注意：ui.main_window 及其内部导入的 torch/rasterio 放在这里之后
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DSM Generation Tool")
    app.setOrganizationName("DSM_GEN")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()