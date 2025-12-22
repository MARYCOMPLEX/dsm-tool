"""
Nuitka 打包脚本 - DSM Generation Tool

确保所有本地包和资源文件都被正确打包。
"""

import os
import sys
import subprocess
import site
from pathlib import Path

# --- 配置区 ---
MAIN_SCRIPT = "main.py"
EXE_NAME = "DSM_Tool"

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# --- GDAL/PROJ 数据路径检测 ---
try:
    import osgeo
    
    # 获取 osgeo 模块所在的实际根目录
    osgeo_path = Path(osgeo.__file__).parent
    
    # 尝试查找 GDAL 和 PROJ 的数据路径
    gdal_data_candidates = [
        osgeo_path / "data" / "gdal",
        osgeo_path.parent / "osgeo" / "data" / "gdal",
    ]
    
    proj_data_candidates = [
        osgeo_path / "data" / "proj",
        osgeo_path.parent / "osgeo" / "data" / "proj",
    ]
    
    GDAL_DATA_SRC = None
    PROJ_DATA_SRC = None
    
    for candidate in gdal_data_candidates:
        if candidate.exists():
            GDAL_DATA_SRC = candidate
            break
    
    for candidate in proj_data_candidates:
        if candidate.exists():
            PROJ_DATA_SRC = candidate
            break
    
    # 如果还是找不到，尝试从 site-packages 查找
    if not GDAL_DATA_SRC or not PROJ_DATA_SRC:
        site_packages = site.getsitepackages()
        if site_packages:
            sp_path = Path(site_packages[0])
            if not GDAL_DATA_SRC:
                gdal_candidate = sp_path / "osgeo" / "data" / "gdal"
                if gdal_candidate.exists():
                    GDAL_DATA_SRC = gdal_candidate
            if not PROJ_DATA_SRC:
                proj_candidate = sp_path / "osgeo" / "data" / "proj"
                if proj_candidate.exists():
                    PROJ_DATA_SRC = proj_candidate
    
    print(f"GDAL 数据路径: {GDAL_DATA_SRC} (存在: {GDAL_DATA_SRC.exists() if GDAL_DATA_SRC else False})")
    print(f"PROJ 数据路径: {PROJ_DATA_SRC} (存在: {PROJ_DATA_SRC.exists() if PROJ_DATA_SRC else False})")
    
except ImportError:
    print("警告: 未找到 osgeo 模块，GDAL/PROJ 数据路径将跳过")
    GDAL_DATA_SRC = None
    PROJ_DATA_SRC = None

# --- 构建 Nuitka 命令 ---
cmd = [
    sys.executable, "-m", "nuitka",
    "--standalone",
    f"--output-filename={EXE_NAME}",
    "--show-progress",
    "--show-memory",
    "--plugin-enable=pyqt6",
    "--windows-console-mode=disable",
    
    # 包含所有本地包（递归包含子包）
    # 使用 --include-package 会递归包含所有子包和模块
    "--include-package=processors",  # 包含 processors 及其所有子包（如 processors.sr, processors.sr.sr）
    "--include-package=tabs",
    "--include-package=loaders",
    "--include-package=views",
    "--include-package=ui",
    "--include-package=utils",
    
    # 确保动态导入的模块也被包含（pkgutil.iter_modules 等）
    "--include-module=pkgutil",
    "--include-module=importlib",
    
    # 包含第三方包
    "--include-package=rasterio",
    "--include-package=numpy",
    "--include-package=osgeo",
    
    # PyTorch 支持（如果使用深度学习算法）
    "--include-package=torch",
    "--include-package-data=torch",  # 包含 torch 的数据文件
    
    # OpenCV (cv2) 通常会被自动检测，但如果有问题可以取消下面的注释
    # "--include-package=cv2",
    
    # GDAL/PROJ 数据文件
]

# 添加 GDAL/PROJ 数据目录（如果找到）
if GDAL_DATA_SRC and GDAL_DATA_SRC.exists():
    cmd.append(f"--include-data-dir={GDAL_DATA_SRC}=gdal-data")
else:
    print("警告: 未找到 GDAL 数据目录，打包后可能无法正常工作")

if PROJ_DATA_SRC and PROJ_DATA_SRC.exists():
    cmd.append(f"--include-data-dir={PROJ_DATA_SRC}=proj-data")
else:
    print("警告: 未找到 PROJ 数据目录，打包后可能无法正常工作")

# 包含模型文件（如果存在）
model_file = PROJECT_ROOT / "model_best.pt"
if model_file.exists():
    cmd.append(f"--include-data-file={model_file}=model_best.pt")
    print(f"包含模型文件: {model_file}")

# 包含 processors/sr/sr/model 目录（如果存在，用于深度学习模型）
sr_model_dir = PROJECT_ROOT / "processors" / "sr" / "sr" / "model"
if sr_model_dir.exists():
    cmd.append(f"--include-data-dir={sr_model_dir}=processors/sr/sr/model")
    print(f"包含 SR 模型目录: {sr_model_dir}")

# 输出目录
cmd.extend([
    "--output-dir=build",
    str(MAIN_SCRIPT),
])

print("\n" + "="*60)
print("Nuitka 打包配置:")
print("="*60)
print(f"主脚本: {MAIN_SCRIPT}")
print(f"输出文件名: {EXE_NAME}")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"包含的本地包: processors, tabs, loaders, views, ui, utils")
print("="*60 + "\n")

print("正在启动 Nuitka 打包流程...")
print("这可能需要几分钟时间，请耐心等待...\n")

# 执行打包
result = subprocess.run(cmd, check=False)

if result.returncode == 0:
    print("\n" + "="*60)
    print("打包成功！")
    print("="*60)
    print(f"可执行文件位置: build/{EXE_NAME}.dist/{EXE_NAME}.exe")
    print("\n提示:")
    print("1. 确保 processors/ 目录下的所有算法文件都在打包后的目录中")
    print("2. 如果算法使用了深度学习模型，确保模型文件也被包含")
    print("3. 测试打包后的程序，确认所有功能正常")
else:
    print("\n" + "="*60)
    print("打包失败！请检查上面的错误信息。")
    print("="*60)
    sys.exit(1)
