"""
Nuitka build script - DSM Generation Tool (Universal Fix Version)
"""

import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import subprocess
import site
import shutil
from pathlib import Path

# --- 配置区 ---
MAIN_SCRIPT = "main.py"
EXE_NAME = "DSM_Tool"
MODEL_FILE_NAME = "model_best.pt"
MODEL_SRC_ENV = "MODEL_BEST_PT_SRC"
PROJECT_ROOT = Path(__file__).parent.absolute()
IS_GHA = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"

# --- 核心修复 1：MSVC 编译器堆栈内存补丁 ---
# 解决 C1002 错误：compiler is out of heap space
if os.name == "nt":
    os.environ["CCFLAGS"] = "/Zm2000"
    # 启用 Nuitka 的 C 编译缓存环境变量
    os.environ["NUITKA_CLCACHE"] = "1"

# --- GDAL/PROJ 数据路径探测 ---
try:
    import osgeo
    osgeo_path = Path(osgeo.__file__).parent
    gdal_data_candidates = [
        osgeo_path / "data" / "gdal",
        osgeo_path.parent / "osgeo" / "data" / "gdal",
    ]
    proj_data_candidates = [
        osgeo_path / "data" / "proj",
        osgeo_path.parent / "osgeo" / "data" / "proj",
    ]
    GDAL_DATA_SRC = next((c for c in gdal_data_candidates if c.exists()), None)
    PROJ_DATA_SRC = next((c for c in proj_data_candidates if c.exists()), None)
except ImportError:
    GDAL_DATA_SRC = PROJ_DATA_SRC = None

# --- 构建 Nuitka 命令 ---
cmd = [
    sys.executable, "-m", "nuitka",
    "--standalone",
    f"--output-filename={EXE_NAME}",
    "--show-progress",
    "--show-memory",
    "--plugin-enable=pyqt6",
    "--windows-console-mode=disable",
]

# --- 核心修复 2：针对 GHA 的并发限制与排除 ---
if IS_GHA:
    # 强制单任务编译，最大化单进程可用内存
    cmd += ["--low-memory", "--jobs=1"] 
    cmd += [
        "--nofollow-import-to=torch.testing",
        "--nofollow-import-to=torch.utils.benchmark",
    ]
else:
    cmd += ["--jobs=4"]

# --- 核心修复 3：Sympy 兼容性策略（替代 --collect-all） ---
# 将庞大的模块设为直接包含，不参与深度 C 编译分析
cmd += [
    "--include-package=sympy",
    "--include-package-data=sympy",
    "--include-package=processors",
    "--include-package=tabs",
    "--include-package=loaders",
    "--include-package=views",
    "--include-package=ui",
    "--include-package=utils",
    "--include-module=pkgutil",
    "--include-module=importlib",
    "--include-package=rasterio",
    "--include-package=numpy",
    "--include-package=osgeo",
    "--include-package-data=torch",
]

# 注入 GDAL 数据
if GDAL_DATA_SRC: cmd.append(f"--include-data-dir={GDAL_DATA_SRC}=gdal-data")
if PROJ_DATA_SRC: cmd.append(f"--include-data-dir={PROJ_DATA_SRC}=proj-data")

# 输出设置
cmd += ["--output-dir=build", str(MAIN_SCRIPT)]

print("\n" + "=" * 60)
print("Nuitka Build Configuration (v1.0.3 Fixed):")
print("=" * 60)
print(f"Main script: {MAIN_SCRIPT}")
print(f"Memory Limit: /Zm2000 (Active)")
print(f"Jobs: {'1 (OOM Prevention Mode)' if IS_GHA else '4'}")
print(f"Sympy Strategy: Data Collection (Bypass C Compilation)")
print("=" * 60 + "\n")

# --- 执行构建 ---
result = subprocess.run(cmd, check=False)

# --- 构建后处理 (外部模型策略) ---
if result.returncode == 0:
    dist_dir = Path("build") / f"{EXE_NAME}.dist"
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)

    # 尝试寻找并复制模型文件
    model_src = os.environ.get(MODEL_SRC_ENV, "").strip()
    model_src_path = Path(model_src) if model_src else (PROJECT_ROOT / "libs" / MODEL_FILE_NAME)
    
    if model_src_path.exists():
        shutil.copy2(model_src_path, libs_dir / MODEL_FILE_NAME)
        print(f"SUCCESS: Model deployed to {libs_dir}")
    else:
        print(f"NOTE: libs/{MODEL_FILE_NAME} not found. Remember to add it manually.")
else:
    sys.exit(1)