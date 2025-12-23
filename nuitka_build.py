"""
Nuitka build script - DSM Generation Tool (Extreme Slim & Fast Version)
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
if os.name == "nt":
    os.environ["CCFLAGS"] = "/Zm2000"
    os.environ["NUITKA_CLCACHE"] = "1"

# --- GDAL/PROJ 数据路径探测 ---
try:
    import osgeo
    osgeo_path = Path(osgeo.__file__).parent
    GDAL_DATA_SRC = next((c for c in [osgeo_path / "data" / "gdal", osgeo_path.parent / "osgeo" / "data" / "gdal"] if c.exists()), None)
    PROJ_DATA_SRC = next((c for c in [osgeo_path / "data" / "proj", osgeo_path.parent / "osgeo" / "data" / "proj"] if c.exists()), None)
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
    # 极限瘦身：禁用全局优化（LTO），换取极快的链接速度
    "--lto=no", 
]

# --- 核心修复 2：针对 GHA 的并发与极限排除 ---
if IS_GHA:
    # 尝试并发 2 提速，同时开启低内存模式
    cmd += ["--low-memory", "--jobs=2"] 
    
    # 极限屏蔽：彻底切除不相关的重型子模块
    cmd += [
        "--nofollow-import-to=sympy",            # 彻底禁用导致超时和崩溃的罪魁祸首
        "--nofollow-import-to=matplotlib",       # 除非你必须在 GUI 里画图表
        "--nofollow-import-to=torch.testing",
        "--nofollow-import-to=torch.distributed",
        "--nofollow-import-to=torch.compiler",
        "--nofollow-import-to=torch.fx",
        "--nofollow-import-to=torch.onnx",
        "--nofollow-import-to=torch.utils.benchmark",
        "--nofollow-import-to=IPython",
    ]
else:
    cmd += ["--jobs=4"]

# --- 核心修复 3：包含必要的包 ---
cmd += [
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
print("Nuitka Build Configuration (Extreme Slim Mode):")
print("=" * 60)
print(f"Main script: {MAIN_SCRIPT}")
print(f"Bypassing modules: sympy, matplotlib, torch.testing/distributed")
print(f"Optimization: LTO Disabled (for speed)")
print(f"Jobs: {'2 (Fast GHA Mode)' if IS_GHA else '4'}")
print("=" * 60 + "\n")

# --- 执行构建 ---
result = subprocess.run(cmd, check=False)

# --- 构建后处理 ---
if result.returncode == 0:
    dist_dir = Path("build") / f"{EXE_NAME}.dist"
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)

    model_src = os.environ.get(MODEL_SRC_ENV, "").strip()
    model_src_path = Path(model_src) if model_src else (PROJECT_ROOT / "libs" / MODEL_FILE_NAME)
    
    if model_src_path.exists():
        shutil.copy2(model_src_path, libs_dir / MODEL_FILE_NAME)
        print(f"SUCCESS: Weight file copied to libs/.")
    
    print(f"\nBuild Finished. Output: {dist_dir}")
else:
    sys.exit(1)