"""
Nuitka build script - DSM Generation Tool (Optimized for Heavy Dependencies)
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

# --- 关键修复：解决 C1002 编译器内存溢出 ---
# 提升 MSVC 编译器的内存分配上限从 500% 到 2000%
if os.name == "nt":
    old = os.environ.get("CCFLAGS", "")
    # 移除可能存在的旧限制，统一设置为 /Zm2000
    os.environ["CCFLAGS"] = (old + " /Zm2000").strip()

# --- GDAL/PROJ 数据路径检测 (保持原样) ---
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

    if not GDAL_DATA_SRC or not PROJ_DATA_SRC:
        for sp in site.getsitepackages():
            sp_path = Path(sp)
            if not GDAL_DATA_SRC:
                c = sp_path / "osgeo" / "data" / "gdal"
                if c.exists(): GDAL_DATA_SRC = c
            if not PROJ_DATA_SRC:
                c = sp_path / "osgeo" / "data" / "proj"
                if c.exists(): PROJ_DATA_SRC = c
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

# --- 针对 GitHub Actions 环境的内存优化 ---
if IS_GHA:
    # 核心修复：强制单任务编译，防止多核并行抢占内存导致 C1002 错误
    cmd += ["--low-memory", "--jobs=1"] 
    cmd += [
        "--nofollow-import-to=torch.testing",
        "--nofollow-import-to=torch.testing._internal",
        "--nofollow-import-to=torch.utils.benchmark",
    ]
else:
    cmd += ["--jobs=4"] # 本地内存充裕可多核

# --- 依赖包含策略 ---
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
    
    # 核心修复：Sympy 的 C 代码过于庞大，强制作为字节码收集，不进行 C 编译
    "--collect-all=sympy", 
]

# 注入数据目录
if GDAL_DATA_SRC: cmd.append(f"--include-data-dir={GDAL_DATA_SRC}=gdal-data")
if PROJ_DATA_SRC: cmd.append(f"--include-data-dir={PROJ_DATA_SRC}=proj-data")

# 输出配置
cmd += ["--output-dir=build", str(MAIN_SCRIPT)]

# --- 打印构建信息 ---
print("\n" + "=" * 60)
print("Nuitka Build Configuration (Anti-OOM Mode):")
print("=" * 60)
print(f"Main script: {MAIN_SCRIPT}")
print(f"Output name: {EXE_NAME}")
print(f"Sympy Strategy: Collect-All (Prevent C1002)")
print(f"Memory Flag: {os.environ.get('CCFLAGS','')}")
print(f"Jobs: {'1 (Safe Mode)' if IS_GHA else '4'}")
print("=" * 60 + "\n")

# --- 执行构建 ---
result = subprocess.run(cmd, check=False)

# --- 构建后处理 (外部模型策略) ---
if result.returncode == 0:
    print("\nBuild succeeded.")
    dist_dir = Path("build") / f"{EXE_NAME}.dist"
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)

    # 模型复制逻辑
    model_src_path = Path(os.environ.get(MODEL_SRC_ENV, "")).strip() or (PROJECT_ROOT / "libs" / MODEL_FILE_NAME)
    model_dst_path = libs_dir / MODEL_FILE_NAME

    if Path(model_src_path).exists():
        shutil.copy2(model_src_path, model_dst_path)
        print(f"Copied weight to: {model_dst_path}")
    else:
        print(f"WARNING: No weight file found. Manual placement needed at: dist/libs/{MODEL_FILE_NAME}")

    print(f"\nDone. Exe: {dist_dir}/{EXE_NAME}.exe")
else:
    print("\nBuild failed.")
    sys.exit(1)