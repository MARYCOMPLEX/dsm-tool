"""
Nuitka 打包脚本 - DSM Generation Tool

确保所有本地包和资源文件都被正确打包。
"""
# --- 强制 Python 输出为 UTF-8，避免 Windows CI cp1252 报错 ---
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

# ✅ libs/ 下的权重文件名（运行时会从 dist/libs/ 读取）
MODEL_FILE_NAME = "model_best.pt"

# ✅ 可选：用环境变量指定权重源文件路径（CI 推荐）
# 例如在 GitHub Actions 里下载到 $GITHUB_WORKSPACE/weights/model_best.pt
# 然后设置：MODEL_BEST_PT_SRC=.../weights/model_best.pt
MODEL_SRC_ENV = "MODEL_BEST_PT_SRC"

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
    "--include-package=processors",
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
    "--include-package-data=torch",

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

# ❌ 关键修改：不再把 model_best.pt 打进包里
# model_file = PROJECT_ROOT / "model_best.pt"
# if model_file.exists():
#     cmd.append(f"--include-data-file={model_file}=model_best.pt")
#     print(f"包含模型文件: {model_file}")

# 保留你原来的 SR 模型目录逻辑（如你也想外置，可再按同样思路改）
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
print(f"权重外置策略: 打包后复制到 dist/libs/{MODEL_FILE_NAME}（不打进 exe）")
print("="*60 + "\n")

print("正在启动 Nuitka 打包流程...\n")

# 执行打包
result = subprocess.run(cmd, check=False)

if result.returncode == 0:
    print("\n" + "="*60)
    print("打包成功！")
    print("="*60)

    dist_dir = Path("build") / f"{EXE_NAME}.dist"
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 权重来源优先级：
    # 1) 环境变量指定的路径
    # 2) 项目根目录/libs/model_best.pt
    model_src = os.environ.get(MODEL_SRC_ENV, "").strip()
    model_src_path = Path(model_src) if model_src else (PROJECT_ROOT / "libs" / MODEL_FILE_NAME)
    model_dst_path = libs_dir / MODEL_FILE_NAME

    if model_src_path.exists():
        shutil.copy2(model_src_path, model_dst_path)
        print(f"✅ 已复制权重到: {model_dst_path}")
        print(f"   源文件: {model_src_path}")
    else:
        print("⚠️ 未复制权重文件（源文件不存在）")
        print(f"   你可以：")
        print(f"   1) 放置到项目根目录: {PROJECT_ROOT / 'libs' / MODEL_FILE_NAME}")
        print(f"   或 2) 设置环境变量 {MODEL_SRC_ENV} 指向权重文件路径（CI 推荐）")

    print(f"\n可执行文件位置: {dist_dir}/{EXE_NAME}.exe")
    print("\n提示:")
    print("1. 运行时请确保 dist/libs/ 下存在 model_best.pt")
    print("2. 测试打包后的程序，确认所有功能正常")
else:
    print("\n" + "="*60)
    print("打包失败！请检查上面的错误信息。")
    print("="*60)
    sys.exit(1)
