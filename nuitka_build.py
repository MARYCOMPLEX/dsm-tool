"""
Nuitka build script - DSM Generation Tool

Make sure all local packages and resource files are properly bundled.
"""

# --- Force UTF-8 output (best effort). Avoid non-ASCII prints in CI anyway. ---
import os
import sys

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

# --- Config ---
MAIN_SCRIPT = "main.py"
EXE_NAME = "DSM_Tool"

# Model filename under libs/ (runtime will load from dist/libs/)
MODEL_FILE_NAME = "model_best.pt"

# Optional: provide model source path via env in CI
# Example: MODEL_BEST_PT_SRC=D:/a/.../weights/model_best.pt
MODEL_SRC_ENV = "MODEL_BEST_PT_SRC"

# Project root dir
PROJECT_ROOT = Path(__file__).parent.absolute()

# --- GDAL/PROJ data path detection ---
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

    # Fallback: search under site-packages
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

    print(f"GDAL data dir: {GDAL_DATA_SRC} (exists: {GDAL_DATA_SRC.exists() if GDAL_DATA_SRC else False})")
    print(f"PROJ data dir: {PROJ_DATA_SRC} (exists: {PROJ_DATA_SRC.exists() if PROJ_DATA_SRC else False})")

except ImportError:
    print("WARNING: osgeo module not found. GDAL/PROJ data dir detection will be skipped.")
    GDAL_DATA_SRC = None
    PROJ_DATA_SRC = None

# --- Build Nuitka command ---
cmd = [
    sys.executable, "-m", "nuitka",
    "--standalone",
    f"--output-filename={EXE_NAME}",
    "--show-progress",
    "--show-memory",
    "--plugin-enable=pyqt6",
    "--windows-console-mode=disable",

    # Include all local packages (recursive)
    "--include-package=processors",
    "--include-package=tabs",
    "--include-package=loaders",
    "--include-package=views",
    "--include-package=ui",
    "--include-package=utils",

    # Ensure dynamic imports are included
    "--include-module=pkgutil",
    "--include-module=importlib",

    # Third-party packages
    "--include-package=rasterio",
    "--include-package=numpy",
    "--include-package=osgeo",

    # PyTorch support
    "--include-package=torch",
    "--include-package-data=torch",
]

# Add GDAL/PROJ data directories
if GDAL_DATA_SRC and GDAL_DATA_SRC.exists():
    cmd.append(f"--include-data-dir={GDAL_DATA_SRC}=gdal-data")
else:
    print("WARNING: GDAL data dir not found. The packaged app may not work correctly.")

if PROJ_DATA_SRC and PROJ_DATA_SRC.exists():
    cmd.append(f"--include-data-dir={PROJ_DATA_SRC}=proj-data")
else:
    print("WARNING: PROJ data dir not found. The packaged app may not work correctly.")

# IMPORTANT: do NOT embed model_best.pt into the executable/package.
# We will copy it to dist/libs/ after build (optional).
# model_file = PROJECT_ROOT / "model_best.pt"
# if model_file.exists():
#     cmd.append(f"--include-data-file={model_file}=model_best.pt")

# Keep your SR model directory packaging as-is
sr_model_dir = PROJECT_ROOT / "processors" / "sr" / "sr" / "model"
if sr_model_dir.exists():
    cmd.append(f"--include-data-dir={sr_model_dir}=processors/sr/sr/model")
    print(f"Include SR model dir: {sr_model_dir}")

# Output directory
cmd.extend([
    "--output-dir=build",
    str(MAIN_SCRIPT),
])

print("\n" + "=" * 60)
print("Nuitka build configuration:")
print("=" * 60)
print(f"Main script: {MAIN_SCRIPT}")
print(f"Output exe name: {EXE_NAME}")
print(f"Project root: {PROJECT_ROOT}")
print("Included local packages: processors, tabs, loaders, views, ui, utils")
print(f"External model strategy: copy to dist/libs/{MODEL_FILE_NAME} after build (not embedded)")
print("=" * 60 + "\n")

print("Starting Nuitka build...\n")

# Run build
result = subprocess.run(cmd, check=False)

if result.returncode == 0:
    print("\n" + "=" * 60)
    print("Build succeeded.")
    print("=" * 60)

    dist_dir = Path("build") / f"{EXE_NAME}.dist"
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)

    # Model source priority:
    # 1) env MODEL_BEST_PT_SRC
    # 2) PROJECT_ROOT/libs/model_best.pt
    model_src = os.environ.get(MODEL_SRC_ENV, "").strip()
    model_src_path = Path(model_src) if model_src else (PROJECT_ROOT / "libs" / MODEL_FILE_NAME)
    model_dst_path = libs_dir / MODEL_FILE_NAME

    if model_src_path.exists():
        shutil.copy2(model_src_path, model_dst_path)
        print(f"Model copied to: {model_dst_path}")
        print(f"Model source: {model_src_path}")
    else:
        print("Model NOT copied (source file not found).")
        print("You can do one of the following:")
        print(f"1) Put the model here: {PROJECT_ROOT / 'libs' / MODEL_FILE_NAME}")
        print(f"2) Or set env {MODEL_SRC_ENV} to point to the model file path (recommended for CI).")

    print(f"\nExecutable location: {dist_dir}/{EXE_NAME}.exe")
    print("\nNotes:")
    print("1) At runtime, make sure dist/libs/model_best.pt exists for model inference features.")
    print("2) Test the packaged app to confirm all features work.")
else:
    print("\n" + "=" * 60)
    print("Build failed. Check the error logs above.")
    print("=" * 60)
    sys.exit(1)
