"""
Nuitka build script - DSM Generation Tool

Goal:
- Bundle local packages and resources correctly.
- Do NOT embed model weights into the exe.
- After build, copy model_best.pt into: build/DSM_Tool.dist/libs/model_best.pt
- Work for both local build and GitHub Actions build.
"""

# --- Force UTF-8 output to avoid Windows CI cp1252 errors ---
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

# --- Config ---
MAIN_SCRIPT = "main.py"
EXE_NAME = "DSM_Tool"

# Weights are external (runtime loads from dist/libs/)
MODEL_FILE_NAME = "model_best.pt"

# Optional: specify weight source path via env (recommended in CI)
# e.g. MODEL_BEST_PT_SRC=D:/a/.../weights/model_best.pt
MODEL_SRC_ENV = "MODEL_BEST_PT_SRC"

# Optional: force mingw in CI if you want (can help MSVC heap issues)
# set NUITKA_USE_MINGW=1
USE_MINGW_ENV = "NUITKA_USE_MINGW"

# Detect CI (GitHub Actions)
IS_GHA = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# --- MSVC heap / memory tuning (Windows) ---
# Nuitka will pass CCFLAGS to C compiler. (Nuitka docs)
# Increase MSVC heap for large generated C files in CI.
if os.name == "nt":
    extra_ccflags = []
    if IS_GHA:
        extra_ccflags.append("/Zm300")  # try 300; if still fails, bump to /Zm500
    # Append to existing CCFLAGS (do not overwrite user's settings)
    if extra_ccflags:
        old = os.environ.get("CCFLAGS", "")
        os.environ["CCFLAGS"] = (old + " " + " ".join(extra_ccflags)).strip()

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

    for c in gdal_data_candidates:
        if c.exists():
            GDAL_DATA_SRC = c
            break

    for c in proj_data_candidates:
        if c.exists():
            PROJ_DATA_SRC = c
            break

    if not GDAL_DATA_SRC or not PROJ_DATA_SRC:
        site_packages = site.getsitepackages()
        if site_packages:
            sp_path = Path(site_packages[0])
            if not GDAL_DATA_SRC:
                c = sp_path / "osgeo" / "data" / "gdal"
                if c.exists():
                    GDAL_DATA_SRC = c
            if not PROJ_DATA_SRC:
                c = sp_path / "osgeo" / "data" / "proj"
                if c.exists():
                    PROJ_DATA_SRC = c

    print(f"GDAL data dir: {GDAL_DATA_SRC} (exists: {GDAL_DATA_SRC.exists() if GDAL_DATA_SRC else False})")
    print(f"PROJ data dir: {PROJ_DATA_SRC} (exists: {PROJ_DATA_SRC.exists() if PROJ_DATA_SRC else False})")

except ImportError:
    print("WARNING: osgeo not found, GDAL/PROJ data dir detection skipped.")
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
]

# Optional: force mingw (can avoid MSVC heap issues)
if os.environ.get(USE_MINGW_ENV, "").strip() == "1":
    cmd.append("--mingw64")

# CI memory stabilization (do NOT affect local unless in GHA)
if IS_GHA:
    cmd += [
        "--low-memory",
        "--jobs=2",
    ]

# Include local packages
cmd += [
    "--include-package=processors",
    "--include-package=tabs",
    "--include-package=loaders",
    "--include-package=views",
    "--include-package=ui",
    "--include-package=utils",
]

# Ensure dynamic-import helpers included
cmd += [
    "--include-module=pkgutil",
    "--include-module=importlib",
]

# Include third-party packages
cmd += [
    "--include-package=rasterio",
    "--include-package=numpy",
    "--include-package=osgeo",
]

# Torch
cmd += [
    "--include-package=torch",
    "--include-package-data=torch",
]

# IMPORTANT: avoid MSVC heap crash by not compiling sympy to C in CI.
# Torch may pull sympy; in standalone it must be present, but we can keep it uncompiled.
if IS_GHA:
    cmd += [
        "--include-package=sympy",
        "--uncompiled-module=sympy",
    ]

# GDAL/PROJ data dirs
if GDAL_DATA_SRC and GDAL_DATA_SRC.exists():
    cmd.append(f"--include-data-dir={GDAL_DATA_SRC}=gdal-data")
else:
    print("WARNING: GDAL data dir not found; runtime may fail.")

if PROJ_DATA_SRC and PROJ_DATA_SRC.exists():
    cmd.append(f"--include-data-dir={PROJ_DATA_SRC}=proj-data")
else:
    print("WARNING: PROJ data dir not found; runtime may fail.")

# Keep your SR model dir logic (if you also want it external later, we can do the same as pt)
sr_model_dir = PROJECT_ROOT / "processors" / "sr" / "sr" / "model"
if sr_model_dir.exists():
    cmd.append(f"--include-data-dir={sr_model_dir}=processors/sr/sr/model")
    print(f"Include SR model dir: {sr_model_dir}")

# Output dir + main script
cmd += [
    "--output-dir=build",
    str(MAIN_SCRIPT),
]

print("\n" + "=" * 60)
print("Nuitka build configuration:")
print("=" * 60)
print(f"Main script: {MAIN_SCRIPT}")
print(f"Output exe name: {EXE_NAME}")
print(f"Project root: {PROJECT_ROOT}")
print("Included local packages: processors, tabs, loaders, views, ui, utils")
print(f"External model strategy: copy to dist/libs/{MODEL_FILE_NAME} after build (not embedded)")
print(f"Running in GitHub Actions: {IS_GHA}")
if os.name == "nt":
    print(f"CCFLAGS: {os.environ.get('CCFLAGS','')}")
print("=" * 60 + "\n")

print("Starting Nuitka build...\n")

result = subprocess.run(cmd, check=False)

if result.returncode == 0:
    print("\n" + "=" * 60)
    print("Build succeeded.")
    print("=" * 60)

    dist_dir = Path("build") / f"{EXE_NAME}.dist"
    libs_dir = dist_dir / "libs"
    libs_dir.mkdir(parents=True, exist_ok=True)

    # Weight source priority:
    # 1) env MODEL_BEST_PT_SRC
    # 2) project_root/libs/model_best.pt
    model_src = os.environ.get(MODEL_SRC_ENV, "").strip()
    model_src_path = Path(model_src) if model_src else (PROJECT_ROOT / "libs" / MODEL_FILE_NAME)
    model_dst_path = libs_dir / MODEL_FILE_NAME

    if model_src_path.exists():
        shutil.copy2(model_src_path, model_dst_path)
        print(f"Copied weight to: {model_dst_path}")
        print(f"Weight source:   {model_src_path}")
    else:
        print("WARNING: weight file not copied (source not found).")
        print(f"Place it at: {PROJECT_ROOT / 'libs' / MODEL_FILE_NAME}")
        print(f"Or set env {MODEL_SRC_ENV} to the weight path (recommended in CI).")

    print(f"\nExe location: {dist_dir}/{EXE_NAME}.exe")
    print("\nRuntime note:")
    print(f"- Ensure dist/libs/{MODEL_FILE_NAME} exists when running.")
else:
    print("\n" + "=" * 60)
    print("Build failed. Check the error logs above.")
    print("=" * 60)
    sys.exit(1)
