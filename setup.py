import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": [
        "os", "pygame", "matplotlib", "numpy", "json", "math", 
        "random", "sys", "time", "scipy"
    ],
    "excludes": [],
    "include_files": [
        "README.md", 
        "dual_n_back_basic_rules.md",
        ("logs", "logs"),  # (source, destination)
    ]
}

# Base for Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="DualNBack",
    version="1.0",
    description="Dual N-Back Cognitive Training Game",
    options={"build_exe": build_exe_options},
    executables=[Executable("dual_nback.py", base=base, target_name="dual_nback")]
)