import glob
import subprocess
from pathlib import Path

path_fluids = (
    Path().parent.absolute()
    / "fluids"
    / "optional"
)

def build_hwm14(hwm="hwm14"):
    subprocess.run(
        [
            "f2py",
            "-m",
            hwm,
            "-h",
            path_fluids / f"{hwm}.pyf",
            path_fluids / f"{hwm}.f90",
            "--overwrite-signature",
        ]
    )
    subprocess.run(["f2py", "-c", path_fluids / f"{hwm}.pyf", path_fluids / f"{hwm}.f90"], check=True)
    files = glob.glob( f"{hwm}.*.so")
    print(files,path_fluids)
    subprocess.run(["mv", Path() / files[0], path_fluids], check=True)

def build_hwm93(hwm="hwm93"):
    subprocess.run(["f2py", "-c", path_fluids / f"{hwm}.pyf", path_fluids / f"{hwm}.for"], check=True)
    files = glob.glob(f"{hwm}.*.so")
    print(files,path_fluids)
    subprocess.run(["mv", Path() / files[0], path_fluids], check=True)


def main():
    """Build HWM14 and HWM93 Fortran modules using f2py"""
    build_hwm14()
    build_hwm93()

if __name__ == "__main__":
    main()
