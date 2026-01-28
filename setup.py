#!/usr/bin/env python
"""Setup script with custom build to include C library."""
import subprocess
import shutil
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Custom build_py to compile C library and include it in package."""
    
    def run(self):
        # First, build the C library
        print("Building C library...")
        csrc_dir = Path(__file__).parent / "csrc"
        if csrc_dir.exists():
            result = subprocess.run(
                ["make", "clean", "all"],
                cwd=str(csrc_dir),
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                raise RuntimeError("Failed to build C library")
            
            # Copy the built library to the package directory
            build_dir = csrc_dir / "build"
            for lib_file in build_dir.glob("libml.*"):
                if lib_file.is_file():
                    dest = Path(__file__).parent / "python" / "ml_core" / lib_file.name
                    print(f"Copying {lib_file} to {dest}")
                    shutil.copy2(lib_file, dest)
        
        # Run the original build_py
        super().run()


if __name__ == "__main__":
    setup(cmdclass={"build_py": build_py})
