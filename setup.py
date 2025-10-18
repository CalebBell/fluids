"""Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016-2025, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import shutil
import tempfile
from pathlib import Path

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class bdist_wheel_light(bdist_wheel):
    description = "Build a light wheel package with minified Python files and without type stubs"
    user_options = bdist_wheel.user_options + [
        ('use-pyc', None, 'Use .pyc files instead of minified .py files'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.use_pyc = False

    def finalize_options(self):
        super().finalize_options()

    def minify_python_file(self, file_path):
        """Minify a Python file and return the minified content"""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        import python_minifier
        return python_minifier.minify(content, remove_annotations=True, remove_pass=True, remove_literal_statements=True)

    def compile_to_pyc(self, file_path):
        """Compile a .py file to .pyc and return the .pyc path"""
        import py_compile
        pyc_path = file_path.with_suffix('.pyc')
        py_compile.compile(file_path, pyc_path, optimize=2)
        return pyc_path

    def minify_json(self, input_path, output_path=None):
        """Minify a JSON file"""
        import json
        with open(input_path, 'r') as f:
            data = json.load(f)
        output_path = output_path or input_path
        with open(output_path, 'w') as f:
            json.dump(data, f, separators=(',', ':'), sort_keys=True)

    def run(self):
        pkg_dir = Path(os.path.abspath("fluids"))

        # Files to exclude (relative to fluids directory)
        exclude_files = [
            "data",
        ]

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            moved_files = []
            minified_files = []
            json_files = []
            pyc_files = []

            try:
                # Move files to temporary location
                for rel_path in exclude_files:
                    orig_path = pkg_dir / rel_path
                    if orig_path.exists():
                        # Create path in temp dir maintaining structure
                        temp_path = Path(temp_dir) / rel_path
                        temp_path.parent.mkdir(parents=True, exist_ok=True)

                        if orig_path.is_dir():
                            shutil.move(str(orig_path), str(temp_path))
                        else:
                            shutil.move(str(orig_path), str(temp_path))
                        moved_files.append((orig_path, temp_path))

                # Handle .pyi files
                for pyi_file in pkg_dir.rglob("*.pyi"):
                    rel_path = pyi_file.relative_to(pkg_dir)
                    temp_path = Path(temp_dir) / rel_path
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(pyi_file), str(temp_path))
                    moved_files.append((pyi_file, temp_path))

                # Handle Python files (.py)
                for py_file in pkg_dir.rglob("*.py"):
                    if self.use_pyc:
                        # First minify the Python file
                        with open(py_file, encoding="utf-8") as f:
                            original_content = f.read()
                        minified_content = self.minify_python_file(py_file)
                        with open(py_file, "w", encoding="utf-8") as f:
                            f.write(minified_content)

                        # Then compile to .pyc
                        pyc_path = self.compile_to_pyc(py_file)
                        pyc_files.append((py_file, pyc_path))

                        # Store original content for restoration
                        minified_files.append((py_file, original_content))

                        # Move .py to temp
                        temp_path = Path(temp_dir) / py_file.relative_to(pkg_dir)
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(py_file), str(temp_path))
                        moved_files.append((py_file, temp_path))
                    else:
                        # Minify .py files as before
                        with open(py_file, encoding="utf-8") as f:
                            original_content = f.read()
                        minified_content = self.minify_python_file(py_file)
                        with open(py_file, "w", encoding="utf-8") as f:
                            f.write(minified_content)
                        minified_files.append((py_file, original_content))

                # Minify JSON files
                for json_file in pkg_dir.rglob('*.json'):
                    with open(json_file, 'r') as f:
                        original_content = f.read()
                    self.minify_json(json_file)
                    json_files.append((json_file, original_content))

                # Build the wheel
                super().run()

            finally:
                if self.use_pyc:
                    # Remove .pyc files and restore .py files
                    for py_file, pyc_path in pyc_files:
                        if pyc_path.exists():
                            pyc_path.unlink()
                else:
                    # Restore original Python files
                    for file_path, original_content in minified_files:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(original_content)

                # Restore JSON files
                for file_path, original_content in json_files:
                    with open(file_path, 'w') as f:
                        f.write(original_content)

                # Restore moved files
                for orig_path, temp_path in moved_files:
                    if temp_path.exists():
                        orig_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(temp_path), str(orig_path))

classifiers=[
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Manufacturing",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: POSIX :: BSD",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
    "Programming Language :: Python :: Implementation :: MicroPython",
    "Topic :: Education",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Physics",
]

description = "Fluid dynamics component of Chemical Engineering Design Library (ChEDL)"
keywords = ("fluid dynamics atmosphere pipe fluids compressible fluid dynamics "
            "chemical engineering mechanical engineering valve open channel "
            "tank friction pressure drop two phase pump drag reynolds "
            "sedimentation engineering pipeline process simulation particle "
            "size distribution")

setup(
    name="fluids",
    packages=["fluids"],
    license="MIT",
    version="1.1.0",
    download_url="https://github.com/CalebBell/fluids/tarball/1.1.0",
    description=description,
    long_description=open("README.rst").read(),
    install_requires=["numpy>=1.5.0", "scipy>=1.6.0"],
    extras_require={
        "Coverage documentation":  ["wsgiref>=0.1.2", "coverage>=4.0.3", "pint"]
    },
    author="Caleb Bell",
    author_email="Caleb.Andrew.Bell@gmail.com",
    platforms=["Windows", "Linux", "Mac OS", "Unix"],
    url="https://github.com/CalebBell/fluids",
    keywords=keywords,
    classifiers=classifiers,
    package_data={
        "fluids": [
            "data/*.csv", "nrlmsise00/*",
             "optional/*.py",
             "numerics/*", "constants/*"
        ]
    },
    cmdclass={
        "bdist_wheel_light": bdist_wheel_light,
    }

)
