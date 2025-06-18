import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def minify_python_file(file_path):
    """Minify a Python file and return the minified content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    import python_minifier
    return python_minifier.minify(content, remove_annotations=True, remove_pass=True, remove_literal_statements=True)

def main():
    pkg_dir = Path(os.path.abspath('fluids'))

    # Files to exclude (relative to fluids directory)
    exclude_files = [
        'data',
    ]

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        moved_files = []
        minified_files = []

        try:
            # Move files to temporary location
            for rel_path in exclude_files:
                orig_path = pkg_dir / rel_path
                if orig_path.exists():
                    temp_path = Path(temp_dir) / rel_path
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(orig_path), str(temp_path))
                    moved_files.append((orig_path, temp_path))

            # Handle .pyi files
            for pyi_file in pkg_dir.rglob('*.pyi'):
                rel_path = pyi_file.relative_to(pkg_dir)
                temp_path = Path(temp_dir) / rel_path
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pyi_file), str(temp_path))
                moved_files.append((pyi_file, temp_path))

            # Minify .py files
            for py_file in pkg_dir.rglob('*.py'):
                with open(py_file, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                minified_content = minify_python_file(py_file)
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(minified_content)
                minified_files.append((py_file, original_content))

            # Build the wheel using Poetry or setuptools
            # You can use either, depending on your workflow:
            # subprocess.check_call([sys.executable, "-m", "poetry", "build"])
            subprocess.check_call([sys.executable, "setup.py", "bdist_wheel"])

        finally:
            # Restore original Python files
            for file_path, original_content in minified_files:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            # Restore moved files
            for orig_path, temp_path in moved_files:
                if temp_path.exists():
                    orig_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temp_path), str(orig_path))

if __name__ == "__main__":
    main()
