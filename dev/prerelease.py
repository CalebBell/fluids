import os
import shutil
import sys

if sys.version_info.major != 3 and sys.version_info.minor != 11:
	raise ValueError("""This prerelease script will only run on Python 3.11.
Some parts of a library change the last few decimals numbers between releases,
and other parts only have obsolete dependencies i.e. pint on earlier Python versions.
For that reason, while the pytest test suite runs everywhere,
the notebooks and doctests only run on one platform.""")

import os
import sys
from datetime import datetime


def set_file_modification_time(filename, mtime):
    atime = os.stat(filename).st_atime
    os.utime(filename, times=(atime, mtime.timestamp()))

now = datetime.now()

main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

remove_folders = ('__pycache__', '.mypy_cache', '_build', '.cache', '.ipynb_checkpoints')
bad_extensions = ('.pyc', '.nbi', '.nbc')


paths = [main_dir]

for p in paths:
    for (dirpath, dirnames, filenames) in os.walk(p):
        for bad_folder in remove_folders:
            if dirpath.endswith(bad_folder):
                shutil.rmtree(dirpath)
                continue
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if not os.path.exists(full_path):
                continue
            set_file_modification_time(full_path, now)
            for bad_extension in bad_extensions:
                if full_path.endswith(bad_extension):
                    os.remove(full_path)




test_dir = os.path.join(main_dir, 'tests')
os.chdir(test_dir)

# mod_spec = importlib.util.spec_from_file_location("make_test_stubs", os.path.join(test_dir, "make_test_stubs.py"))
# make_test_stubs = importlib.util.module_from_spec(mod_spec)
# mod_spec.loader.exec_module(make_test_stubs)

import pytest

os.chdir(main_dir)
pytest.main(["--doctest-glob='*.rst'", "--doctest-modules", "--nbval", "-n", "8", "--dist", "loadscope", "-v"])
