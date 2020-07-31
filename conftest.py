import sys
import platform

def pytest_ignore_collect(path):
    path = str(path)
    if 'manual_runner' in path or 'make_test_stubs' in path or 'plot' in path:
        return True
    if sys.version[0] == '2':
        if 'numba' in path or 'typing_utils' in path:
            return True
        if 'rst' in path:
            if platform.python_version_tuple()[0:2] != ('3', '7'):
                return True
    if 'ipynb' in path and 'bench' in path:
        return True
