import sys

def pytest_ignore_collect(path):
    path = str(path)
    if 'manual_runner' in path or 'make_test_stubs' in path or 'plot' in path:
        return True
    if sys.version[0] == '2':
        if 'numba' in path or 'typing_utils' in path:
            return True
