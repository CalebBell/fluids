import platform
import sys

is_pypy = 'PyPy' in sys.version
ver_tup = platform.python_version_tuple()[0:2]
ver_tup = tuple(int(i) for i in ver_tup)


is_x86_or_x86_64 = platform.machine().lower() in ('i386', 'i686', 'x86', 'x86_64', 'amd64')


def pytest_ignore_collect(path):
    path = str(path)
    if 'manual_runner' in path or 'make_test_stubs' in path or 'plot' in path or 'prerelease' in path:
        return True
    if 'benchmarks' in path:
        return True
    if 'conf.py' in path:
        return True
    if 'is_pypy' and 'test_spa' in path:
        return True
    if ver_tup < (3, 7) or ver_tup >= (3, 13) or is_pypy or not is_x86_or_x86_64:
        # numba does not yet run under pypy
        if 'numba' in path:
            return True
        if '.rst' in path: # skip .rst tests as different rendering from pint and no support for NUMBER flag
            return True
    if sys.version[0] == '2':
        if 'numba' in path or 'typing_utils' in path:
            return True
        #if 'rst' in path:
        #    if platform.python_version_tuple()[0:2] != ('3', '7'):
        #        return True
        if 'test' not in path:
            return True
    if 'ipynb' in path and 'bench' in path:
        return True

#def pytest_addoption(parser, pluginmanager):
#    if sys.version[0] == '323523':
#        parser.addoption("--doctest-modules")
#        parser.addini(name="doctest_optionflags", help="", default="NORMALIZE_WHITESPACE NUMBER")

#def pytest_configure(config):
#    print(config)
    #open('/home/caleb/testoutput', 'w').write(str(1))
    #if sys.version[0] == '2':
    #    args = []
    #    #print(args)

def pytest_load_initial_conftests(args):
    a = 1
    b = 2


def pytest_configure(config):
    if sys.version[0] == '3':
        import pytest
        if pytest.__version__.split('.')[0] >= '6':
            config.addinivalue_line("addopts", '--doctest-modules')
            config.option.doctestmodules = True
            config.addinivalue_line("doctest_optionflags", "NUMBER")
#        config.addinivalue_line("addopts", config.inicfg['addopts'].replace('//', '') + ' --doctest-modules')
        #config.inicfg['addopts'] = config.inicfg['addopts'] + ' --doctest-modules'
        #
        config.addinivalue_line("doctest_optionflags", "NORMALIZE_WHITESPACE")
