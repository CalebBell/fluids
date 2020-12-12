import sys
import platform

def pytest_ignore_collect(path):
    path = str(path)
    if 'manual_runner' in path or 'make_test_stubs' in path or 'plot' in path or 'prerelease' in path:
        return True
    ver_tup = platform.python_version_tuple()[0:2]
    if ver_tup < ('3', '6') or ver_tup >= ('3', '9'):
        if 'numba' in path:
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
