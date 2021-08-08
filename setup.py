from setuptools import setup

setup(
    name = 'tools',
    where = 'src',
    version = '0.0.1',
    packages = ['tools'],
    install_requires = [
        'requests',
        'importlib; python_version >= "3.0"',
    ],
)


#from . import tools_test
#from . import tools_test_drive