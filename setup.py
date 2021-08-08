from setuptools import setup
from setuptools import find_packages
from setuptools import find_namespace_packages

setup(
    # ...
    packages = find_packages(
        where = 'src',
    ),
    package_dir = {"":"src"}
    # ...
)