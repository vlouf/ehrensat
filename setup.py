#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'ehrensat'
DESCRIPTION = """GPM satellite data reader."""
URL = 'https://github.com/vlouf/ehrensat'
EMAIL = 'valentin.louf@monash.edu'
AUTHOR = 'Valentin Louf'

# What packages are required for this module to be executed?
REQUIRED = ["h5py",
            "numpy",
            "pyproj",
            "xarray",]

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

class PublishCommand(Command):
    """Support setup.py publish."""
    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except FileNotFoundError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        sys.exit()

setup(
    name=NAME,
    version='0.0.1',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=REQUIRED,
    include_package_data=True,
    license='ISC',
    classifiers=['Development Status :: 3 - alpha',
                 'License :: OSI Approved :: MIT License',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering :: Atmospheric Science',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8'],
    keywords='GPM satellite data reader',
    cmdclass={'publish': PublishCommand,},
)