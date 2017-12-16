"""
NOTE : double check setup fields
"""
from __future__ import print_function
from setuptools import setup, find_packages
import os
import voxel_model

def prepend_find_packages(*roots):
    ''' Recursively traverse nested packages under the root directories
    '''
    packages = []

    for root in roots:
        packages += [root]
        packages += [root + '.' + s for s in find_packages(root)]

    return packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    version=voxel_model.__version__,
    name='voxel_model',
    author='Joseph Knox',
    author_email='josephk@alleninstitute.org',
    packages=perpend_find_packages('voxel_model'),
    package_data={'': ['*.cfg', '*.md', '*.txt', 'bps', 'Makefile', 'LICENSE'] },
    description = 'Core library for voxel scale mesoscale connectivity model.',
    install_requires=required,
    tests_require=test_required,
    setup_require=['setuptools'],
    url='https://github.com/jknox13/voxel_model',
    keywords=['neuroscience', 'scientific', 'bioinformatics'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
