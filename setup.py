from setuptools import setup, find_packages

DISTNAME = 'mouse_connectivity_models'
DESCRIPTION = 'Python package providing mesoscale connectivity models for mouse.'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = 'Joseph Knox'
AUTHOR_EMAIL = 'josephk@alleninstitute.org'
URL = 'http://mouse-connectivity-models.readthedocs.io/en/latest/'
DOWNLOAD_URL = 'https://github.com/AllenInstitute/mouse_connectivity_models'
LICENSE = 'Allen Institute Software License'

import mcmodels
VERSION = mcmodels.__version__

extra_setuptools_args = dict(include_package_data=True,
                             setup_requires=['pytest-runner'])


def setup_package():
    metadata = dict(name=DISTNAME,
                    author=AUTHOR,
                    author_email=AUTHOR_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Development Status :: 3 - Alpha',
                                 'Intended Audience :: Science/Research',
                                 'Natural Language :: English',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python',
                                 'Programming Language :: Python :: 2',
                                 'Programming Language :: Python :: 2.7',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 'Topic :: Scientific/Engineering :: Bio-Informatics'],
                    **extra_setuptools_args)

    metadata['packages'] = find_packages()

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
