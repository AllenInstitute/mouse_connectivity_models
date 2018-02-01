from setuptools import setup, find_packages

setup(
    name = 'voxel_model',
    version = '0.1.0',
    description = """High resolution data-driven model of the mouse connectome""",
    author = "Joseph Knox",
    author_email = "josephk@alleninstitute.org",
    url = 'https://github.com/AllenInstitute/voxel_model',
    packages = find_packages(),
    include_package_data=True,
    entry_points={
          'console_scripts': [
              'voxel_model = voxel_model.__main__:main'
        ]
    },
    setup_requires=['pytest-runner'],
)
