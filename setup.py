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
    setup_requires=['pytest-runner'],
)
