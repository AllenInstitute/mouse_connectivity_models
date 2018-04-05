from setuptools import setup#, find_packages

setup(
    name = 'mouse_connectivity_models',
    version = '0.1.0',
    description = """High resolution data-driven model of the mouse connectome""",
    author = "Joseph Knox",
    author_email = "josephk@alleninstitute.org",
    url = 'https://github.com/AllenInstitute/mouse_connectivity_models',
    packages = ["mcmodels"],#find_packages(),
    include_package_data=True,
    setup_requires=['pytest-runner'],
)
