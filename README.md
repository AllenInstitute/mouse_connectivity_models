voxel_model
===============================

https://github.com/AllenInstitute/voxel_model

High resolution data-driven model of the mouse connectome

Level of Support
===============================
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Installation
================================

tl;dr
--------------------------------
Assuming pipenv is installed, and one wants to use the directory `<analysis_dir>` for computation:
```
cd <analysis_dir>
$ git clone https://github.com/AllenInstitute/voxel_model.git
cd voxel_model
pipenv install --dev
pipenv shell
pip install .
cd ..
python examples/build_model.py
```
This will build the model and place the model components in `<analysis_dir>/data/`

Reading the documentation
--------------------------------
From the `voxel_model` base directory, where `<port>` is an open port (say `8000`):

1. change to `html` build directory:
```
cd docs/_build/html
```

2. start a local HTTP server

`python 2.x`:
```
python -m SimpleHTTPServer <port>
```

`python 3.x`:
```
python -m http.server <port>
```

3. view the documentation in your favorite browser @
```
http://0.0.0.0:<port>
```

Getting `pipenv`
--------------------------------
From https://github.com/pypa/pipenv:

" `pipenv` is the officially recommended Python packaging tool from Python.org "

It ties `pip` and `virualenv` together as well as generates the `Pipfile.lock` file, which is used to produce derteministic builds.

To install `pipenv` simply run:
```
pip install pipenv
```

Alternatively, one can use 'fancier' installation methods outlined here: https://docs.pipenv.org/install/#fancy-installation-of-pipenv

Testing
--------------------------------
If one would like to run the tests (and consequently build the documentation html), simply run
```
tox
```
inside the `voxel_model` directory.
