.. -*- mode: rst -*-

|Travis|_ |Codecov|_ |Python27|_ |Python35|_ |Python36|_

.. |Travis| image:: https://api.travis-ci.org/AllenInstitute/mouse_connectivity_models.svg?branch=master
.. _Travis: https://api.travis-ci.org/AllenInstitute/mouse_connectivity_models

.. |Codecov| image:: https://codecov.io/github/AllenInstitute/mouse_connectivity_models/badge.svg?branch=master&svg=true
.. _Codecov: https://codecov.io/github/AllenInstitute/mouse_connectivity_models?branch=master

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://badge.fury.io/py/mouse_connectivity_models

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/mouse_connectivity_models

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://badge.fury.io/py/mouse_connectivity_models


mouse_connectivity_models
===============================

mouse_connectivity_models is a Python module for constructing and testing
mesoscale connectivity models using data from the `Allen Institute for Brain
Science <https://brain-map.org>`_.

It provides models written in `Scikit-Learn <http://scikit-learn.org>`_
``estimator`` style, and has been used in several publications INSERT CITATIONS.

Website: http://AllenInstitute.github.io/mouse_connectivity_models


Installation
------------

Dependencies
~~~~~~~~~~~~

mouse_connectivity_models requires:

- Python (>=2.7 or >= 3.4)
- scikit-learn (>= 0.19)
- allensdk (>= 0.14.4)

For running the examples Matplotlib >= 1.3.1 is requried.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of the allensdk, the easiest way to
install mouse_connectivity_models is using ``pip`` ::

        pip install -U mouse_connectivity_models

The documentation includes more detailed `installation instructions
<http://AllenInstitute.github.io/mouse_connectivity_models/installation.html>`_

Level of Support
----------------
We are not currently supporting this code, but simply releasing it to the
community AS IS but are not able to provide any guarantees of support. The
community is welcome to submit issues, but you should not expect an active
response.

Contrubuting
------------
We encourage the community to contribute! Please first review the `Allen
Institute Contributing Agreement <https://github.com/AllenInstitute/
mouse_connectivity_models/blob/master/CONTRIBUTING.md>`_.

How to contribute
~~~~~~~~~~~~~~~~~

1. Fork the `main repository <https://github.com/AllenInstitute/
mouse_connectivity_models>`_ on Github by following `this guide
<https://help.github.com/articles/fork-a-repo>`_

2. `Clone <https://git-scm.com/docs/git-clone>`_ your fork to your local
   machine ::

   $ git clone git@github.com:<github_username>/mouse_connectivity_models.git
   $ cd mouse_connectivity_models

3. Create a feature branch to hold your development changes ::

   $ git checkout -b new-feature

4. Use `pipenv <https://github.com/pypa/pipenv>`_ to install the ``dev``
   dependencies. If you do not have ``pipenv`` currently installed ::

   $ pip install pipenv

Then install ``dev`` dependencies ::

   $ pipenv install --dev

This will create a virtual environment on your machine for this project. To
activate the virtual environment (to develop) ::

   $ pipenv shell

5. All contributions must pass tests to be merged with the main repository. Any
   new features must have test scripts written. We use `pytest
   <https://pytest.org/>`_ for all testing. From outside the source directory
   (``mcmodels``) ::

   $ pytest mcmodels

4. After your new feature passes all tests, `add
   https://git-scm.com/docs/git-add>`_ and `commit
   https://git-scm.com/docs/git-commit>`_ your changes ::

   $ git add modified_files
   $ git commit

Please leave an informative, specific commit message. Push your changes to your
remote branch ::

   $ git push -u origin my-feature

5. Create a pull request from your fork following these `instructions
<https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_


Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (``mcmodels``) using ``pytest``::

        pytest mcmodels


Help and Support
----------------

Documentation
~~~~~~~~~~~~~
The documentation that supports mouse_connectivity_models can be found at the
`Website <http://AllenInstitute.github.io/mouse_connectivity_models>_`.
