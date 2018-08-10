.. -*- mode: rst -*-

|Travis|_ |Codecov|_ |Readthedocs|_

.. |Travis| image:: https://travis-ci.org/AllenInstitute/mouse_connectivity_models.svg?branch=master
.. _Travis: https://travis-ci.org/AllenInstitute/mouse_connectivity_models

.. |Codecov| image:: https://codecov.io/gh/AllenInstitute/mouse_connectivity_models/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/AllenInstitute/mouse_connectivity_models

.. |Readthedocs| image:: https://readthedocs.org/projects/mouse-connectivity-models/badge/?version=latest
.. _Readthedocs: http://mouse-connectivity-models.readthedocs.io/en/latest/?badge=latest


.. image:: cortical_projection.gif

mouse_connectivity_models
=========================

mouse_connectivity_models is a Python module for constructing and testing
mesoscale connectivity models using data from the `Allen Institute for Brain
Science <https://brain-map.org>`_.

It provides models written in `Scikit-Learn <http://scikit-learn.org>`_
``estimator`` style, and has been used in the following publications:

- `High resolution data-driven model of the mouse connectome
  <https://www.biorxiv.org/content/early/2018/04/01/293019>`_

**Website**: http://mouse-connectivity-models.readthedocs.io/en/latest/

Installation
------------

Dependencies
~~~~~~~~~~~~

mouse_connectivity_models requires:

- Python (>=2.7 or >= 3.4)
- scikit-learn (>= 0.19)
- allensdk (>= 0.14.4)

For running the examples Matplotlib >= 1.3.1 is required.

User installation
~~~~~~~~~~~~~~~~~

We use `Git <https://git-scm.com/>`_ for our version control and `Github
<https://github.com/>`_ for hosting our main repository.

You can check out the latest sources and install using ``pip``::

    $ git clone https://github.com/AllenInstitute/mouse_connectivity_models.git
    $ cd mouse_connectivity_models
    $ pip install .


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
mouse_connectivity_models/blob/master/CONTRIBUTING.md>`_, then refer to the
`contributing guide <http://AllenInstitute.github.io/mouse_connectivity_models/
contributing.html>`_.


Installing the ``dev`` requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use `pipenv <https://github.com/pypa/pipenv>`_ to install the ``dev``
dependencies. If you do not have ``pipenv`` currently installed ::

   $ pip install pipenv

Then install ``dev`` dependencies ::

   $ pipenv install --dev

This will create a virtual environment on your machine for this project. To
activate the virtual environment (to develop) ::

   $ pipenv shell


Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (``mcmodels``) using `pytest <https://pytest.org>`_ ::

   $ pytest mcmodels


Help and Support
----------------

Documentation
~~~~~~~~~~~~~
The documentation that supports mouse_connectivity_models can be found at the
`Website <http://mouse-connectivity-models.readthedocs.io/en/latest/>`_.
