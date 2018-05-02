
.. _installation-instructions:

.. highlight:: shell

============
Installation
============

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


.. _testing:

Testing
=======

Testing requires having the `pytest <https://docs.pytest.org>`_ library. After
installation, the package can be tested by executing *from outside* the source
directory ::

        $ pytest mcmodels

This should finish with a message similar to::

        ======== 94 passed, 2 skipped, 4 wargnings in 6.46 seconds ==========

Otherwise, please consider posting an issue into the `bug tracker
<https://github.com/AllenInstitute/mouse_connectivity_models/issues>`_. Please
include your version of Python, allensdk, and mouse_connectivity_models, and
how you installed mouse_connectivity_models.
