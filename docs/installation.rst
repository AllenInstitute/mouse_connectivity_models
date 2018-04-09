
.. _installation-instructions:

.. highlight:: shell

===============
Installation
===============

.. note::
        This project is currently under development. You may wish to
        :ref:`install the latest development verision<install_bleeding_edge>`.


Installing the latest release
=============================

mouse_connectivity_models requires:

- Python (>= 2.7 or >= 3.5)
- allensdk (>= 0.14.4)

If you already have a working installation of the allensdk, the easiest way to
install mouse_connectivity_models is using ``pip`` ::

        pip install -U mouse_connectivity_models


.. _install_bleeding_edge:

Bleeding Edge
=============

We use `Git <https://git-scm.com/>`_ for our version control and `Github
<https://github.com/>`_ for hosting our main repository.

You can check out the latest sources and install using ``pip``::

    $ git clone https://github.com/AllenInstitute/mouse_connectivity_models.git
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
