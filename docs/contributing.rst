
.. _contributing_guide:

.. highlight:: shell

============
Contrubuting
============

We encourage the community to contribute! Please first review the :ref:`Allen
Institute Contributing Agreement <contributing_agreement>` then follow the
:ref:`how to <contributing_how_to>`.


.. _contributing_agreement:

Allen Institute Contributing Agreement
--------------------------------------

.. include:: ../CONTRIBUTING.rst


.. _contributing_how_to:

How to contribute
-----------------

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
   <https://pytest.org/>`_ for all testing. *From outside* the source directory
   (``mcmodels``) ::

        $ pytest mcmodels

4. After your new feature passes all tests, `add
   <https://git-scm.com/docs/git-add>`_ and `commit
   <https://git-scm.com/docs/git-commit>`_ your changes ::

        $ git add modified_files
        $ git commit

   Please leave an informative, specific commit message. Push your changes to your
   remote branch ::

        $ git push -u origin my-feature

5. Create a pull request from your fork following these `instructions
   <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
