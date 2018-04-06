.. _api_ref

=================
API Reference
=================

This is the class and function reference for mouse_connectivity_models. Please
refer to the :ref:`full user guide <user_guide>` for further details.


.. _core_ref:

:mod:`mcmodels.core`: Core data related classes and utility functions
=====================================================================
.. automodule:: mcmodels.core
        :nomembers:
        :no-inherited-members:

.. currentmodule:: mcmodels

.. autosummary::
        :toctree: generated/

        core.base
        core.experiment.Experiment
        core.masks.Mask


.. _regressors_ref:

:mod:`mcmodels.regressors`: Classes for Performing Regressions
=====================================================================
.. automodule:: mcmodels.regressors
        :nomembers:
        :no-inherited-members:


.. _least_squares_ref:

:mod:`mcmodels.regressors.least_squares`: Scipy Optimize Least Squares Wrapper
=====================================================================
.. automodule:: mcmodels.regressors.least_squares
        :nomembers:
        :no-inherited-members:

**User Guide** See the :ref:`least_squares` section for further details.

.. currentmodule:: mcmodels.regressors.least_squares

.. autosummary::
        :toctree: generated/

        ScipyLeastSquares
        Linear


.. _nonnegative_linear_ref:

:mod:`mcmodels.regressors.nonnegative_linear`: Nonnegative Least squares
=====================================================================
.. automodule:: mcmodels.regressors.nonnegative_linear
        :nomembers:
        :no-inherited-members:

**User Guide** See the :ref:`nonnegative_linear` section for further details.

.. currentmodule:: mcmodels.regressors.nonnegative_linear

.. autosummary::
        :toctree: generated/

        NonnegativeLinear
        NonnegativeRidge


.. _nonparametric_ref:

:mod:`mcmodels.regressors.nonparametric`: Nonparametric Regression
=====================================================================
.. automodule:: mcmodels.regressors.nonparametric
        :nomembers:
        :no-inherited-members:

**User Guide** See the :ref:`nonparametric` section for further details.

.. currentmodule:: mcmodels.regressors.nonparametric

.. autosummary::
        :toctree: generated/

        NadarayaWatson

Kernels:
---------

.. autosummary::
        :toctree: generated/

        kernels.Polynomial
        kernels.Uniform
        kernels.Epanechnikov
        kernels.Biweight
        kernels.Triweight


.. _models_ref:

:mod:`mcmodels.models`: Published* Mesoscale Connectivity Models
=====================================================================
.. automodule:: mcmodels.models
        :nomembers:
        :no-inherited-members:


.. _homogeneous_ref:

:mod:`mcmodels.regressors.homogeneous`: Homogeneous Regional Model
=====================================================================
.. automodule:: mcmodels.models
        :nomembers:
        :no-inherited-members:

**User Guide** See the :ref:`homogeneous` section for further details.

Classes
------------
.. currentmodule:: mcmodels.models.homogeneous

.. autosummary::
        :toctree: generated/

        HomogeneousModel

Functions
-------------

.. currentmodule:: mcmodels.models

.. autosummary::
        :toctree: generated/

        homogeneous.svd_subset_selection
        homogeneous.condition_with_svd_subset_selection


.. _voxel_ref:

:mod:`mcmodels.regressors.voxel`: Voxel-scale Model
=====================================================================
.. automodule:: mcmodels.models
        :nomembers:
        :no-inherited-members:

**User Guide** See the :ref:`voxel` section for further details.

.. currentmodule:: mcmodels.models.voxel

.. autosummary::
        :toctree: generated/

        RegionalizedModel
        VoxelConnectivityArray
        VoxelModel


.. _utils_ref:

:mod:`mcmodels.utils`: Utilities
=====================================================================
.. automodule:: mcmodels
        :nomembers:
        :no-inherited-members:

.. currentmodule:: mcmodels

.. autosummary::
        :toctree: generated/

        utils.get_experiment_ids
        utils.get_mcc
        utils.lex_ordered_unique
        utils.nonzero_unique
        utils.ordered_unique
        utils.padded_diagonal_fill
        utils.unionize
