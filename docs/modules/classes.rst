.. _api_ref:

=================
API Reference
=================

This is the class and function reference for mouse_connectivity_models. Please
refer to the :ref:`full user guide <user_guide>` for further details.


.. _core_ref:

:mod:`mcmodels.core`: Core data related classes and utility functions
=====================================================================
.. automodule:: mcmodels.core
        :no-members:
        :no-inherited-members:

**User Guide** See the :ref:`allen_data` section for further details.

.. currentmodule:: mcmodels

.. autosummary::
        :toctree: generated/

        core.VoxelData
        core.RegionalData
        core.Experiment
        core.Mask

Utitility fucntions
-------------------
.. automodule:: mcmodels.core.utils
        :no-members:
        :no-inherited-members:

.. currentmodule:: mcmodels

.. autosummary::
        :toctree: generated/

        core.utils.compute_centroid
        core.utils.get_injection_hemisphere_id


.. _regressors_ref:

:mod:`mcmodels.regressors`: Classes for Performing Regressions
=====================================================================
.. automodule:: mcmodels.regressors
        :no-members:
        :no-inherited-members:


.. _nonnegative_linear_ref:

:mod:`mcmodels.regressors.nonnegative_linear`: Nonnegative Least squares
============================================================================
.. automodule:: mcmodels.regressors.nonnegative_linear
        :no-members:
        :no-inherited-members:

**User Guide** See the :ref:`nonnegative_linear` section for further details.

.. currentmodule:: mcmodels

Classes
-------

.. autosummary::
        :toctree: generated/

        regressors.NonnegativeLinear
        regressors.NonnegativeRidge

Functions
---------

.. autosummary::
        :toctree: generated/

        regressors.nonnegative_regression
        regressors.nonnegative_ridge_regression

.. _nonparametric_ref:

:mod:`mcmodels.regressors.nonparametric`: Nonparametric Regression
=====================================================================
.. automodule:: mcmodels.regressors.nonparametric
        :no-members:
        :no-inherited-members:

**User Guide** See the :ref:`nonparametric` section for further details.

.. currentmodule:: mcmodels

.. autosummary::
        :toctree: generated/

        regressors.NadarayaWatson
        regressors.NadarayaWatsonCV

Kernels:
---------

.. autosummary::
        :toctree: generated/

        regressors.nonparametric.kernels.Polynomial
        regressors.nonparametric.kernels.Uniform
        regressors.nonparametric.kernels.Epanechnikov
        regressors.nonparametric.kernels.Biweight
        regressors.nonparametric.kernels.Triweight


.. _models_ref:

:mod:`mcmodels.models`: Published* Mesoscale Connectivity Models
=====================================================================
.. automodule:: mcmodels.models
        :no-members:
        :no-inherited-members:


.. _homogeneous_ref:

:mod:`mcmodels.models.homogeneous`: Homogeneous Regional Model
=====================================================================
.. automodule:: mcmodels.models.homogeneous
        :no-members:
        :no-inherited-members:

**User Guide** See the :ref:`homogeneous` section for further details.

Classes
------------
.. currentmodule:: mcmodels

.. autosummary::
        :toctree: generated/

        models.HomogeneousModel

Functions
-------------

.. currentmodule:: mcmodels

.. autosummary::
        :toctree: generated/

        models.homogeneous.svd_subset_selection
        models.homogeneous.forward_subset_selection_conditioning
        models.homogeneous.backward_subset_selection_conditioning


.. _voxel_ref:

:mod:`mcmodels.models.voxel`: Voxel-scale Model
=====================================================================
.. automodule:: mcmodels.models.voxel
        :no-members:
        :no-inherited-members:

**User Guide** See the :ref:`voxel` section for further details.

.. currentmodule:: mcmodels

.. autosummary::
        :toctree: generated/

        models.VoxelModel

.. autosummary::
        :toctree: generated/

        models.voxel.RegionalizedModel
        models.voxel.VoxelConnectivityArray


.. _utils_ref:

:mod:`mcmodels.utils`: Utilities
=====================================================================
.. automodule:: mcmodels.utils
        :no-members:
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
