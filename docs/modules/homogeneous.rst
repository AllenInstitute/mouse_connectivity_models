.. _homogeneous:

=========================================
Homogeneous Model from [Oh2014]_
=========================================

.. currentmodule:: mcmodels.models

The homogeneous model from [Oh2014]_ is a linear connectivity model via
constrained optimization and linear regression of the form:

.. math::
        \underset{w_{x, y} \geq 0 }{\text{min}} \sum_{i=1}^{|S_E|}
        \sqrt{( \sum_{x \in S_x} w_{x, y} PV(x \cap E_i) - PV(y) )^2}

that best fits the data given by the injections in the set :math:`S_E`.

This is perhaps more clearly represented as a :ref:`nonnegative least squares
regression problem <nonnegative_linear>`:

.. math::
        \underset{x}{\text{argmin}} \| Ax - b \|_2^2, \text{subject to } x \geq 0

This model seeks set of positive linear weight coefficients :math:`w_{x,y}` that
minimize the L2 prediction error. Because many injections overlap several regions,
the model attempls to assign credit to each of the source regions by relying on
multiple non-overlapping injections.

:class:`HomogeneousModel` implements MORE

Assumptions
-----------
- Homogeneity: two injections of identical volume into region X result in the
  same flourescence in a target region, irrespective of the exact posistion of
  the injection within the source area
- Additivity: the flourescence observed in a target region can be explained by
  a linear sum of appropriately weighted sources.


Region selection criteria
-------------------------

This model only fits a connectivity matrix over a subset of the 292
:term:`summary structures`. First, a region was only included if for at least
one injection experiment the injection infected at least 50 voxels in the region.
Additionally, since the injection matrix :math:`x` was poorly conditioned using
all of the remaining regions, regions were heuristically removed one-by-one to
reduce the condition number :math:`\kappa` to a predefined threshold of 1000.

        TODO: THIS IS FALSE: the paper used forward subset selection not backward:
                a greedy algorithm that iteratively added regions to a final list. After each
                addition, the conditioning number was computed and the next region added minimally
                increased this value.`

Conditioning
~~~~~~~~~~~~

explain algorithm





.. topic:: References

        .. [Oh2014] "A mesoscale connectome of the mouse brain", Oh et al,
          Nature. 2014.

