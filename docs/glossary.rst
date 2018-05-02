.. currentmodule:: mcmodels

.. _glossary:

============================================
Glossary of Technical Terms and API Elements
============================================

The following is a brief glossary of core terms, expressions, and notation
used in mouse_connectivity_models and its API indended for users and
contributers.

General Terms
=============

.. glossary::

        voxel model
                The connectivity model based on :term:`nonparametric regression`
                at the resolution at the :term:`voxel` scale.

        voxel
                A 3-D cubic volume element; the generalization of a pixel.



.. _glossary_technical_terms:

Technical Terms
===============

.. glossary::

        connection density
                The :term:`connection strength` between two regions divided by
                the size of the target :term:`region`.

        connection strength
                The sum of the connection weights from all voxels in a
                source :term`region` to all voxels in a target :term:`region`.

        edge density
                The total number of edges in a graph over the total possible
                number of edges in a graph.

        frobeneous norm
                The 2-norm of a matrix viewed as a vector.

        nonparametric regression
                :term:`regression` which does not assume the form of the
                estimator.

        normalized connection density
                The :term:`connection strength` between two regions divided by
                the product of their sizes.

        normalized connection strength
                The :term:`connection strength` between two regions divided by
                the size of the source :term:`region`.

        radial basis function
                A monotonically decreasing, nonnegative function.

        regression
                Explain


.. _glossary_biological_terms:

Biological Terms
================

.. glossary::

        coarse structures
        major brain divisions
                The set of 12 major brain divisions from the 3D Allen Mouse
                Brain Reference Atlas. These include:

                * Isocortex
                * Olfactory Areas
                * Hippocampus
                * Cortical Subplate
                * Striatum
                * Pallidum
                * Thalamus
                * Hypothalamus
                * Midbrain
                * Pons
                * Medulla
                * Cerebellum

        region
        regional
                A :term:`structure` in the brain, usually refering to a fine scale
                :term:`structure`.

        structure
        structures
                See :term:`region`

        summary structures
                The set of 292 brain structures.

        wildtype
        C57BL/6J
                Mice of strain C57BL/6J which have not been genetically altered.

