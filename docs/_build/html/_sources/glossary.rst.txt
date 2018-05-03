.. currentmodule:: mcmodels

.. _glossary_ref:

============================================
Glossary of Technical Terms and API Elements
============================================

.. glossary::

        additivity
                The property which states that a given response can be explained
                by a linear sum of the inputs.

        connection density
                The :term:`connection strength` between two regions divided by
                the size of the target :term:`region`.

        connection strength
                The sum of the connection weights from all voxels in a
                source :term`region` to all voxels in a target :term:`region`.

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

        edge density
                The total number of edges in a graph over the total possible
                number of edges in a graph.

        frobeneous norm
                The 2-norm of a matrix viewed as a vector.

        homogeneous model
                The connectivity model at the level of :term:`regions` that
                assumes that intra-regional connectivity is homogeneous and that
                inter-regional connectivity satisfies an :term:`additivity`
                property. This model is based on non-negative :term:`linear least
                squares`.

        linear least squares
                Method of approximately solving a system of linear equations by
                minimizing the sum of squared difference between the prediction
                and the data.

        mesoscale
                A coarser scale than that of single neurons or cortical columns,
                but finer than the set of :term:`coarse structures`. Can refer
                to the level of :term:`regions` (especially :term:`summary structures`)
                or to the level of :term:`voxels`.

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
                A real-valued function whose value only depends on the distance
                from the origin.

        region
        regions
        regional
                A :term:`structure` in the brain, usually refering to a fine scale
                :term:`structure`.

        regionalized model
                The connectivity model at the level of :term:`regions` constructed
                by integrating the :term:`voxel model`.

        regressor
        regressors
        regression
                Predictive modeling technique that attempts to determine the
                strength of relation between inputs and responses.

        structure
        structures
                See :term:`region`

        summary structures
                The set of 293 brain structures representing a summary level
                ontology for the mouse brain.

        voxel model
                The connectivity model based on :term:`nonparametric regression`
                at the resolution at the :term:`voxel` scale.

        voxel
        voxels
                A 3-D cubic volume element; the generalization of a pixel.

        white matter
        fiber tracts
                Tissue that surrounds and insulates neurons.

        wildtype
        C57BL/6J
                Mice of strain C57BL/6J which have not been genetically altered.
