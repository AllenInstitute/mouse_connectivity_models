.. currentmodule:: mcmodels

.. _glossary_ref:

============================================
Glossary of Technical Terms and API Elements
============================================

.. glossary::

        additivity
                The property which states that a given response can be explained
                by a linear sum of the inputs.

        condition number
                A measure of the sensitivity of a function. In terms of linear
                equations, the condition number gives a bound on the inaccuracy
                of the solution and is a property of the matrix of the linear
                coefficients, not of the input.

        conditioning
                The process of decreasing the :term:`condition number` of a
                function.

        connection density
                The :term:`connection strength` between two regions divided by
                the size of the target :term:`region`.

        connection strength
                The sum of the connection weights from all voxels in a
                source :term`region` to all voxels in a target :term:`region`.

        coarse structures
        major brain divisions
                The set of 12 major brain divisions from the `3D Allen Mouse
                Brain Reference Atlas <http://atlas.brain-map.org/>`_. These include:

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

        cross validation
                A model validation technique used in estimating the predictive
                accuracy of a model. Typically, the data are partitioned into
                two sets:

                - a training set with which the model is fit
                - a testing set with which the prediction error of the fitted
                  model is determined.

                Often, this partitioning is repeated `K` times (typically 5 or 10)
                using unique samples for the testing set accross each of the `K`
                folds.

                See https://en.wikipedia.org/wiki/Cross-validation_(statistics)
                for more information.

        edge density
                The total number of edges in a graph over the total possible
                number of edges in a graph.

        elastic-net
                A regularization that combines the :term:`lasso` with
                :term:`ridge regression`

        frobeneous norm
                The 2-norm of a matrix viewed as a vector.

        homogeneous
        homogeneity
                Generally: of the same kind.

        homogeneous model
                The connectivity model at the level of :term:`regions` that
                assumes that intra-regional connectivity is :term:`homogeneous` and that
                inter-regional connectivity satisfies an :term:`additivity`
                property. This model is based on non-negative :term:`linear least
                squares`.

        ill-conditioned
                A function with a high :term:`condition number`.

        kernel
                In nonparametric statistics, a kernel is a window function that
                describes the weighting method with which to combine sample
                information.

        lasso
                A regrularization technique that utilizes the :term:`L1` norm
                to promote sparsity in terms of the model parameters. See
                https://en.wikipedia.org/wiki/Lasso_(statistics)

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

        quadratic program
        quadratic programming
        qp
                See https://en.wikipedia.org/wiki/Quadratic_programming

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

        ridge regression
        tikhonov regularization
                A regularization technique that shrinks the model parameter
                coefficients by utilizing an :term:`L2` penalty.
                See https://en.wikipedia.org/wiki/Tikhonov_regularization

        singular value decomposition
                See https://en.wikipedia.org/wiki/Singular-value_decomposition

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

        well-conditioned
                A function with a low :term:`condition number`.

        white matter
        fiber tracts
                Tissue that surrounds and insulates neurons.

        wildtype
        C57BL/6J
                Mice of strain C57BL/6J which have not been genetically altered.
