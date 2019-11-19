Derivation and Internals
========================

This section of the documentation aims to give a detailed derivation of the
procedures used to generate the bit width figures and test patterns produced by
this software. In the process, each of the key parts of this software, and its
low-level APIs are introduced.

Two different, but complimentary approaches to determining the bit widths
requirements are used: theoretical modelling (using affine arithmetic) and
empirical evaluation (using specially designed test patterns). Each approach is
introduced in turn in the following sections.

.. toctree::
    :maxdepth: 2
    :caption: Contents:
    
    computing_bounds
    quantisation_bounds
    linexp
    signal_bounds
    quantisation
    infinite_arrays


..
    Outline plan...
    * Computing theoretical bounds
        * Linear filter optimisation
        * Accounting for non-linearities
            * Ignore them
            * Integer arithmetic analysis
            * Interval arithmetic
            * Affine arithmetic
            * Theorem provers
        * Quantisation effects
        * LinExp
        * signal_bounds
    * Generating expressions for VC-2 filters
        * Edge effects
        * InfiniteArrays
        * vc2_filters
    * Heuristic test patterns
        * Analysis: just linear worst case
        * Synthesis
            * Effects of quantisation
                * Gain
                * Cancelling terms
            * Greedy pattern combination
            * pattern_generation (generation and evaluation routines)
    * Optimising (synthesis) test patterns
        * Motivation
        * pattern_optimisation (Greedy random search algorithm & params)
        * Evaluating candidates quickly
            * Fast analysis
                * fast_partial_analysis_transform
            * Fast synthesis
                * PyExp
            * fast_partial_analyse_quantise_synthesise
