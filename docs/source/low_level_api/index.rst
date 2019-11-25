Low-level Python API
====================

The following documentation defines the low-level APIs which are used
internally to implement the high-level APIs (see :ref:`high-level-api`). These
sections are intended for developers of :py:mod:`vc2_bit_widths` and for
researchers requiring unusual or low-level functionality.

The sections below are arranged in approximately reverse-dependency order with
higher-level utilities listed first.

Evaluation and deployment of test patterns
------------------------------------------

The :py:mod:`~vc2_bit_widths.pattern_evaluation` module provide functions for
evaluating test patterns (by passing them through a VC-2 encoder/decoder). Once
evaluated, these test patterns may then be grouped together (according to e.g.
required quantisation index) and packed into full-sized pictures using
:py:mod:`~vc2_bit_widths.picture_packing`.

.. toctree::
    :maxdepth: 2
    
    pattern_evaluation
    picture_packing

Calculating signal bounds and test patterns
-------------------------------------------

These modules perform static analysis on VC-2 filters to calculate theoretical
worst-case signal bounds and test patterns for given filter combinations.

.. toctree::
    :maxdepth: 2
    
    signal_bounds
    pattern_generation
    pattern_optimisation

Specialised partial evaluation functions for VC-2 filters
---------------------------------------------------------

These modules provide specialised implementations of VC-2's analysis and
synthesis transforms which compute individual intermediate and final filter
outputs in isolation. As well as providing access to intermediate values not
exposed by the VC-2 pseudocode, these implementations generally perform the
minimum possible work and so execute much more quickly.

.. note::
    
    These implementations are still Python based (albeit with some
    :py:mod:`numpy` acceleration) and so are only 'fast' when compared with the
    VC-2 pseudocode implemented in Python.

.. toctree::
    :maxdepth: 2
    
    fast_partial_analysis_transform
    fast_partial_analyse_quantise_synthesise

Implementations of VC-2's quantiser and wavelet filters
-------------------------------------------------------

The :py:mod:`~vc2_bit_widths.quantisation` module provides an implementation of
VC-2's quantisation and dequantisation routines, along with various utilities
for bounding the quantiser's outputs or inputs.

The :py:mod:`~vc2_bit_widths.vc2_filters` implements VC-2's wavelet filters in
terms of :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`\ s. These
are used to construct the algebraic descriptions of VC-2's filters used to
construct test patterns and signal bounds. They are also used to derive
efficient partial implementations of these filters in
:py:mod:`~vc2_bit_widths.fast_partial_analyse_quantise_synthesise`.

.. toctree::
    :maxdepth: 2
    
    quantisation
    vc2_filters

Utilities for representing and implementing filter behaviour
------------------------------------------------------------

These final modules implement the tools with which much of the
:py:mod:`vc2_bit_widths` software is built.

The :py:mod:`~vc2_bit_widths.linexp` module implements a specialised `Computer
Algebra System <https://en.wikipedia.org/wiki/Computer_algebra_system>`_
optimised for building and manipulating algebraic descriptions of VC-2's
filters.

The :py:mod:`~vc2_bit_widths.pyexp` module implements a scheme for extracting
Python functions which compute a single value from the output of a function
acting on whole arrays. This functionality is used within
:py:mod:`~vc2_bit_widths.fast_partial_analyse_quantise_synthesise` to build
efficient partial implementations of VC-2's synthesis filters.

Finally, :py:mod:`~vc2_bit_widths.infinite_arrays` provide an abstraction for
describing the operation of VC-2's filters in terms of array operations. As
well as being suited to building the algebraic descriptions used in this
module, this form of description itself directly provides several useful
metrics about a filter and its outputs.

.. toctree::
    :maxdepth: 2
    
    linexp
    pyexp
    infinite_arrays

