Low-level API Overview
======================

The remaining documentation defines the low-level APIs which are used
internally to implement the high-level APIs (see :ref:`high-level-api`). This
documentation is primarily intended for developers of :py:mod:`vc2_bit_widths`
but also potentially also researchers requiring unusual or low-level
functionality.

The sections below are arranged in approximately reverse-dependency order with
higher-level utilities listed first.

Evaluation and deployment of test patterns
------------------------------------------

The :py:mod:`~vc2_bit_widths.pattern_evaluation` module provide functions for
evaluating test patterns (by passing them through a VC-2 encoder/decoder). Once
evaluated, these test patterns may then be grouped together (according to e.g.
required quantisation index) and packed into full-sized pictures using
:py:mod:`~vc2_bit_widths.picture_packing`.

Calculating signal bounds and test patterns
-------------------------------------------

The :py:mod:`~vc2_bit_widths.signal_bounds` module performs static analysis on
VC-2 filters to calculate theoretical worst-case signal bounds. These analyses
may then be converted into test patterns using the components in the
:py:mod:`~vc2_bit_widths.pattern_generation` module and optionally further
optimised using algorithms in the
:py:mod:`~vc2_bit_widths.pattern_optimisation` module.


Specialised partial evaluation functions for VC-2 filters
---------------------------------------------------------

The :py:mod:`~vc2_bit_widths.fast_partial_analysis_transform` and
:py:mod:`~vc2_bit_widths.fast_partial_analyse_quantise_synthesise` modules
provide specialised implementations of VC-2's analysis and synthesis transforms
(respectively).

These implementations compute 'partial' analyses and syntheses meaning they
only calculate one analysis or synthesis output value. Since this is often all
that is required by this software, ths can save a significant amount of
computation.

These implementations are also able to directly compute intermediate results
which would ordinarily be accessible in ordinary analysis or synthesis
implementations.

.. note::
    
    These implementations are still Python based (albeit with some
    :py:mod:`numpy` acceleration) and so are so the 'fast' monicer only truly
    applies when compared with the VC-2 pseudocode implemented in Python.


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


Utilities for representing and implementing filter behaviour
------------------------------------------------------------

Finally :py:mod:`vc2_bit_widths` contains a number of low-level mathematical
modules which provide the foundations upon which the others are built.

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
