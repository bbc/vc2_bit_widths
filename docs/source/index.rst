``vc2_bit_widths``
==================

The :py:mod:`vc2_bit_widths` Python package provides routines for computing how
many bits of numerical precision are required for implementations of the SMPTE
ST 2042-1 `VC-2 professional video codec
<https://www.bbc.co.uk/rd/projects/vc-2>`_. In addition it also provides
routines for producing test pictures which produce large signal values in
actual video codecs.

This manual is split into three parts. In :ref:`user-manual` a general
introduction to the purpose, terminology and usage of this module is given. In
:ref:`theory-and-design`, the underlying theory and mathematical approach are
described and evaluated. Finally, :ref:`internals-and-low-level-api` gives a
more detailed overview of the implementation and lower-level features of this
software.

.. toctree::
   :hidden:

   bibliography.rst


.. _user-manual:

User's Manual
-------------

.. toctree::
    :maxdepth: 2
    
    introduction/index.rst
    usage_overview.rst
    cli/index.rst
    high_level_api/index.rst


.. _theory-and-design:

Theory and Design
-----------------

.. toctree::
    :maxdepth: 2
    
    theory_and_design/overview.rst
    theory_and_design/related_work.rst
    theory_and_design/aa_integer_arithmetic.rst
    theory_and_design/test_patterns.rst
    theory_and_design/results.rst


.. _internals-and-low-level-api:

Internals and Low-Level API
---------------------------

.. toctree::
    :maxdepth: 2
    
    low_level_api/overview.rst
    low_level_api/pattern_evaluation.rst
    low_level_api/picture_packing.rst
    low_level_api/signal_bounds.rst
    low_level_api/pattern_generation.rst
    low_level_api/pattern_optimisation.rst
    low_level_api/fast_partial_analysis_transform.rst
    low_level_api/fast_partial_analyse_quantise_synthesise.rst
    low_level_api/quantisation.rst
    low_level_api/vc2_filters.rst
    low_level_api/linexp.rst
    low_level_api/pyexp.rst
    low_level_api/infinite_arrays.rst

