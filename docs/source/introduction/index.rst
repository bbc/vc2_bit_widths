.. _introduction:

Introduction and overview
=========================

The VC-2 standard defines the video decoding process using infinite precision
integer arithmetic. For practical implementations, however, fixed-width
integers must be used to achieve useful performance.

If a codec is built with too few bits of precision, artefacts may be produced
in the event of integer wrap-around or saturation (see examples below). If too
many bits are used, however, the implementation will consume more resources
than necessary.

.. image:: /_static/bit_width_artefacts.png

..
    The examples above were produced when encoding 10-bit, YCbCr, HD pictures
    using bit-widths at the 75th percentile of all peak-bit-widths found for
    natural luma images.

Perhaps surprisingly, the question 'how many bits do I need?' is not a simple
one to answer. This software attempts to provide useful estimates for these
figures based on mathematical analyses of VC-2's filters

Before introducing this software it is important to understand its limitations
and the terminology it uses. These will be introduced in the next few sections
before the command line and Python library interfaces of
:py:mod:`vc2_bit_widths` is introduced.


.. toctree::
    :maxdepth: 2
    :caption: Contents:
    
    caveats
    terminology
    overview
