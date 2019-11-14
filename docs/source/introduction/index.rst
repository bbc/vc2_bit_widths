.. _introduction:

Introduction and tutorial
=========================

The VC-2 standard defines the video decoding process using infinite precision
integer arithmetic. For practical implementations, however, fixed-width
integers must be used to achieve useful performance. If a codec is built with
too few bits of precision, potentially significant artefacts may be produced in
the event of integer wrap-around or saturation. If too many bits are used,
however, the implementation will consume more resources than necessary.

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
   command_line_tools
