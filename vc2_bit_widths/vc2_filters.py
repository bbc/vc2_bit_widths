r"""
:py:mod:`vc2_bit_widths.vc2_filters`: VC-2 Filters Implemented as :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`\ s
==============================================================================================================================

This module provides an implementation of the complete VC-2 wavelet filtering
process in terms of :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`\
s.

By using :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`\ s as inputs,
algebraic descriptions (using :py:class:`~vc2_bit_widths.linexp.LinExp`) of the
VC-2 filters may be assembled. From these analyses may be performed to
determine, for example, signal ranges and rounding error bounds.

Using :py:class:`~vc2_bit_widths.infinite_arrays.VariableArray`\ s as inputs,
Python functions may be generated (using :py:mod:`~vc2_bit_widths.pyexp`) which
efficiently compute individual filter outputs or intermediate values in
isolation.


Usage
-----

To create an :py:class:`InfiniteArray`\ s-based description of a VC-2 filter we
must first define the filter to be implemented. In particular, we need a set of
:py:class:`vc2_data_tables.LiftingFilterParameters` describing the wavelets to
use. In practice these are easily obtained from
:py:data:`vc2_data_tables.LIFTING_FILTERS` like so::

    >>> from vc2_data_tables import WaveletFilters, LIFTING_FILTERS
    
    >>> wavelet_index = WaveletFilters.haar_with_shift
    >>> wavelet_index_ho = WaveletFilters.le_gall_5_3
    >>> dwt_depth = 1
    >>> dwt_depth_ho = 3
    
    >>> h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    >>> v_filter_params = LIFTING_FILTERS[wavelet_index]

Given this description we can construct a set of symbolic analysis filters
using :py:func:`analysis_transform`::

    >>> from vc2_bit_widths.infinite_arrays import SymbolArray
    >>> from vc2_bit_widths.vc2_filters import analysis_transform
    
    >>> input_picture = SymbolArray(2)
    >>> output_coeff_arrays, intermediate_analysis_arrays = analysis_transform(
    ...     h_filter_params,
    ...     v_filter_params,
    ...     dwt_depth,
    ...     dwt_depth_ho,
    ...     input_picture,
    ... )

Two dictionaries are returned. The first dictionary, ``output_coeff_arrays``,
provides a nested dictionary of the form ``{level: {orient: array, ...}, ...}``
containing the :py:class:`InfiniteArray`\ s representing the generated
transform coefficients.

The second dictionary, ``intermediate_analysis_arrays``, is of the form
``{(level, array_name): array, ...}`` and exposes every intermediate
:py:class:`InfiniteArray` from the filtering process (see :ref:`terminology`
for a guide to the naming convention used). This dictionary contains a superset
of the arrays contained in the first.

Similarly we can use :py:func:`synthesis_transform` to construct an algebraic
description of the synthesis filters. This function takes an array for
each transform component as input. The :py:func:`make_symbol_coeff_arrays`
utility function provides a convenient way to produce the necessary
:py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`\ s::

    >>> from vc2_bit_widths.vc2_filters import (
    ...     make_symbol_coeff_arrays,
    ...     synthesis_transform,
    ... )
    
    >>> input_coeff_arrays = make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho)
    >>> output_picture, intermediate_synthesis_arrays = synthesis_transform(
    ...     h_filter_params,
    ...     v_filter_params,
    ...     dwt_depth,
    ...     dwt_depth_ho,
    ...     input_coeff_arrays,
    ... )

As before, two values are returned. The first, ``output_picture``, is a
:py:class:`InfiniteArray` representing the final decoded picture. The second,
``intermediate_synthesis_arrays``, again contains all of the intermediate
:py:class:`InfiniteArray`\ s (and the output picture).

.. warning::

    The :py:func:`analysis_transform` and :py:func:`synthesis_transform`
    functions always return almost immediately since
    :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`\ s only compute
    their values on-demand. For very large transforms, accessing values within
    these arrays (and triggering their evaluation) can take a non-trivial
    amount of time and memory.


Omitting arrays
---------------

Some of the intermediate arrays returned by :py:func:`analysis_transform` and
:py:func:`synthesis_transform` are simple interleavings/subsamplings/renamings
of other intermediate arrays. These arrays may be identified using their
:py:attr:`~vc2_bit_widths.infinite_arrays.InfiniteArray.nop` property and
skipped to avoid duplicating work when performing filter analysis.

When arrays have been skipped during processing it can still be helpful to show
the duplicate entries when results are presented. The
:py:func:`add_missing_analysis_values` and
:py:func:`add_missing_synthesis_values` functions are provided to perform
exactly this task.

For example, lets count up the number of symbols in each filter phase in the
example wavelet transforms, skipping duplicate arrays::

    >>> def count_symbols(expression):
    ...     return len(list(expression.symbols()))
    
    >>> analysis_symbol_counts = {
    ...     (level, array_name, x, y): count_symbols(array[x, y])
    ...     for (level, array_name), array in intermediate_analysis_arrays.items()
    ...     for x in range(array.period[0])
    ...     for y in range(array.period[1])
    ...     if not array.nop
    ... }
    >>> synthesis_symbol_counts = {
    ...     (level, array_name, x, y): count_symbols(array[x, y])
    ...     for (level, array_name), array in intermediate_synthesis_arrays.items()
    ...     for x in range(array.period[0])
    ...     for y in range(array.period[1])
    ...     if not array.nop
    ... }

We can then fill in all of the missing entries and present the results to the
user::

    >>> from vc2_bit_widths.vc2_filters import (
    ...     add_missing_analysis_values,
    ...     add_missing_synthesis_values,
    ... )
    
    >>> full_analysis_symbol_counts = add_missing_analysis_values(
    ...     h_filter_params,
    ...     v_filter_params,
    ...     dwt_depth,
    ...     dwt_depth_ho,
    ...     analysis_symbol_counts,
    ... )
    >>> full_synthesis_symbol_counts = add_missing_synthesis_values(
    ...     h_filter_params,
    ...     v_filter_params,
    ...     dwt_depth,
    ...     dwt_depth_ho,
    ...     synthesis_symbol_counts,
    ... )
    
    >>> for (level, array_name, x, y), symbol_count in full_analysis_symbol_counts.items():
    ...     print("{} {} {} {}: {} symbols".format(
    ...         level, array_name, x, y, symbol_count
    ...     ))
    4 Input 0 0: 1 symbols
    4 DC 0 0: 1 symbols
    4 DC' 0 0: 1 symbols
    4 DC' 1 0: 4 symbols
    <...snip...>
    1 DC'' 0 0: 340 symbols
    1 DC'' 1 0: 245 symbols
    1 L 0 0: 340 symbols
    1 H 0 0: 245 symbols
    
    >>> for (level, array_name, x, y), symbol_count in full_synthesis_symbol_counts.items():
    ...     print("{} {} {} {}: {} symbols".format(
    ...         level, array_name, x, y, symbol_count
    ...     ))
    1 L 0 0: 1 symbols
    1 H 0 0: 1 symbols
    1 DC'' 0 0: 1 symbols
    1 DC'' 1 0: 1 symbols
    <...snip...>
    4 Output 14 0: 34 symbols
    4 Output 14 1: 38 symbols
    4 Output 15 0: 42 symbols
    4 Output 15 1: 48 symbols


API
---

Transforms
``````````

.. autofunction:: analysis_transform

.. autofunction:: synthesis_transform


Coefficient array creation utilities
````````````````````````````````````

.. autofunction:: make_symbol_coeff_arrays

.. autofunction:: make_variable_coeff_arrays


Omitted value insertion
```````````````````````

.. autofunction:: add_missing_analysis_values

.. autofunction:: add_missing_synthesis_values

"""

__all__ = [
    "analysis_transform",
    "synthesis_transform",
    "make_coeff_arrays",
    "make_symbol_coeff_arrays",
    "make_variable_coeff_arrays",
    "add_missing_analysis_values",
    "add_missing_synthesis_values",
]

import math

from collections import defaultdict, OrderedDict

from vc2_data_tables import (
    LiftingFilterParameters,
    LiftingStage,
    LiftingFilterTypes,
)

from vc2_bit_widths.infinite_arrays import (
    SymbolArray,
    VariableArray,
    LiftedArray,
    RightShiftedArray,
    LeftShiftedArray,
    SubsampledArray,
    InterleavedArray,
)

from vc2_bit_widths.pyexp import Argument


def convert_between_synthesis_and_analysis(lifting_filter_parameters):
    """
    Given a synthesis wavelet filter specification, return the complementary
    analysis filter (or visa versa).
    
    Parameters
    ==========
    lifting_filter_parameters : :py:class:`vc2_data_tables.LiftingFilterParameters`
    
    Returns
    =======
    lifting_filter_parameters : :py:class:`vc2_data_tables.LiftingFilterParameters`
    """
    return LiftingFilterParameters(
        stages=[
            LiftingStage(
                lift_type={
                    LiftingFilterTypes.even_add_odd: LiftingFilterTypes.even_subtract_odd,
                    LiftingFilterTypes.even_subtract_odd: LiftingFilterTypes.even_add_odd,
                    LiftingFilterTypes.odd_add_even: LiftingFilterTypes.odd_subtract_even,
                    LiftingFilterTypes.odd_subtract_even: LiftingFilterTypes.odd_add_even,
                }[stage.lift_type],
                S=stage.S,
                L=stage.L,
                D=stage.D,
                taps=stage.taps,
            )
            for stage in reversed(lifting_filter_parameters.stages)
        ],
        filter_bit_shift=lifting_filter_parameters.filter_bit_shift,
    )


def make_coeff_arrays(dwt_depth, dwt_depth_ho, make_array):
    r"""
    Create a set of :py:class:`InfiniteArray`\ s representing transform
    coefficient values, as expected by :py:func:`synthesis_transform`.
    
    See also: :py:func:`make_symbol_coeff_arrays` and
    :py:func:`make_variable_coeff_arrays`.
    
    Parameters
    ==========
    make_array : function(level, orientation) -> :py:class:`InfiniteArray`
        A function which produces an :py:class:`InfiniteArray` for a specified
        transform level (int) and orientation (one of "L", "H", "LL", "LH",
        "HL" or "HH").
    
    Returns
    =======
    coeff_arrays : {level: {orientation: :py:class:`InfiniteArray`, ...}, ...}
        The transform coefficient values. These dictionaries are indexed the
        same way as 'coeff_data' in the idwt pseudocode function in (15.4.1) in
        the VC-2 specification.
    """
    coeff_arrays = {}
    
    if dwt_depth_ho > 0:
        coeff_arrays[0] = {"L": make_array(0, "L")}
        for level in range(1, dwt_depth_ho + 1):
            coeff_arrays[level] = {"H": make_array(level, "H")}
    else:
        coeff_arrays[0] = {"LL": make_array(0, "LL")}
    
    for level in range(dwt_depth_ho + 1, dwt_depth + dwt_depth_ho + 1):
        coeff_arrays[level] = {
            "LH": make_array(level, "LH"),
            "HL": make_array(level, "HL"),
            "HH": make_array(level, "HH"),
        }
    
    return coeff_arrays


def make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho, prefix="coeff"):
    r"""
    Create a set of :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`\ s
    representing transform coefficient values, as expected by
    :py:func:`synthesis_transform`.
    
    Returns
    =======
    coeff_arrays : {level: {orientation: :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`, ...}, ...}
        The transform coefficient values. These dictionaries are indexed the
        same way as 'coeff_data' in the idwt pseudocode function in (15.4.1) in
        the VC-2 specification.
        
        The symbols will have the naming convention ``((prefix, level, orient),
        x, y)``
        where:
        
        * prefix is given by the 'prefix' argument
        * level is an integer giving the level number
        * orient is the transform orientation (one of "L", "H", "LL", "LH",
          "HL" or "HH").
        * x and y are the coordinate of the coefficient within that subband.
    """
    def make_array(level, orient):
        return SymbolArray(2, (prefix, level, orient))
    
    return make_coeff_arrays(dwt_depth, dwt_depth_ho, make_array)


def make_variable_coeff_arrays(dwt_depth, dwt_depth_ho, exp=Argument("coeffs")):
    r"""
    Create a set of :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`\ s
    representing transform coefficient values, as expected by
    :py:func:`synthesis_transform`.
    
    Returns
    =======
    coeff_arrays : {level: {orientation: :py:class:`~vc2_bit_widths.infinite_arrays.VariableArray`, ...}, ...}
        The transform coefficient values. These dictionaries are indexed the
        same way as 'coeff_data' in the idwt pseudocode function in (15.4.1) in
        the VC-2 specification.
        
        The expressions within the
        :py:class:`~vc2_bit_widths.infinite_arrays.VariableArray`\ s will be
        indexed as follows::
        
            >>> from vc2_bit_widths.pyexp import Argument
            
            >>> coeffs_arg = Argument("coeffs_arg")
            >>> coeff_arrays = make_variable_coeff_arrays(3, 0, coeffs_arg)
            >>> coeff_arrays[2]["LH"][3, 4] == coeffs_arg[2]["LH"][3, 4]
            True
    """
    def make_array(level, orient):
        return VariableArray(2, exp[level][orient])
    
    return make_coeff_arrays(dwt_depth, dwt_depth_ho, make_array)


def analysis_transform(h_filter_params, v_filter_params, dwt_depth, dwt_depth_ho, array):
    """
    Perform a multi-level VC-2 analysis Discrete Wavelet Transform (DWT) on a
    :py:class:`InfiniteArray` in a manner which is the complement of the 'idwt'
    pseudocode function described in (15.4.1) in the VC-2 standard.
    
    Parameters
    ==========
    h_filter_params, v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
        Horizontal and vertical filter parameters for the corresponding
        *synthesis* trhansform (e.g. from
        :py:data:`vc2_data_tables.LIFTING_FILTERS`). These filter parameters
        will be transformed into analysis lifting stages internally.
    dwt_depth, dwt_depth_ho: int
        Transform depths for 2D and horizontal-only transforms.
    array : :py:class:`InfiniteArray`
        The array representing the picture to be analysed.
    
    Returns
    =======
    coeff_arrays : {level: {orientation: :py:class:`InfiniteArray`, ...}, ...}
        The output transform coefficient values. These nested dictionaries are
        indexed the same way as 'coeff_data' in the idwt pseudocode function in
        (15.4.1) in the VC-2 specification.
    intermediate_arrays : {(level, array_name): :py:class:`InfiniteArray`, ...}
        All intermediate (and output) value arrays, named according to the
        convention described in :ref:`terminology`.
        
        This value is returned as an :py:class:`~collections.OrderedDict`
        giving the arrays in their order of creation; a sensible order for
        display purposes.
    """
    intermediate_arrays = OrderedDict()
    
    h_filter_params = convert_between_synthesis_and_analysis(h_filter_params)
    v_filter_params = convert_between_synthesis_and_analysis(v_filter_params)
    
    input = array
    for level in reversed(range(1, dwt_depth_ho + dwt_depth + 1)):
        intermediate_arrays[(level, "Input")] = input
        
        # Bit shift
        dc = intermediate_arrays[(level, "DC")] = LeftShiftedArray(
            input,
            h_filter_params.filter_bit_shift,
        )
        
        # Horizontal lifting stages
        for num, stage in enumerate(h_filter_params.stages):
            name = "DC{}".format("'"*(num + 1))
            dc = intermediate_arrays[(level, name)] = LiftedArray(dc, stage, 0)
        
        # Horizontal subsample
        l = intermediate_arrays[(level, "L")] = SubsampledArray(dc, (2, 1), (0, 0))
        h = intermediate_arrays[(level, "H")] = SubsampledArray(dc, (2, 1), (1, 0))
        
        if level > dwt_depth_ho:
            # Vertical lifting stages
            for num, stage in enumerate(v_filter_params.stages):
                name = "L{}".format("'"*(num + 1))
                l = intermediate_arrays[(level, name)] = LiftedArray(l, stage, 1)
                name = "H{}".format("'"*(num + 1))
                h = intermediate_arrays[(level, name)] = LiftedArray(h, stage, 1)
            
            # Vertical subsample
            ll = intermediate_arrays[(level, "LL")] = SubsampledArray(l, (1, 2), (0, 0))
            lh = intermediate_arrays[(level, "LH")] = SubsampledArray(l, (1, 2), (0, 1))
            hl = intermediate_arrays[(level, "HL")] = SubsampledArray(h, (1, 2), (0, 0))
            hh = intermediate_arrays[(level, "HH")] = SubsampledArray(h, (1, 2), (0, 1))
            
            input = ll
        else:
            input = l
    
    # Separately enumerate just the final output arrays
    coeff_arrays = {}
    for level in range(1, dwt_depth_ho + dwt_depth + 1):
        coeff_arrays[level] = {}
        if level > dwt_depth_ho:
            for orient in ["LH", "HL", "HH"]:
                coeff_arrays[level][orient] = intermediate_arrays[(level, orient)]
        else:
            coeff_arrays[level]["H"] = intermediate_arrays[(level, "H")]
    if dwt_depth_ho > 0:
        coeff_arrays[0] = {"L": input}
    else:
        coeff_arrays[0] = {"LL": input}
    
    return coeff_arrays, intermediate_arrays


def synthesis_transform(h_filter_params, v_filter_params, dwt_depth, dwt_depth_ho, coeff_arrays):
    """
    Perform a multi-level VC-2 synthesis Inverse Discrete Wavelet Transform
    (IDWT) on a :py:class:`InfiniteArray` in a manner equivalent to the 'idwt'
    pseudocode function in (15.4.1) of the VC-2 standard.
    
    Parameters
    ==========
    h_filter_params, v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
        Horizontal and vertical filter synthesis filter parameters (e.g. from
        :py:data:`vc2_data_tables.LIFTING_FILTERS`).
    dwt_depth, dwt_depth_ho: int
        Transform depths for 2D and horizontal-only transforms.
    coeff_arrays : {level: {orientation: :py:class:`InfiniteArray`, ...}, ...}
        The transform coefficient arrays to be used for synthesis. These nested
        dictionaries are indexed the same way as 'coeff_data' in the idwt
        pseudocode function in (15.4.1) in the VC-2 specification. See
        :py:func:`make_symbol_coeff_arrays` and
        :py:func:`make_variable_coeff_arrays`.
    
    Returns
    =======
    array : :py:class:`InfiniteArray`
        The final output array (i.e. decoded picture).
    intermediate_arrays : {(level, array_name): :py:class:`InfiniteArray`, ...}
        All intermediate (and output) value arrays, named according to the
        convention described in :ref:`terminology`.
        
        This value is returned as an :py:class:`~collections.OrderedDict`
        giving the arrays in their order of creation; a sensible order for
        display purposes.
    """
    intermediate_arrays = OrderedDict()
    
    if dwt_depth_ho > 0:
        output = coeff_arrays[0]["L"]
    else:
        output = coeff_arrays[0]["LL"]
    
    for level in range(1, dwt_depth_ho + dwt_depth + 1):
        if level > dwt_depth_ho:
            ll = intermediate_arrays[(level, "LL")] = output
            lh = intermediate_arrays[(level, "LH")] = coeff_arrays[level]["LH"]
            hl = intermediate_arrays[(level, "HL")] = coeff_arrays[level]["HL"]
            hh = intermediate_arrays[(level, "HH")] = coeff_arrays[level]["HH"]
            
            # Vertical interleave
            name = "L{}".format("'"*len(v_filter_params.stages))
            l = intermediate_arrays[(level, name)] = InterleavedArray(ll, lh, 1)
            name = "H{}".format("'"*len(v_filter_params.stages))
            h = intermediate_arrays[(level, name)] = InterleavedArray(hl, hh, 1)
            
            # Vertical lifting stages
            for num, stage in enumerate(v_filter_params.stages):
                name = "L{}".format("'"*(len(v_filter_params.stages) - num - 1))
                l = intermediate_arrays[(level, name)] = LiftedArray(l, stage, 1)
                name = "H{}".format("'"*(len(v_filter_params.stages) - num - 1))
                h = intermediate_arrays[(level, name)] = LiftedArray(h, stage, 1)
        else:
            l = intermediate_arrays[(level, "L")] = output
            h = intermediate_arrays[(level, "H")] = coeff_arrays[level]["H"]
        
        # Horizontal interleave
        name = "DC{}".format("'"*len(h_filter_params.stages))
        dc = intermediate_arrays[(level, name)] = InterleavedArray(l, h, 0)
        
        # Horizontal lifting stages
        for num, stage in enumerate(h_filter_params.stages):
            name = "DC{}".format("'"*(len(h_filter_params.stages) - num - 1))
            dc = intermediate_arrays[(level, name)] = LiftedArray(dc, stage, 0)
        
        # Bit shift
        output = intermediate_arrays[(level, "Output")] = RightShiftedArray(
            dc,
            h_filter_params.filter_bit_shift,
        )
    
    return output, intermediate_arrays


def add_missing_analysis_values(
    h_filter_params,
    v_filter_params,
    dwt_depth,
    dwt_depth_ho,
    analysis_values,
):
    """
    Fill in results for omitted (duplicate) filter arrays and phases.
    
    Parameters
    ==========
    h_filter_params, v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
    dwt_depth, dwt_depth_ho: int
        The filter parameters.
    analysis_values : {(level, array_name, x, y): value, ...}
        A dictionary of values associated with individual intermediate analysis
        filter phases with entries omitted where arrays are just
        interleavings/subsamplings/renamings.
    
    Returns
    =======
    full_analysis_values : {(level, array_name, x, y): value, ...}
        A new dictionary of values with missing filters and phases filled in.
    """
    # NB: Used only to enumerate the complete set of arrays and
    # get array periods
    _, intermediate_arrays = analysis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        SymbolArray(2),
    )
    
    out = OrderedDict()
    
    for (level, array_name), array in intermediate_arrays.items():
        for x in range(array.period[0]):
            for y in range(array.period[1]):
                # Below we work out which source array to use to populate the
                # current array/phase
                src_level = level
                src_array_name = array_name
                src_x = x
                src_y = y
                if (level, array_name, x, y) not in analysis_values:
                    if array_name == "Input":
                        src_level = level + 1
                        src_array_name = (
                            "LL"
                            if (src_level, "LL") in intermediate_arrays else
                            "L"
                        )
                    elif array_name == "DC":
                        src_array_name = "Input"
                    elif array_name in ("L", "H"):
                        src_array_name = [
                            int_array_name
                            for int_level, int_array_name in intermediate_arrays
                            if int_level == src_level and int_array_name.startswith("DC'")
                        ][-1]
                        if array_name == "L":
                            src_x = x * 2
                        elif array_name == "H":
                            src_x = (x * 2) + 1
                    elif array_name in ("LL", "LH"):
                        src_array_name = [
                            int_array_name
                            for int_level, int_array_name in intermediate_arrays
                            if int_level == src_level and int_array_name.startswith("L'")
                        ][-1]
                        if array_name == "LL":
                            src_y = y * 2
                        elif array_name == "LH":
                            src_y = (y * 2) + 1
                    elif array_name in ("HL", "HH"):
                        src_array_name = [
                            int_array_name
                            for int_level, int_array_name in intermediate_arrays
                            if int_level == src_level and int_array_name.startswith("H'")
                        ][-1]
                        if array_name == "HL":
                            src_y = y * 2
                        elif array_name == "HH":
                            src_y = (y * 2) + 1
                    else:
                        # Should never reach this point so long as only
                        # nops are omitted
                        assert False
                
                out[(level, array_name, x, y)] = analysis_values.get((
                    src_level,
                    src_array_name,
                    src_x,
                    src_y,
                ), out.get((
                    src_level,
                    src_array_name,
                    src_x,
                    src_y,
                )))
    
    return out


def add_missing_synthesis_values(
    h_filter_params,
    v_filter_params,
    dwt_depth,
    dwt_depth_ho,
    synthesis_values,
    fill_in_equivalent_phases=True,
):
    r"""
    Fill in results for omitted (duplicate) filter arrays and phases.
    
    Parameters
    ==========
    h_filter_params, v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
    dwt_depth, dwt_depth_ho: int
        The filter parameters.
    synthesis_values : {(level, array_name, x, y): value, ...}
        A dictionary of values associated with individual intermediate
        synthesis filter phases with entries omitted where arrays are just
        interleavings/subsamplings/renamings.
    fill_in_equivalent_phases : bool
        When two :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`\ s
        with different periods are interleaved, the interleaved signal's period
        will include repetitions of some phases of one of the input arrays. For
        example, consider the following arrays::
        
             a = ... a0 a1 a0 a1 ...    (Period = 2)
             b = ... b0 b0 b0 b0 ...    (Period = 1)
            
            ab = ... a0 b0 a1 b0 ...    (Period = 4)
        
        In this example, the interleaved array has a period of 4 with two
        phases coming from a and two from b. Since b has a period of one,
        however, one of the phases of ab contains a repeat of one of the phases
        of 'b'.
        
        Since in a de-duplicated set of filters and phases, duplicate phases
        appearing in interleaved arrays are not present, some other value must
        be used when filling in these phases. If the
        'fill_in_equivalent_phases' argument is True (the default), the value
        from an equivalent phase will be copied in. If False, None will be used
        instead.
        
        Where the dictionary being filled in contains results generic to the
        phase being used (and not the specific filter coordinates), the default
        value of 'True' will give the desired results.
    
    Returns
    =======
    full_synthesis_values : {(level, array_name, x, y): value, ...}
        A new dictionary of values with missing filters and phases filled in.
    """
    # NB: Used only to enumerate the complete set of arrays and
    # get array periods
    _, intermediate_arrays = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        make_variable_coeff_arrays(dwt_depth, dwt_depth_ho),
    )
    
    out = OrderedDict()
    
    for (level, array_name), array in intermediate_arrays.items():
        for x in range(array.period[0]):
            for y in range(array.period[1]):
                # Below we work out which source array to use to populate the
                # current array/phase
                src_level = level
                src_array_name = array_name
                src_x = x
                src_y = y
                if (level, array_name, x, y) not in synthesis_values:
                    if array_name.startswith("L'"):
                        if y % 2 == 0:
                            src_array_name = "LL"
                        else:
                            src_array_name = "LH"
                        src_y = y // 2
                    elif array_name.startswith("H'"):
                        if y % 2 == 0:
                            src_array_name = "HL"
                        else:
                            src_array_name = "HH"
                        src_y = y // 2
                    elif array_name.startswith("DC'"):
                        if x % 2 == 0:
                            src_array_name = "L"
                        else:
                            src_array_name = "H"
                        src_x = x // 2
                    elif array_name == "Output":
                        src_array_name = "DC"
                    elif array_name == "LL" or array_name == "L":
                        src_level = level - 1
                        src_array_name = "Output"
                    else:
                        # Should never reach this point so long as only
                        # nops are omitted
                        assert False
                
                if fill_in_equivalent_phases:
                    px, py = intermediate_arrays[(src_level, src_array_name)].period
                    src_x %= px
                    src_y %= py
                
                out[(level, array_name, x, y)] = synthesis_values.get((
                    src_level,
                    src_array_name,
                    src_x,
                    src_y,
                ), out.get((
                    src_level,
                    src_array_name,
                    src_x,
                    src_y,
                )))
    
    return out
