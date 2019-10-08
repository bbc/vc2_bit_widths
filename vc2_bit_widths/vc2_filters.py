r"""
VC-2 Wavelet Filters Implemented as :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`\ s
================================================================================================

This module provides an implementation of the VC-2 wavelet filtering process in
terms of :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`\ s.

By using :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`\ s as inputs,
algebraic descriptions (using :py:class:`~vc2_bit_widths.linexp.LinExp`) of the
VC-2 filters may be assembled. From these analyses may be performed to
determine, for example, signal ranges and rounding error bounds.

Using :py:class:`~vc2_bit_widths.infinite_arrays.VariableArray`\ s as inputs,
Python functions may be generated (using :py:mod:`~vc2_bit_widths.pyexp`) which
efficiently compute single filter outputs or intermediate values.

.. autofunction:: analysis_transform

.. autofunction:: synthesis_transform

.. autofunction:: make_coeff_arrays

.. autofunction:: make_symbol_coeff_arrays

.. autofunction:: make_variable_coeff_arrays

"""

__all__ = [
    "analysis_transform",
    "synthesis_transform",
    "make_coeff_arrays",
    "make_symbol_coeff_arrays",
    "make_variable_coeff_arrays",
]

import math

from collections import defaultdict

from vc2_conformance.decoder.transform_data_syntax import quant_factor

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
    coefficient values, as expected by :py:func:`idwt`.
    
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
    representing transform coefficient values, as expected by :py:func:`idwt`.
    
    Returns
    =======
    coeff_arrays : {level: {orientation: :py:class:`SymbolArray`, ...}, ...}
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
    representing transform coefficient values, as expected by :py:func:`idwt`.
    
    Returns
    =======
    coeff_arrays : {level: {orientation: :py:class:`VariableArray`, ...}, ...}
        The transform coefficient values. These dictionaries are indexed the
        same way as 'coeff_data' in the idwt pseudocode function in (15.4.1) in
        the VC-2 specification.
        
        The expressions within the :py:class:`VariableArrays`` will be indexed
        as follows::
        
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
    Perform a multi-level VC-2 (analysis) Discrete Wavelet Transform (DWT) on a
    :py:class:`InfiniteArray` in a manner which is the complement of the 'idwt'
    pseudocode function in (15.4.1).
    
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
        The picture to be analysed.
    
    Returns
    =======
    coeff_arrays : {level: {orientation: :py:class:`InfiniteArray`, ...}, ...}
        The output transform coefficient values. These nested dictionaries are
        indexed the same way as 'coeff_data' in the idwt pseudocode function in
        (15.4.1) in the VC-2 specification.
    intermediate_values : {(level, array_name): :py:class:`InfiniteArray`, ...}
        All intermediate (and output) values.
        
        The analysis transform consists of a series of transform levels. During
        each level, a series of lifting filter stages, bit shifts and
        sub-samplings take place, illustrated below for the case where both
        horizontal and vertical wavelets have two lifting stages each.
        
        .. image:: /_static/bit_widths/intermediate_analysis_values.svg
            :alt: A single analysis level is shown as a pipeline: bit shift, horizontal lifting stages, horizontal subsampling, vertical lifting stages, vertical subsampling.
        
        Each of these intermediate transform results are added to the
        'intermediate_values' dictionary. The 'level' part of the dictionary
        key is an integer (running from ``dwt_depth_ho + dwt_depth`` for the
        first level to be computed during analysis to ``1``, the final level).
        The 'array_name' part of the key is a string giving a name following
        the naming convention shown in the figure above.
    """
    intermediate_values = {}
    
    h_filter_params = convert_between_synthesis_and_analysis(h_filter_params)
    v_filter_params = convert_between_synthesis_and_analysis(v_filter_params)
    
    input = array
    for level in reversed(range(1, dwt_depth_ho + dwt_depth + 1)):
        intermediate_values[(level, "Input")] = input
        
        # Bit shift
        dc = intermediate_values[(level, "DC")] = LeftShiftedArray(
            input,
            h_filter_params.filter_bit_shift,
        )
        
        # Horizontal lifting stages
        for num, stage in enumerate(h_filter_params.stages):
            name = "DC{}".format("'"*(num + 1))
            dc = intermediate_values[(level, name)] = LiftedArray(dc, stage, 0)
        
        # Horizontal subsample
        l = intermediate_values[(level, "L")] = SubsampledArray(dc, (2, 1), (0, 0))
        h = intermediate_values[(level, "H")] = SubsampledArray(dc, (2, 1), (1, 0))
        
        if level > dwt_depth_ho:
            # Vertical lifting stages
            for num, stage in enumerate(v_filter_params.stages):
                name = "L{}".format("'"*(num + 1))
                l = intermediate_values[(level, name)] = LiftedArray(l, stage, 1)
                name = "H{}".format("'"*(num + 1))
                h = intermediate_values[(level, name)] = LiftedArray(h, stage, 1)
            
            # Vertical subsample
            ll = intermediate_values[(level, "LL")] = SubsampledArray(l, (1, 2), (0, 0))
            lh = intermediate_values[(level, "LH")] = SubsampledArray(l, (1, 2), (0, 1))
            hl = intermediate_values[(level, "HL")] = SubsampledArray(h, (1, 2), (0, 0))
            hh = intermediate_values[(level, "HH")] = SubsampledArray(h, (1, 2), (0, 1))
            
            input = ll
        else:
            input = l
    
    # Separately enumerate just the final output arrays
    coeff_arrays = {}
    for level in range(1, dwt_depth_ho + dwt_depth + 1):
        coeff_arrays[level] = {}
        if level > dwt_depth_ho:
            for orient in ["LH", "HL", "HH"]:
                coeff_arrays[level][orient] = intermediate_values[(level, orient)]
        else:
            coeff_arrays[level]["H"] = intermediate_values[(level, "H")]
    if dwt_depth_ho > 0:
        coeff_arrays[0] = {"L": input}
    else:
        coeff_arrays[0] = {"LL": input}
    
    return coeff_arrays, intermediate_values


def synthesis_transform(h_filter_params, v_filter_params, dwt_depth, dwt_depth_ho, coeff_arrays):
    """
    Perform a multi-level VC-2 (synthesis) Inverse Discrete Wavelet Transform
    (IDWT) on a :py:class:`InfiniteArray` in a manner equivalent
    of the 'idwt' pseudocode function in (15.4.1).
    
    Parameters
    ==========
    h_filter_params, v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
        Horizontal and vertical filter synthesis filter parameters (e.g. from
        :py:data:`vc2_data_tables.LIFTING_FILTERS`).
    dwt_depth, dwt_depth_ho: int
        Transform depths for 2D and horizontal-only transforms.
    coeff_arrays : {level: {orientation: :py:class:`InfiniteArray`, ...}, ...}
        The transform coefficient values to be used for synthesis. These nested
        dictionaries are indexed the same way as 'coeff_data' in the idwt
        pseudocode function in (15.4.1) in the VC-2 specification. The
        :py:func:`make_coeff_arrays` function (or one of its derivatives) may
        be used to quickly construct values for this argument.
    
    Returns
    =======
    array : :py:class:`InfiniteArray`
        The final output value (i.e. decoded picture).
    intermediate_values : {(level, array_name): :py:class:`InfiniteArray`, ...}
        All intermediate (and output) values.
        
        The analysis transform consists of a series of transform levels. During
        each level, a series of lifting filter stages, bit shifts and
        sub-samplings take place, illustrated below for the case where both
        horizontal and vertical wavelets have two lifting stages each.
        
        .. image:: /_static/bit_widths/intermediate_synthesis_values.svg
            :alt: A single analysis level is shown as a pipeline: bit shift, horizontal lifting stages, horizontal subsampling, vertical lifting stages, vertical subsampling.
        
        Each of these intermediate transform results are added to the
        'intermediate_values' dictionary. The 'level' part of the dictionary
        key is an integer (running from ``dwt_depth_ho + dwt_depth`` for the
        first level to be computed during analysis to ``1``, the final level).
        The 'array_name' part of the key is a string giving a name following
        the naming convention shown in the figure above.
    """
    intermediate_values = {}
    
    if dwt_depth_ho > 0:
        output = coeff_arrays[0]["L"]
    else:
        output = coeff_arrays[0]["LL"]
    
    for level in range(1, dwt_depth_ho + dwt_depth + 1):
        if level > dwt_depth_ho:
            ll = intermediate_values[(level, "LL")] = output
            lh = intermediate_values[(level, "LH")] = coeff_arrays[level]["LH"]
            hl = intermediate_values[(level, "HL")] = coeff_arrays[level]["HL"]
            hh = intermediate_values[(level, "HH")] = coeff_arrays[level]["HH"]
            
            # Vertical interleave
            name = "L{}".format("'"*len(v_filter_params.stages))
            l = intermediate_values[(level, name)] = InterleavedArray(ll, lh, 1)
            name = "H{}".format("'"*len(v_filter_params.stages))
            h = intermediate_values[(level, name)] = InterleavedArray(hl, hh, 1)
            
            # Vertical lifting stages
            for num, stage in enumerate(v_filter_params.stages):
                name = "L{}".format("'"*(len(v_filter_params.stages) - num - 1))
                l = intermediate_values[(level, name)] = LiftedArray(l, stage, 1)
                name = "H{}".format("'"*(len(v_filter_params.stages) - num - 1))
                h = intermediate_values[(level, name)] = LiftedArray(h, stage, 1)
        else:
            l = intermediate_values[(level, "L")] = output
            h = intermediate_values[(level, "H")] = coeff_arrays[level]["H"]
        
        # Horizontal interleave
        name = "DC{}".format("'"*len(h_filter_params.stages))
        dc = intermediate_values[(level, name)] = InterleavedArray(l, h, 0)
        
        # Horizontal lifting stages
        for num, stage in enumerate(h_filter_params.stages):
            name = "DC{}".format("'"*(len(h_filter_params.stages) - num - 1))
            dc = intermediate_values[(level, name)] = LiftedArray(dc, stage, 0)
        
        # Bit shift
        output = intermediate_values[(level, "Output")] = RightShiftedArray(
            dc,
            h_filter_params.filter_bit_shift,
        )
    
    return output, intermediate_values
