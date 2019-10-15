r"""
Find Bounds for Filter Output Values
====================================

This module contains functions for computing theoretical lower- and
upper-bounds for VC-2 codec intermediate and final values. These bounds are
based on :py:mod:`~vc2_bit_widths.affine_arithmetic` models of rounding and
quantisation errors. As a consequence, the computed ranges will over-estimate
the true range of values possible but are guaranteed to not be an
under-estimate.

The following functions compute :py:class:`~vc2_bit_widths.linexp.LinExp`\ s
which give the upper and lower bounds of a particular filter expression in
terms of input picture signal ranges or transform coefficient ranges.

.. autofunction:: analysis_filter_bounds

.. autofunction:: synthesis_filter_bounds

The following utility functions may be used to substitute concrete numbers into
the output of the above to get concrete lower and upper bounds.

.. autofunction:: evaluate_analysis_filter_bounds

.. autofunction:: evaluate_synthesis_filter_bounds

Finally the following utility function may be used to find the number of bits
required to hold a particular value:

.. autofunction:: twos_compliment_bits

.. autofunction:: signed_integer_range

"""

import math

from vc2_bit_widths.linexp import (
    LinExp,
    AAError,
    affine_lower_bound,
    affine_upper_bound,
)


def round_away_from_zero(value):
    """Round a number away from zero."""
    if value < 0:
        return math.floor(value)
    else:
        return math.ceil(value)


def signed_integer_range(num_bits):
    """
    Return the lower- and upper-bound values for a signed integer with the
    specified number of bits.
    """
    return (
        -1 << (num_bits - 1),
        (1 << (num_bits - 1)) - 1,
    )


def twos_compliment_bits(value):
    """
    How many bits does the provided integer require for a two's compliment
    (signed) integer representation?
    """
    if value < 0:
        value = -value - 1
    return value.bit_length() + 1


def analysis_filter_bounds(expression):
    """
    Find the lower- and upper bound reachable in a
    :py:class:`~vc2_bit_widths.linexp.LinExp` containing an analysis filter
    expression.
    
    Parameters
    ==========
    expression : :py:class:`~vc2_bit_widths.linexp.LinExp`
        All symbols which are not :py:mod:`~vc2_bit_widths.affine_arithmetic`
        error terms will be treated as picture values in the specified range.
    picture_lower_bound, picture_upper_bound : (lower, upper)
        The lower and upper bounds for the values of pixels in the input
        picture..
    
    Returns
    =======
    lower_bound : :py:class:`~vc2_bit_widths.linexp.LinExp`
    upper_bound : :py:class:`~vc2_bit_widths.linexp.LinExp`
        Lower and upper bounds for the signal level in terms of the symbols
        ``LinExp("signal_min")`` and ``LinExp("signal_max")``, representing the
        minimum and maximum signal picture levels respectively.
    """
    signal_min = LinExp("signal_min")
    signal_max = LinExp("signal_max")
    
    expression = LinExp(expression)
    
    lower_bound = affine_lower_bound(expression.subs({
        sym: signal_min if coeff > 0 else signal_max
        for sym, coeff in expression
        if sym is not None and not isinstance(sym, AAError)
    }))
    
    upper_bound = affine_upper_bound(expression.subs({
        sym: signal_min if coeff < 0 else signal_max
        for sym, coeff in expression
        if sym is not None and not isinstance(sym, AAError)
    }))
    
    return (lower_bound, upper_bound)


def evaluate_analysis_filter_bounds(lower_bound_exp, upper_bound_exp, num_bits):
    """
    Convert a pair of symbolic analysis filter bounds into concrete, numerical
    lower/upper bounds for an input picture signal with the specified number of
    bits.
    
    Parameters
    ==========
    lower_bound_exp : :py:class:`~vc2_bit_widths.linexp.LinExp`
    upper_bound_exp : :py:class:`~vc2_bit_widths.linexp.LinExp`
        Analysis filter bounds expressions from
        :py:func:`analysis_filter_bounds`.
    num_bits : int
        The number of bits in the signal values being analysed.
    
    Returns
    =======
    lower_bound : int
    upper_bound : int
        The concrete (numerical) bounds for the analysis filter given
        ``num_bits`` input pictures.
    """
    signal_min, signal_max = signed_integer_range(num_bits)
    signal_range = {
        "signal_min": signal_min,
        "signal_max": signal_max,
    }
    
    lower_bound = round_away_from_zero(lower_bound_exp.subs(signal_range).constant)
    upper_bound = round_away_from_zero(upper_bound_exp.subs(signal_range).constant)
    
    return (lower_bound, upper_bound)


def synthesis_filter_bounds(expression):
    """
    Find the lower- and upper bound reachable in a
    :py:class:`~vc2_bit_widths.linexp.LinExp` containing a synthesis filter
    expression.
    
    Parameters
    ==========
    expression : :py:class:`~vc2_bit_widths.linexp.LinExp`
        All symbols which are not :py:mod:`~vc2_bit_widths.affine_arithmetic`
        error terms will be treated as transform coefficient values in the
        specified range. Coefficients are assumed to be named as per the
        convention used by
        :py:func:`vc2_bit_widths.vc2_filters.analysis_transform` and
        :py:func:`vc2_bit_widths.vc2_filters.make_symbol_coeff_arrays`.
    
    Returns
    =======
    (lower_bound, upper_bound) : :py:class:`~vc2_bit_widths.linexp.LinExp`
        Lower and upper bounds for the signal level in terms of the symbols of
        the form ``LinExp("signal_LEVEL_ORIENT_min")`` and
        ``LinExp("signal_LEVEL_ORIENT_max")``, representing the minimum and
        maximum signal levels for transform coefficients in level ``LEVEL`` and
        orientation ``ORIENT`` respectively.
    """
    # Replace all transform coefficients with affine error terms scaled to
    # appropriate ranges
    expression = LinExp(expression)
    
    lower_bound = affine_lower_bound(expression.subs({
        sym: ("coeff_{}_{}_min" if coeff > 0 else "coeff_{}_{}_max").format(
            sym[0][1],
            sym[0][2],
        )
        for sym, coeff in expression
        if sym is not None and not isinstance(sym, AAError)
    }))
    
    upper_bound = affine_upper_bound(expression.subs({
        sym: ("coeff_{}_{}_min" if coeff < 0 else "coeff_{}_{}_max").format(
            sym[0][1],
            sym[0][2],
        )
        for sym, coeff in expression
        if sym is not None and not isinstance(sym, AAError)
    }))
    
    return (lower_bound, upper_bound)


def evaluate_synthesis_filter_bounds(lower_bound_exp, upper_bound_exp, coeff_bounds):
    """
    Convert a pair of symbolic synthesis filter bounds into concrete, numerical
    lower/upper bounds given the bounds of the input transform coefficients.
    
    Parameters
    ==========
    lower_bound_exp : :py:class:`~vc2_bit_widths.linexp.LinExp`
    upper_bound_exp : :py:class:`~vc2_bit_widths.linexp.LinExp`
        Synthesis filter bounds expressions from
        :py:func:`synthesis_filter_bounds`.
    coeff_bounds : {(level, orient): (lower_bound, upper_bound), ...}
        For each transform coefficient, the concrete lower and upper bounds
        (e.g. as computed using :py:func:`evaluate_analysis_filter_bounds` and
        :py:func:`vc2_bit_widths.quantisation.maximum_dequantised_magnitude`
        for each transform subband).
        
        As a convenience, the DC band may be provided as either (0, "L")/(0,
        "LL"), as is convention in the main VC-2 specification, or (1, "L")/(1,
        "LL"), as is convention in the bit width guide tables.
    
    Returns
    =======
    lower_bound : int
    upper_bound : int
        The concrete (numerical) bounds for the synthesis filter.
    """
    coeff_bounds = coeff_bounds.copy()
    if (1, "L") in coeff_bounds:
        coeff_bounds[(0, "L")] = coeff_bounds.pop((1, "L"))
    if (1, "LL") in coeff_bounds:
        coeff_bounds[(0, "LL")] = coeff_bounds.pop((1, "LL"))
    
    signal_range = {
        "coeff_{}_{}_{}".format(level, orient, minmax): value
        for (level, orient), bounds in coeff_bounds.items()
        for minmax, value in zip(["min", "max"], bounds)
    }
    
    lower_bound = round_away_from_zero(lower_bound_exp.subs(signal_range).constant)
    upper_bound = round_away_from_zero(upper_bound_exp.subs(signal_range).constant)
    
    return (lower_bound, upper_bound)
