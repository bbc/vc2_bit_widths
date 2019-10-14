"""
Find Bounds for Filter Output Values
====================================

This module contains functions for computing theoretical lower- and
upper-bounds for VC-2 codec intermediate and final values. These bounds are
based on :py:mod:`~vc2_bit_widths.affine_arithmetic` models of rounding and
quantisation errors. As a consequence, the computed ranges will over-estimate
the true range of values possible but are guaranteed to not be an
under-estimate.

TODO: Describe how quantisation is modelled (i.e. as independent).

.. autofunction:: analysis_filter_bounds

.. autofunction:: synthesis_filter_bounds

.. autofunction:: parse_coeff_symbol_name

"""

from vc2_bit_widths.linexp import (
    LinExp,
    AAError,
    affine_lower_bound,
    affine_upper_bound,
)


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
    (lower_bound, upper_bound) : :py:class:`~vc2_bit_widths.linexp.LinExp`
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
        orientation ``ORIENT`` respectively. See also
        :py:func:`parse_coeff_symbol_name`.
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


def parse_coeff_symbol_name(symbol_name):
    """
    Extract the fields from a symbol name produced by
    :py:func:`synthesis_filter_bounds`.
    
    Parameters
    ==========
    symbol_name : str
        A symbol name from :py:class:`synthesis_filter_bounds`.
    
    Returns
    =======
    (level, orient, minmax)
        The level (int), orientation (str) and "min" or "max" extracted from
        the symbol.
    """
    _, level, orient, minmax = symbol_name.split("_")
    return (int(level), orient, minmax)
