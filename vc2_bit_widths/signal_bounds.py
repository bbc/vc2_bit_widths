r"""
:py:mod:`vc2_bit_widths.signal_bounds`: Finding bounds for filter output values
===============================================================================

The :py:mod:`vc2_bit_widths.signal_bounds` module contains functions for
computing theoretical lower- and upper-bounds for VC-2 codec intermediate and
final values.

Finding signal bounds
---------------------

The following functions may be used to convert algebraic descriptions of a VC-2
filters into worst-case signal ranges (according to an affine arithmetic based
model of rounding and quantisation).

The process of finding signal bounds is split into two stages:

1. Finding a generic algebraic expression for worst-case signal bounds.
2. Evaluating those expressions for a particular picture bit width.

The two steps are split this way to allow step 2 to be inexpensively repeated
for different picture bit widths.

Analysis filter
```````````````

.. autofunction:: analysis_filter_bounds

.. autofunction:: evaluate_analysis_filter_bounds

Synthesis filter
````````````````

.. autofunction:: synthesis_filter_bounds

.. autofunction:: evaluate_synthesis_filter_bounds


Integer representation utilities
--------------------------------

The following utility functions compute the relationship between bit-width and
numerical range.

.. autofunction:: twos_compliment_bits

.. autofunction:: signed_integer_range

.. autofunction:: unsigned_integer_range

The following function may be used to pessimistically round values to integers:

.. autofunction:: round_away_from_zero

"""

import math

from vc2_bit_widths.linexp import (
    LinExp,
    AAError,
    affine_lower_bound,
    affine_upper_bound,
)

from vc2_bit_widths.quantisation import maximum_useful_quantisation_index


def round_away_from_zero(value):
    """Round a number away from zero."""
    if value < 0:
        return int(math.floor(value))
    else:
        return int(math.ceil(value))


def signed_integer_range(num_bits):
    """
    Return the lower- and upper-bound values for a signed integer with the
    specified number of bits.
    """
    return (
        -1 << (num_bits - 1),
        (1 << (num_bits - 1)) - 1,
    )


def unsigned_integer_range(num_bits):
    """
    Return the lower- and upper-bound values for an unsigned integer with the
    specified number of bits.
    """
    return (
        0,
        (1 << num_bits) - 1,
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
    Find the lower- and upper-bound reachable in a
    :py:class:`~vc2_bit_widths.linexp.LinExp` describing an analysis filter.
    
    The filter expression must consist of only affine error symbols
    (:py:class:`~vc2_bit_widths.linexp.AAError`) and symbols of the form ``(_,
    x, y)`` representing pixel values in an input picture.
    
    Parameters
    ==========
    expression : :py:class:`~vc2_bit_widths.linexp.LinExp`
    
    Returns
    =======
    lower_bound : :py:class:`~vc2_bit_widths.linexp.LinExp`
    upper_bound : :py:class:`~vc2_bit_widths.linexp.LinExp`
        Algebraic expressions for the lower and upper bounds for the signal
        level. These expressions are given in terms of the symbols
        ``LinExp("signal_min")`` and ``LinExp("signal_max")``, which represent
        the minimum and maximum picture signal levels respectively.
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
    Find the lower- and upper-bound reachable in a
    :py:class:`~vc2_bit_widths.linexp.LinExp` describing a synthesis filter.
    
    The filter expression must contain only affine error symbols
    (:py:class:`~vc2_bit_widths.linexp.AAError`) and symbols of the form ``((_,
    level, orient), x, y)`` representing transform coefficients.
    
    Parameters
    ==========
    expression : :py:class:`~vc2_bit_widths.linexp.LinExp`
    
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
    
    Returns
    =======
    lower_bound : int
    upper_bound : int
        The concrete (numerical) bounds for the synthesis filter.
    """
    signal_range = {
        "coeff_{}_{}_{}".format(level, orient, minmax): value
        for (level, orient), bounds in coeff_bounds.items()
        for minmax, value in zip(["min", "max"], bounds)
    }
    
    lower_bound = round_away_from_zero(lower_bound_exp.subs(signal_range).constant)
    upper_bound = round_away_from_zero(upper_bound_exp.subs(signal_range).constant)
    
    return (lower_bound, upper_bound)

