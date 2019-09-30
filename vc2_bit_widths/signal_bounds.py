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

.. autoclass:: PostQuantisationTransformCoeffBoundsLookup

"""

from vc2_bit_widths.linexp import LinExp

import vc2_bit_widths.affine_arithmetic as aa

from vc2_bit_widths.quantisation import maximum_dequantised_magnitude


def analysis_filter_bounds(expression, picture_lower_bound, picture_upper_bound):
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
    (lower_bound, upper_bound) : numbers
    """
    # Replace all picture coefficients with affine error terms scaled to
    # appropriate ranges
    expression = LinExp(expression)
    expression = expression.subs({
        sym: aa.error_in_range(picture_lower_bound, picture_upper_bound)
        for sym in expression.symbols()
        if sym is not None and not isinstance(sym, aa.Error)
    })
    
    # Find bounds using affine arithmetic
    return (
        aa.lower_bound(expression).constant,
        aa.upper_bound(expression).constant,
    )


class PostQuantisationTransformCoeffBoundsLookup(object):

    def __init__(self, coeff_arrays, picture_lower_bound, picture_upper_bound):
        """
        Given the output transform coefficient arrays from an analysis filter
        (e.g. from :py:func:`vc2_bit_widths.vc2_filters.analysis_transform`),
        provides a lookup for lower- and upper-bounds for these transform
        coefficient values. These bounds also account for worst-case
        quantisation errors.
        
        This object is essentially a thin wrapper for calling
        :py:func:`analysis_filter_bounds` and
        :py:func:`~vc2_bit_widths.quantisation.maximum_dequantised_magnitude`
        but also includes some simple caching logic to speed up repeated
        accesses. For example::
        
            >>> coeff_arrays = analysis_transform(...)
            
            >>> lookup = TransformCoeffBoundsLookup(coeff_arrays, -512, +511)
            >>> lookup_lower_bound, lookup_upper_bound = lookup[2, "LH", 10, 3]
            
            >>> manual_lower_bound, manual_upper_bound = map(
            ...     maximum_dequantised_magnitude,
            ...     analysis_filter_bounds(
            ...         coeff_arrays[2]["LH"][10, 3],
            ...         -512,
            ...         +511,
            ...     ),
            ... )

            >>> lookup_lower_bound == manual_lower_bound
            True
            >>> lookup_upper_bound == manual_upper_bound
            True
        
        Parameters
        ==========
        coeff_arrays : {level: {orientation: :py:class:`InfiniteArray`, ...}, ...}
            The transform coefficient values. These dictionaries are indexed
            the same way as 'coeff_data' in the idwt pseudocode function in
            (15.4.1) in the VC-2 specification. As returned by
            :py:func:`vc2_bit_widths.vc2_filters.analysis_transform`.  All
            symbols which are not :py:mod:`~vc2_bit_widths.affine_arithmetic`
            error terms will be treated as picture values in the specified
            range.
        picture_lower_bound, picture_upper_bound : (lower, upper)
            The lower and upper bounds for the values of pixels in the input
            picture.
        """
        
        self._coeff_arrays = coeff_arrays
        self._picture_lower_bound = picture_lower_bound
        self._picture_upper_bound = picture_upper_bound
        
        # A cache of previously computed value bounds. Coordinates are
        # noramlised to the first period of the relevant array.
        #
        # {(level, orient, normalised_x, normalised_y): (lower, upper), ...}
        self._cache = {}
    
    def __getitem__(self, key):
        level, orient, x, y = key
        
        array = self._coeff_arrays[level][orient]
        
        normalised_x = x % array.period[0]
        normalised_y = y % array.period[1]
        
        cache_key = (level, orient, normalised_x, normalised_y)
        
        if cache_key not in self._cache:
            lower_bound, upper_bound = analysis_filter_bounds(
                array[normalised_x, normalised_y],
                self._picture_lower_bound,
                self._picture_upper_bound,
            )
            
            lower_bound = maximum_dequantised_magnitude(lower_bound)
            upper_bound = maximum_dequantised_magnitude(upper_bound)
            
            self._cache[cache_key] = (lower_bound, upper_bound)
        
        return self._cache[cache_key]


def synthesis_filter_bounds(expression, coeff_value_ranges):
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
        :py:func:`vc2_bit_widths.vc2_filters.make_coeff_arrays`.
    coeff_value_ranges : :py:class:`PostQuantisationTransformCoeffBoundsLookup`
        A lookup giving the lower- and upper-bounds of all transform
        coefficients. The lookup should support keys of the form (level,
        orient, x, y) and contain values of the form (lower_bound,
        upper_bound).
    
    Returns
    =======
    (lower_bound, upper_bound)
    """
    # Replace all transform coefficients with affine error terms scaled to
    # appropriate ranges
    expression = LinExp(expression)
    expression = expression.subs({
        sym: aa.error_in_range(*coeff_value_ranges[
            sym[0][1],
            sym[0][2],
            sym[1],
            sym[2],
        ])
        for sym in expression.symbols()
        if sym is not None and not isinstance(sym, aa.Error)
    })
    
    # Find bounds using affine arithmetic
    return (
        aa.lower_bound(expression).constant,
        aa.upper_bound(expression).constant,
    )
