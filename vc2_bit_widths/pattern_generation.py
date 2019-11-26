"""
:py:mod:`vc2_bit_widths.pattern_generation`: Heuristic test pattern generation
==============================================================================

The routines in this module implement heuristcs for producing test patterns for
VC-2 filters which produce near-maximum magnitude values in a target
intermediate or final value.

Due to non-linearities in VC-2's filters (i.e. rounding and quantisation), the
test patterns generated are not true worst-case signals but rather a 'best
effort' to get close to the worst case. Analysis test patterns will tend to be
very close to worst-case signals while synthesis signals are likely to be
modest under-estimates. Nevertheless, these signals often reach values well
above real pictures and noise.

Test pattern generators
-----------------------

Analysis
````````

.. autofunction:: make_analysis_maximising_pattern

Synthesis
`````````

.. autofunction:: make_synthesis_maximising_pattern

"""

from collections import namedtuple

from vc2_bit_widths.patterns import TestPattern, TestPatternSpecification

from vc2_bit_widths.linexp import (
    LinExp,
    AAError,
)



def get_maximising_inputs(expression):
    """
    Find the symbol value assignment which maximises the provided expression.
    
    Parameters
    ==========
    expression : :py:class:`~vc2_bit_widths.linexp.LinExp`
        The expression whose value is to be maximised.
    
    Returns
    =======
    maximising_symbol_assignment : {sym: +1 or -1, ...}
        A dictionary giving the polarity of a maximising assignment to each
        non-:py:class:`~vc2_bit_widths.affine_arithmetic.Error` term in the
        expression.
    """
    return {
        sym: +1 if coeff > 0 else -1
        for sym, coeff in LinExp(expression)
        if sym is not None and not isinstance(sym, AAError)
    }


def make_analysis_maximising_pattern(input_array, target_array, tx, ty):
    """
    Create a test pattern which maximises a value within an intermediate/final
    output of an analysis filter.
    
    .. note::
        
        In lossless coding modes, test patterns which maximise a given value in
        the encoder also maximise the corresponding value in the decoder.
        Consequently this function may also be used to (indirectly) produce
        lossless decoder test patterns.
    
    .. warning::
        
        The returned test pattern is designed to maximise a real-valued
        implementation of the target filter. Though it is likely that this
        signal also maximises integer-based implementations (such as those used
        by VC-2) it is not guaranteed.
    
    Parameters
    ==========
    input_array : :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`
        The input array to the analysis filter.
    target_array : :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`
        An intermediate or final output array produced by the analysis filter
        within which a value should be maximised.
    tx, ty : int
        The coordinates of the target value within target_array which is to be
        maximised.
    
    Returns
    =======
    test_pattern_specification : :py:class:`TestPatternSpecification`
    """
    test_pattern_dict = {
        (x, y): polarity
        for (prefix, x, y), polarity in get_maximising_inputs(
            target_array[tx, ty]
        ).items()
    }
    
    xs, ys = zip(*test_pattern_dict)
    min_x = min(xs)
    min_y = min(ys)
    
    # Find the multiple by which test pattern coordinates must be translated to
    # achieve equivalent filter behaviour
    tmx, tmy = target_array.period
    mx, my = target_array.relative_step_size_to(input_array)
    assert mx.denominator == my.denominator == 1
    mx = int(mx) * tmx
    my = int(my) * tmy
    
    translate_steps_x = min_x // mx
    translate_steps_y = min_y // my
    
    test_pattern_dict = {
        (x - (translate_steps_x * mx), y - (translate_steps_y * my)): polarity
        for (x, y), polarity in test_pattern_dict.items()
    }
    
    tx -= translate_steps_x * tmx
    ty -= translate_steps_y * tmy
    
    return TestPatternSpecification(
        target=(tx, ty),
        pattern=TestPattern(test_pattern_dict),
        pattern_translation_multiple=(mx, my),
        target_translation_multiple=(tmx, tmy),
    )


def make_synthesis_maximising_pattern(
    analysis_input_array,
    analysis_transform_coeff_arrays,
    synthesis_target_array,
    synthesis_output_array,
    tx, ty,
):
    """
    Create a test pattern which, after lossy encoding, is likely to maximise an
    intermediate/final value of the synthesis filter.
    
    .. warning::
        
        Because (heavy) lossy VC-2 encoding is a non-linear process, finding
        encoder inputs which maximise the decoder output is not feasible in
        general. This function uses a simple heuristic (see
        :ref:`theory-test-patterns`) to attempt to achieve this goal but
        cannot provide any guarantees about the extent to which it succeeds.
    
    Parameters
    ==========
    analysis_input_array : :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`
        The input array to a compatible analysis filter for the synthesis
        filter whose values will be maximised.
    analysis_transform_coeff_arrays : {level: {orient: :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`, ...}}
        The final output arrays from a compatible analysis filter for the
        synthesis filter whose values will be maximised. These should be
        provided as a nested dictionary of the form produced by
        :py:func:`~vc2_bit_widths.vc2_filters.analysis_transform`.
    synthesis_target_array : :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`
        The intermediate or final output array produced by the synthesis filter
        within which a value should be maximised.
    synthesis_output_array : :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`
        The output array for the synthesis filter.
    tx, ty : int
        The index of the value within the synthesis_target_array which is to be
        maximised.
    
    Returns
    =======
    test_pattern_specification : :py:class:`TestPatternSpecification`
    """
    # Enumerate the transform coefficients which maximise the target value
    # {(level, orient, x, y): coeff, ...}
    target_transform_coeffs = {
        (sym[0][1], sym[0][2], sym[1], sym[2]): coeff
        for sym, coeff in synthesis_target_array[tx, ty]
        if sym is not None and not isinstance(sym, AAError)
    }
    
    # In the loop below we'll compute two things: the set of pixels which directly
    # and indirectly effect the target value. NB: These steps are interleaved
    # to ensure that the transform coefficient expression only has to be
    # calculated once (giving a ~15% runtime saving for large filters).
    
    # The LinExp below collects the input pixels which (in the absence of
    # quantisation) contribute to the target value. In theory it it is
    # sufficient to maximise/minimise these to achieve a maximum value in the
    # target.
    directly_contributing_input_expr = LinExp(0)
    
    # As well as setting the pixels which directly contribute to maximising the
    # target (see above), we will also set other nearby pixels which contribute
    # to the same transform coefficients. In the absence of quantisation this
    # will have no effect on the target value, however quantisation can cause
    # the extra energy in the transform coefficients to 'leak' into the target
    # value.
    #
    # A simple greedy approach is used to maximising all of the coefficient
    # magnitudes simultaneously: priority is given to transform coefficients
    # with the greatest weight. The dictionary below collects the polarity
    # assigned to each input pixel and is set starting with the lowest weighted
    # transform coefficients which may be partially overwritten by higher
    # priority coefficients later on.
    test_pattern_dict = {}
    
    for (level, orient, cx, cy), transform_coeff in sorted(
        target_transform_coeffs.items(),
        # NB: Key includes (level, orient, x, y) to break ties and ensure
        # deterministic output
        key=lambda loc_coeff: (abs(loc_coeff[1]), loc_coeff[0]),
    ):
        coeff_expr = analysis_transform_coeff_arrays[level][orient][cx, cy]
        
        directly_contributing_input_expr += coeff_expr * transform_coeff
        
        for (prefix, px, py), pixel_coeff in get_maximising_inputs(coeff_expr).items():
            test_pattern_dict[px, py] = pixel_coeff * (1 if transform_coeff > 0 else -1)
    
    # To ensure the generated test pattern definately produces near worst-case
    # results under no-quantisation, we give the greatest priority to pixels
    # which directly control the target value!
    directly_contributing_input_pixels = {
        (x, y): coeff
        for (prefix, x, y), coeff in get_maximising_inputs(
            directly_contributing_input_expr
        ).items()
    }
    test_pattern_dict.update(directly_contributing_input_pixels)
    
    # The test pattern may contain negative pixel coordinates. To be useful, it
    # must be translated to a position implementing the same filter but which
    # is free from negative pixel coordinates.
    #
    # Find the multiples by which test pattern and target array coordinates must
    # be translated to still maximise equivalent transform coefficients.
    tmx, tmy = synthesis_target_array.period
    mx, my = synthesis_output_array.relative_step_size_to(synthesis_target_array)
    mx = (1 / mx) * tmx
    my = (1 / my) * tmy
    assert mx.denominator == my.denominator == 1
    mx = int(mx)
    my = int(my)
    
    # Translate the test pattern accordingly
    xs, ys = zip(*test_pattern_dict)
    min_x = min(xs)
    min_y = min(ys)
    
    translate_steps_x = min_x // mx
    translate_steps_y = min_y // my
    
    test_pattern_dict = {
        (x - (translate_steps_x * mx), y - (translate_steps_y * my)): polarity
        for (x, y), polarity in test_pattern_dict.items()
    }
    
    tx -= translate_steps_x * tmx
    ty -= translate_steps_y * tmy
    
    return TestPatternSpecification(
        target=(tx, ty),
        pattern=TestPattern(test_pattern_dict),
        pattern_translation_multiple=(mx, my),
        target_translation_multiple=(tmx, tmy),
    )
