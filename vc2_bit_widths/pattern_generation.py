"""
Heuristic Test Pattern Generation
=================================

The routines in this module are designed to produce test patterns for VC-2
encoders and decoders which produce near-maximum magnitude values in the
target.

Due to non-linearities in VC-2's filters (i.e. rounding and quantisation), the
test patterns generated are not true worst-case signals but rather a 'best
effort' to get close to the worst case. Encoder test patterns will tend to be
very close to worst-case signals while decoder signals are likely to be modest
under-estimates. Nevertheless, these signals are likely to have value ranges
well above real picture signals.

"""

from collections import namedtuple

import numpy as np

from vc2_bit_widths.linexp import (
    LinExp,
    AAError,
)

from vc2_bit_widths.fast_partial_analysis_transform import (
    fast_partial_analysis_transform,
)

from vc2_bit_widths.fast_partial_analyse_quantise_synthesise import (
    FastPartialAnalyseQuantiseSynthesise,
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


TestPatternSpecification = namedtuple(
    "TestPatternSpecification",
    "target,pattern,pattern_translation_multiple,target_translation_multiple",
)
"""
A definition of a test pattern for a VC-2 filter. This test pattern is intended
to maximise the value of a particular intermediate or output value of a VC-2
filter.

Test patterns for both for analysis and synthesis filters are defined in terms
of a picture. For analysis filters, the picture should be fed to an encoder and
the resulting transform coefficients analysed. For synthesis filters, the
picture should be fed to an encoder where it may be quantised before being fed
to a decoder.

The pictures defined for a test pattern tend to be quite small and may be
relocated within a larger picture if required. Translations are only permitted
by multiples of ``pattern_translation_multiple`` and have the effect of moving
the coordinate of the maximised target value by an equivalent multiple of
``target_translation_multiple``.

Parameters
==========
target : (tx, ty)
    The target coordinate which is maximised by this test pattern.
pattern : {(x, y): polarity, ...}
    The input pattern to be fed into a VC-2 encoder. Only those pixels defined
    in this dictionary need be set -- all other pixels may be set to arbitrary
    values and have no effect.
    
    The 'polarity' value will be either +1 or -1. When +1, the corresponding
    pixel should be set to its maximum signal value. When -1, the pixel should
    be set to its minimum value.
    
    To produce a test pattern which minimises, rather than maximises the target
    value, the meaning of the polarity should be inverted.
    
    All pixels be located such that for the left-most pixel, 0 <= x < mx and for the
    top-most pixel, 0 <= y < my (see pattern_translation_multiple).
pattern_translation_multiple : (mx, my)
target_translation_multiple : (tmx, tmy)
    The multiples by which pattern pixel coordinates and target array
    coordinates may be translated when relocating the test pattern. Both the
    pattern and target must be translated by the same multiple of these two
    factors.
    
    For example, if the pattern is translated by (2*mx, 3*my), the target must
    be translated by (2*tmx, 3*tmy).
"""


def invert_test_pattern_specification(test_pattern):
    """
    Given a :py:class:`TestPatternSpecification` or
    :py:class:`~vc2_bit_widths.pattern_optimisation.OptimisedTestPatternSpecification`,
    return a copy with the signal polarity inverted.
    """
    tuple_type = type(test_pattern)
    
    values = test_pattern._asdict()
    values["pattern"] = {
        (x, y): polarity * -1
        for (x, y), polarity in values["pattern"].items()
    }
    
    return tuple_type(**values)


def make_analysis_maximising_signal(input_array, target_array, tx, ty):
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
        The index of the value within the target_array which is to be
        maximised.
    
    Returns
    =======
    test_pattern : :py:class:`TestPatternSpecification`
    """
    test_pattern = {
        (x, y): polarity
        for (prefix, x, y), polarity in get_maximising_inputs(
            target_array[tx, ty]
        ).items()
    }
    
    xs, ys = zip(*test_pattern)
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
    
    test_pattern = {
        (x - (translate_steps_x * mx), y - (translate_steps_y * my)): polarity
        for (x, y), polarity in test_pattern.items()
    }
    
    tx -= translate_steps_x * tmx
    ty -= translate_steps_y * tmy
    
    return TestPatternSpecification(
        target=(tx, ty),
        pattern=test_pattern,
        pattern_translation_multiple=(mx, my),
        target_translation_multiple=(tmx, tmy),
    )


def make_synthesis_maximising_signal(
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
        general. This function uses a simple heuristic to attempt to achieve
        this goal but cannot provide any guarantees about the extent to which
        it succeeds.
    
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
    test_pattern : :py:class:`TestPatternSpecification`
    """
    # Enumerate the transform coefficients which maximise the target value
    # {(level, orient, x, y): coeff, ...}
    target_transform_coeffs = {
        (sym[0][1], sym[0][2], sym[1], sym[2]): coeff
        for sym, coeff in synthesis_target_array[tx, ty]
        if sym is not None and not isinstance(sym, AAError)
    }
    
    # Find the input pixels which (in the absence of quantisation) contribute
    # to the target value. In theory it will be sufficient to maximise/minimise
    # these to achieve a maximum value in the target.
    directly_contributing_input_expr = LinExp(0)
    for (level, orient, x, y), transform_coeff in target_transform_coeffs.items():
        directly_contributing_input_expr += (
            analysis_transform_coeff_arrays[level][orient][x, y] * transform_coeff
        )
    directly_contributing_input_pixels = {
        (x, y): coeff
        for (prefix, x, y), coeff in get_maximising_inputs(
            directly_contributing_input_expr
        ).items()
    }
    
    # As well as setting the pixels which directly contribute to maximising the
    # target (see above), we will also set other nearby pixels which contribute
    # to the same transform coefficients. In the absence of quantisation this
    # will have no effect on the target value, however quantisation can cause
    # the extra energy in the transform coefficients to 'leak' into the target
    # value.
    #
    # A simple greedy approach is used to maximising all of the coefficient
    # magnitudes simultaneously: priority is given to transform coefficients
    # with the greatest weight.
    test_pattern = {}
    for (level, orient, cx, cy), transform_coeff in sorted(
        target_transform_coeffs.items(),
        # NB: Key includes (level, orient, x, y) to break ties and ensure
        # deterministic output
        key=lambda loc_coeff: (abs(loc_coeff[1]), loc_coeff[0]),
    ):
        for (prefix, px, py), pixel_coeff in get_maximising_inputs(
            analysis_transform_coeff_arrays[level][orient][cx, cy]
        ).items():
            test_pattern[px, py] = pixel_coeff * (1 if transform_coeff > 0 else -1)
    
    # The greatest priority, however, must be given to the pixels which
    # directly control the target value!
    test_pattern.update(directly_contributing_input_pixels)
    
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
    xs, ys = zip(*test_pattern)
    min_x = min(xs)
    min_y = min(ys)
    
    translate_steps_x = min_x // mx
    translate_steps_y = min_y // my
    
    test_pattern = {
        (x - (translate_steps_x * mx), y - (translate_steps_y * my)): polarity
        for (x, y), polarity in test_pattern.items()
    }
    
    tx -= translate_steps_x * tmx
    ty -= translate_steps_y * tmy
    
    return TestPatternSpecification(
        target=(tx, ty),
        pattern=test_pattern,
        pattern_translation_multiple=(mx, my),
        target_translation_multiple=(tmx, tmy),
    )


def convert_test_pattern_to_array_and_slice(
    test_pattern,
    input_min,
    input_max,
    dwt_depth,
    dwt_depth_ho,
):
    """
    Convert a description of a test pattern (in terms of polarities) into a
    :py:class:`numpy.array`, padded ready for processing with a filter with the
    specified transform depths.
    
    Parameters
    ==========
    test_pattern : {(x, y): polarity, ...}
    input_min : int
    input_max : int
        The full signal range to expand the test pattern to.
    dwt_depth : int
    dwt_depth_ho : int
        The transform depth used by the filters.
    
    Returns
    =======
    pattern : :py:class:`numpy.array`
    search_slice : (:py:class:`slice`, :py:class:`slice`)
        A 2D slice out of ``test_pattern`` which contains the active pixels in
        ``test_pattern`` (i.e. excluding any padding).
    """
    xs, ys = zip(*test_pattern)
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    search_slice = (slice(y0, y1+1), slice(x0, x1+1))
    
    # Round width/height up to be compatible with the lifting filter
    x_multiple = 2**(dwt_depth + dwt_depth_ho)
    y_multiple = 2**(dwt_depth)
    width = (((x1 + 1) + x_multiple - 1) // x_multiple) * x_multiple
    height = (((y1 + 1) + y_multiple - 1) // y_multiple) * y_multiple
    
    pattern = np.array([
        [
            0
            if test_pattern.get((x, y), 0) == 0 else
            input_min
            if test_pattern.get((x, y), 0) < 0 else
            input_max
            for x in range(width)
        ]
        for y in range(height)
    ], dtype=int)
    
    return pattern, search_slice




def evaluate_analysis_test_pattern_output(
    h_filter_params,
    v_filter_params,
    dwt_depth,
    dwt_depth_ho,
    level,
    array_name,
    test_pattern,
    input_min,
    input_max,
):
    """
    Given an analysis test pattern (e.g. created using
    :py:func:`make_analysis_maximising_signal`), return the actual intermediate
    encoder value when the signal is processed.
    
    Parameters
    ==========
    h_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
    v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
        Horizontal and vertical filter *synthesis* (not analysis!) filter
        parameters (e.g. from :py:data:`vc2_data_tables.LIFTING_FILTERS`)
        defining the wavelet transform used.
    dwt_depth : int
    dwt_depth_ho : int
        The transform depth used by the filters.
    level : int
    array_name : str
        The intermediate value in the encoder the test pattern targets.
    test_pattern : :py:class:`TestPatternSpecification`
        The test pattern to evaluate.
    input_min : int
    input_max : int
        The minimum and maximum value which may be used in the test pattern.
    
    Returns
    =======
    encoded_minimum : int
    encoded_maximum : int
        The target encoder value when the test pattern encoded with minimising
        and maximising signal levels respectively.
    """
    tx, ty = test_pattern.target
    
    # NB: Casting to native int from numpy for JSON serialisability etc.
    return tuple(
        int(fast_partial_analysis_transform(
            h_filter_params,
            v_filter_params,
            dwt_depth,
            dwt_depth_ho,
            convert_test_pattern_to_array_and_slice(
                test_pattern.pattern,
                cur_min,
                cur_max,
                dwt_depth,
                dwt_depth_ho,
            )[0],
            (level, array_name, tx, ty),
        ))
        for cur_min, cur_max in [
            # Minimise encoder output
            (input_max, input_min),
            # Maximise encoder output
            (input_min, input_max),
        ]
    )


def evaluate_synthesis_test_pattern_output(
    h_filter_params,
    v_filter_params,
    dwt_depth,
    dwt_depth_ho,
    quantisation_matrix,
    synthesis_pyexp,
    test_pattern,
    input_min,
    input_max,
    max_quantisation_index,
):
    """
    Given a synthesis test pattern (e.g. created using
    :py:func:`make_synthesis_maximising_signal` or
    :py:func:`optimise_synthesis_maximising_test_pattern`), return the actual decoder
    value, and worst-case quantisation index when the signal is processed.
    
    Parameters
    ==========
    h_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
    v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
        Horizontal and vertical filter *synthesis* (not analysis!) filter
        parameters (e.g. from :py:data:`vc2_data_tables.LIFTING_FILTERS`)
        defining the wavelet transform used.
    dwt_depth : int
    dwt_depth_ho : int
        The transform depth used by the filters.
    quantisation_matrix : {level: {orient: int, ...}, ...}
        The VC-2 quantisation matrix to use.
    synthesis_pyexp : :py:class:`~vc2_bit_widths.PyExp`
        A :py:class:`~vc2_bit_widths.PyExp` expression which defines the
        synthesis process for the decoder value the test pattern is
        maximising/minimising. Such an expression is usually obtained from the
        use of :py:func;`~vc2_bit_widths.vc2_filters.synthesis_transform` and
        :py:func;`~vc2_bit_widths.vc2_filters.make_variable_coeff_arrays`.
    test_pattern : :py:class:`TestPatternSpecification`
        The test pattern to evaluate.
    input_min : int
    input_max : int
        The minimum and maximum value which may be used in the test pattern.
    max_quantisation_index : int
        The maximum quantisation index to use. This should be set high enough
        that at the highest quantisation level all transform coefficients are
        quantised to zero.
    
    Returns
    =======
    decoded_minimum : (value, quantisation_index)
    decoded_maximum : (value, quantisation_index)
        The target decoded value (and worst-case quantisation index) when the
        test pattern is encoded using minimising and maximising values
        respectively.
    """
    quantisation_indices = list(range(max_quantisation_index + 1))
    
    codec = FastPartialAnalyseQuantiseSynthesise(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        quantisation_indices,
        synthesis_pyexp,
    )
    
    out = []
    
    for cur_min, cur_max in [
        # Minimise encoder output
        (input_max, input_min),
        # Maximise encoder output
        (input_min, input_max),
    ]:
        pattern_array, _ = convert_test_pattern_to_array_and_slice(
            test_pattern.pattern,
            cur_min,
            cur_max,
            dwt_depth,
            dwt_depth_ho,
        )
        
        decoded_values = codec.analyse_quantise_synthesise(pattern_array)
        
        i = np.argmax(np.abs(decoded_values))
        
        # NB: Force native Python integer type for JSON serialisability
        out.append((
            int(decoded_values[i]),
            quantisation_indices[i]
        ))
    
    return tuple(out)
