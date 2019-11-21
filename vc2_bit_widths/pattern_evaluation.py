"""
Measure the outputs achieved by test patterns
=============================================

"""

import numpy as np

from vc2_bit_widths.fast_partial_analysis_transform import (
    fast_partial_analysis_transform,
)

from vc2_bit_widths.fast_partial_analyse_quantise_synthesise import (
    FastPartialAnalyseQuantiseSynthesise,
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
    :py:func:`make_analysis_maximising_pattern`), return the actual intermediate
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
    :py:func:`make_synthesis_maximising_pattern` or
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
