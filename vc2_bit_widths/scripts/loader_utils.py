"""
Common file-loading (and simple analysis) routines common to some scripts.
"""

import json

from vc2_bit_widths.patterns import (
    TestPatternSpecification,
    OptimisedTestPatternSpecification,
)

from vc2_bit_widths.json_serialisations import (
    deserialise_signal_bounds,
    deserialise_test_pattern_specifications,
    deserialise_quantisation_matrix,
)

from vc2_bit_widths.scripts.argument_parsers import (
    parse_quantisation_matrix_argument,
)

from vc2_bit_widths.helpers import (
    evaluate_filter_bounds,
    quantisation_index_bound,
)


def load_filter_analysis(
    static_filter_analysis_file,
    optimised_synthesis_patterns_file,
    quantisation_matrix_argument,
    picture_bit_width,
):
    """
    Load a static filter analysis and optionally a set of optimised synthesis
    test patterns, returning all of the loaded data.
    
    Parameters
    ==========
    static_filter_analysis_file : :py:class:`file`
        An open file ready to read the static filter analysis data from a JSON
        file.
    optimised_synthesis_patterns_file : :py:class:`file` or None
        An open file ready to read a set of optimised synthesis test patterns
        for a JSON file. If None, synthesis test patterns will be read from the
        ``static_filter_analysis_file`` instead.
    quantisation_matrix_argument : [str, ...] or None
        The --custom-quantisation-matrix argument which will be parsed (if
        optimised_synthesis_patterns_file is not provided)
    picture_bit_width : int or None
        The --picture-bit-width argument which will be used if no
        optimised_synthesis_test_patterns file is provided.
    
    Returns
    =======
    wavelet_index : int
    wavelet_index_ho : int
    dwt_depth : int
    dwt_depth_ho : int
    quantisation_matrix : {level: {orient: value, ...}, ...}
    picture_bit_width : int
    max_quantisation_index : int
    concrete_analysis_bounds : {(level, array_name, x, y): (lo, hi), ...}
    concrete_synthesis_bounds : {(level, array_name, x, y): (lo, hi), ...}
    analysis_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
    synthesis_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
    """
    # Load precomputed signal bounds
    static_filter_analysis = json.load(static_filter_analysis_file)
    analysis_signal_bounds = deserialise_signal_bounds(
        static_filter_analysis["analysis_signal_bounds"]
    )
    synthesis_signal_bounds = deserialise_signal_bounds(
        static_filter_analysis["synthesis_signal_bounds"]
    )
    
    # Load precomputed test patterns
    analysis_test_patterns = deserialise_test_pattern_specifications(
        TestPatternSpecification,
        static_filter_analysis["analysis_test_patterns"]
    )
    synthesis_test_patterns = deserialise_test_pattern_specifications(
        TestPatternSpecification,
        static_filter_analysis["synthesis_test_patterns"]
    )
    
    # Load optimised synthesis signal
    if optimised_synthesis_patterns_file is not None:
        optimised_json = json.load(optimised_synthesis_patterns_file)
        
        assert static_filter_analysis["wavelet_index"] == optimised_json["wavelet_index"]
        assert static_filter_analysis["wavelet_index_ho"] == optimised_json["wavelet_index_ho"]
        assert static_filter_analysis["dwt_depth"] == optimised_json["dwt_depth"]
        assert static_filter_analysis["dwt_depth_ho"] == optimised_json["dwt_depth_ho"]
        
        picture_bit_width = optimised_json["picture_bit_width"]
        
        quantisation_matrix = deserialise_quantisation_matrix(
            optimised_json["quantisation_matrix"]
        )
        
        synthesis_test_patterns = deserialise_test_pattern_specifications(
            OptimisedTestPatternSpecification,
            optimised_json["optimised_synthesis_test_patterns"]
        )
    else:
        quantisation_matrix = parse_quantisation_matrix_argument(
            quantisation_matrix_argument,
            static_filter_analysis["wavelet_index"],
            static_filter_analysis["wavelet_index_ho"],
            static_filter_analysis["dwt_depth"],
            static_filter_analysis["dwt_depth_ho"],
        )
    
    # Compute signal bounds for all specified bit widths
    #
    # analysis_bounds_dicts = [{(level, array_name, x, y): (lower_bound, upper_bound), ...}, ...]
    # synthesis_bounds_dicts = same as above
    concrete_analysis_bounds, concrete_synthesis_bounds = evaluate_filter_bounds(
        static_filter_analysis["wavelet_index"],
        static_filter_analysis["wavelet_index_ho"],
        static_filter_analysis["dwt_depth"],
        static_filter_analysis["dwt_depth_ho"],
        analysis_signal_bounds,
        synthesis_signal_bounds,
        picture_bit_width,
    )
    
    # Find the maximum quantisation index for each bit width
    max_quantisation_index = quantisation_index_bound(
        concrete_analysis_bounds,
        quantisation_matrix,
    )
    
    return (
        static_filter_analysis["wavelet_index"],
        static_filter_analysis["wavelet_index_ho"],
        static_filter_analysis["dwt_depth"],
        static_filter_analysis["dwt_depth_ho"],
        quantisation_matrix,
        picture_bit_width,
        max_quantisation_index,
        concrete_analysis_bounds,
        concrete_synthesis_bounds,
        analysis_test_patterns,
        synthesis_test_patterns,
    )
