import pytest

from vc2_data_tables import LIFTING_FILTERS, WaveletFilters

from vc2_bit_widths.linexp import (
    LinExp,
    affine_error_with_range,
)

from vc2_bit_widths.quantisation import maximum_dequantised_magnitude

from vc2_bit_widths.infinite_arrays import (
    SymbolArray,
    RightShiftedArray,
    InterleavedArray,
)

from vc2_bit_widths.vc2_filters import (
    make_symbol_coeff_arrays,
    analysis_transform,
    synthesis_transform,
)

from vc2_bit_widths.signal_bounds import (
    analysis_filter_bounds,
    synthesis_filter_bounds,
    parse_coeff_symbol_name,
)


def test_find_analysis_filter_bounds():
    expr = (
        1*LinExp(("pixel", 0, 0)) +
        2*LinExp(("pixel", 1, 0)) +
        -4*LinExp(("pixel", 2, 0)) +
        10*LinExp.new_affine_error_symbol() +
        1
    )
    lower_bound, upper_bound = analysis_filter_bounds(expr)
    assert lower_bound == 3*LinExp("signal_min") + -4*LinExp("signal_max") - 10 + 1
    assert upper_bound == -4*LinExp("signal_min") + 3*LinExp("signal_max") + 10 + 1


def test_find_synthesis_filter_bounds():
    coeff_arrays = make_symbol_coeff_arrays(1, 0)
    
    expr = (
        1*coeff_arrays[0]["LL"][0, 0] +
        2*coeff_arrays[0]["LL"][1, 0] +
        -4*coeff_arrays[0]["LL"][2, 0] +
        10*coeff_arrays[1]["HH"][0, 0] +
        20*coeff_arrays[1]["HH"][1, 0] +
        -40*coeff_arrays[1]["HH"][2, 0] +
        100*LinExp.new_affine_error_symbol() +
        1
    )
    
    lower_bound, upper_bound = synthesis_filter_bounds(expr)
    assert lower_bound == (
        3*LinExp("coeff_0_LL_min") + -4*LinExp("coeff_0_LL_max") +
        30*LinExp("coeff_1_HH_min") + -40*LinExp("coeff_1_HH_max") +
        -100 +
        1
    )
    assert upper_bound == (
        -4*LinExp("coeff_0_LL_min") + 3*LinExp("coeff_0_LL_max") +
        -40*LinExp("coeff_1_HH_min") + 30*LinExp("coeff_1_HH_max") +
        +100 +
        1
    )


def test_integration():
    # A simple integration test which computes signal bounds for a small
    # transform operation
    
    filter_params = LIFTING_FILTERS[WaveletFilters.haar_with_shift]
    dwt_depth = 1
    dwt_depth_ho = 1
    
    input_picture_array = SymbolArray(2)
    analysis_coeff_arrays, analysis_intermediate_values = analysis_transform(
        filter_params, filter_params,
        dwt_depth, dwt_depth_ho,
        input_picture_array,
    )
    
    input_coeff_arrays = make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho)
    synthesis_output, synthesis_intermediate_values = synthesis_transform(
        filter_params, filter_params,
        dwt_depth, dwt_depth_ho,
        input_coeff_arrays,
    )
    
    signal_min = LinExp("signal_min")
    signal_max = LinExp("signal_max")
    
    example_range = {signal_min: -512, signal_max: 511}
    
    # Input signal bounds should be as specified
    assert analysis_filter_bounds(
        analysis_intermediate_values[(2, "Input")][0, 0],
    ) == (signal_min, signal_max)
    
    # Output of final analysis filter should require a greater depth (NB: for
    # the Haar transform it is the high-pass bands which gain the largest
    # signal range)
    analysis_output_lower, analysis_output_upper = analysis_filter_bounds(
        analysis_intermediate_values[(1, "H")][0, 0],
    )
    assert analysis_output_lower.subs(example_range) < signal_min.subs(example_range)
    assert analysis_output_upper.subs(example_range) > signal_max.subs(example_range)
    
    example_coeff_range = {
        "coeff_{}_{}_{}".format(level, orient, minmax):
            maximum_dequantised_magnitude(int(round(value.subs(example_range).constant)))
        for level, orients in analysis_coeff_arrays.items()
        for orient, expr in orients.items()
        for minmax, value in zip(["min", "max"], analysis_filter_bounds(expr))
    }
    
    # Signal range should shrink down by end of synthesis process but should
    # still be larger than the original signal
    final_output_lower, final_output_upper = synthesis_filter_bounds(synthesis_output[0, 0])
    
    assert final_output_upper.subs(example_coeff_range) < analysis_output_upper.subs(example_range)
    assert final_output_lower.subs(example_coeff_range) > analysis_output_lower.subs(example_range)
    
    assert final_output_upper.subs(example_coeff_range) > signal_max.subs(example_range)
    assert final_output_lower.subs(example_coeff_range) < signal_min.subs(example_range)


def test_parse_coeff_symbol_name():
    assert parse_coeff_symbol_name("coeff_123_HL_min") == (123, "HL", "min")
