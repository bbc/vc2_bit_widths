import pytest

import numpy as np

from vc2_data_tables import LIFTING_FILTERS, WaveletFilters

from vc2_bit_widths.infinite_arrays import SymbolArray

from vc2_bit_widths.vc2_filters import (
    make_symbol_coeff_arrays,
    make_variable_coeff_arrays,
    analysis_transform,
    synthesis_transform,
)

from vc2_bit_widths.signal_bounds import (
    analysis_filter_bounds,
    evaluate_analysis_filter_bounds,
    signed_integer_range,
)

from vc2_bit_widths.patterns import TestPattern as TP

from vc2_bit_widths.pattern_optimisation import (
    optimise_synthesis_maximising_test_pattern,
)

from vc2_bit_widths.pattern_generation import (
    make_analysis_maximising_pattern,
    make_synthesis_maximising_pattern,
)

from vc2_bit_widths.pattern_evaluation import (
    convert_test_pattern_to_padded_picture_and_slice,
    evaluate_analysis_test_pattern_output,
    evaluate_synthesis_test_pattern_output,
)


class TestConvertTestPatternToArrayAndSlice(object):
    
    def test_conversion_to_array(self):
        test_pattern, search_slice = convert_test_pattern_to_padded_picture_and_slice(
            # NB: Coordinates are (x, y) here
            test_pattern=TP({
                (2, 3): +1,
                (4, 5): -1,
            }),
            input_min=-512,
            input_max=511,
            dwt_depth=3,
            dwt_depth_ho=0,
        )
        
        assert np.array_equal(test_pattern, np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 511, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -512, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]))
        
        # NB: Coordinates are (y, x) here!
        assert search_slice == (slice(3, 6), slice(2, 5))
    
    @pytest.mark.parametrize("dwt_depth,dwt_depth_ho,exp_shape", [
        (0, 0, (7, 7)),
        (1, 0, (8, 8)),
        (0, 1, (7, 8)),
        (2, 0, (8, 8)),
        (0, 2, (7, 8)),
        (3, 0, (8, 8)),
        (0, 3, (7, 8)),
        (4, 0, (16, 16)),
        (0, 4, (7, 16)),
        (1, 3, (8, 16)),
    ])
    def test_rounding_up_sizes(self, dwt_depth, dwt_depth_ho, exp_shape):
        test_pattern, search_slice = convert_test_pattern_to_padded_picture_and_slice(
            # NB: Coordinates are (x, y) here
            test_pattern=TP({
                (6, 6): +1,
            }),
            input_min=-512,
            input_max=511,
            dwt_depth=dwt_depth,
            dwt_depth_ho=dwt_depth_ho,
        )
        
        assert test_pattern.shape == exp_shape


def test_evaluate_analysis_test_pattern_output():
    # In this test we check that the decoded values are plausible based on them
    # being close to the predicted signal range
    
    wavelet_index = WaveletFilters.haar_with_shift
    wavelet_index_ho = WaveletFilters.le_gall_5_3
    dwt_depth = 1
    dwt_depth_ho = 0
    
    picture_bit_width = 10
    
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    
    input_min, input_max = signed_integer_range(picture_bit_width)
    
    input_array = SymbolArray(2)
    _, intermediate_arrays = analysis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        input_array,
    )
    
    for (level, array_name), target_array in intermediate_arrays.items():
        for x in range(target_array.period[0]):
            for y in range(target_array.period[1]):
                # Compute the expected bounds for this value
                lower_bound, upper_bound = evaluate_analysis_filter_bounds(
                    *analysis_filter_bounds(target_array[x, y]),
                    num_bits=picture_bit_width
                )
                
                # Create a test pattern
                test_pattern = make_analysis_maximising_pattern(
                    input_array,
                    target_array,
                    x, y,
                )
                
                # Find the actual values
                lower_value, upper_value = evaluate_analysis_test_pattern_output(
                    h_filter_params,
                    v_filter_params,
                    dwt_depth,
                    dwt_depth_ho,
                    level,
                    array_name,
                    test_pattern,
                    input_min,
                    input_max,
                )
                
                assert np.isclose(lower_value, lower_bound, rtol=0.01)
                assert np.isclose(upper_value, upper_bound, rtol=0.01)


def test_evaluate_synthesis_test_pattern_output():
    # In this test we simply check that the decoded values match those
    # computed by the optimise_synthesis_maximising_test_pattern function
    
    wavelet_index = WaveletFilters.haar_with_shift
    wavelet_index_ho = WaveletFilters.le_gall_5_3
    dwt_depth = 1
    dwt_depth_ho = 0
    
    picture_bit_width = 10
    
    max_quantisation_index = 64
    
    quantisation_matrix = {
        0: {"LL": 0},
        1: {"LH": 1, "HL": 2, "HH": 3},
    }
    
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    
    input_min, input_max = signed_integer_range(picture_bit_width)
    
    input_array = SymbolArray(2)
    analysis_transform_coeff_arrays, _ = analysis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        input_array,
    )
    
    symbolic_coeff_arrays = make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho)
    symbolic_output_array, symbolic_intermediate_arrays = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        symbolic_coeff_arrays,
    )
    
    pyexp_coeff_arrays = make_variable_coeff_arrays(dwt_depth, dwt_depth_ho)
    _, pyexp_intermediate_arrays = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        pyexp_coeff_arrays,
    )
    
    for (level, array_name), target_array in symbolic_intermediate_arrays.items():
        for x in range(target_array.period[0]):
            for y in range(target_array.period[1]):
                # Create a test pattern
                test_pattern = make_synthesis_maximising_pattern(
                    input_array,
                    analysis_transform_coeff_arrays,
                    target_array,
                    symbolic_output_array,
                    x, y,
                )
                
                synthesis_pyexp = pyexp_intermediate_arrays[(level, array_name)][x, y]
                # Run with no-optimisation iterations but, as a side effect,
                # compute the actual decoded value to compare with
                test_pattern = optimise_synthesis_maximising_test_pattern(
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
                    None,
                    1,
                    None,
                    0.0,
                    0.0,
                    0,
                    0,
                )
                
                # Find the actual values
                lower_value, upper_value = evaluate_synthesis_test_pattern_output(
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
                )
                
                assert upper_value[0] == test_pattern.decoded_value
                assert upper_value[1] == test_pattern.quantisation_index
