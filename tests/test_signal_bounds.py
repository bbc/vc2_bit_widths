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
    PostQuantisationTransformCoeffBoundsLookup,
    synthesis_filter_bounds,
)


@pytest.mark.parametrize("expr,exp_lower,exp_upper", [
    # Constant, no errors, no variables
    (0, 0, 0),
    (123, 123, 123),
    (-123, -123, -123),
    # Error term, no variables
    (affine_error_with_range(-123, 1234), -123, 1234),
    # Variable, no error term, no constants
    (LinExp(("p", 0, 0)), -100, 1000),
    (LinExp({("p", 0, 0): -20}), -20000, 2000),
    # Everything!
    (
        LinExp(("p", 0, 0)) + 10 + affine_error_with_range(-5, 15),
        -100 + 10 - 5,
        1000 + 10 + 15,
    ),
])
def test_find_analysis_filter_bounds(expr, exp_lower, exp_upper):
    assert analysis_filter_bounds(expr, -100, 1000) == (exp_lower, exp_upper)


class TestPostQuantisationTransformCoeffBoundsLookup(object):
    
    @pytest.fixture
    def coeff_arrays(self):
        # Mocked up with examples of all useful InfiniteArray variants
        a = SymbolArray(2, "a")
        b = RightShiftedArray(SymbolArray(2, "b"), 1)
        c = InterleavedArray(a, b, 0)
        coeff_arrays = {2: {"LH": a, "HL": b, "HH": c}}
        return coeff_arrays
    
    def test_simple_symbol(self, coeff_arrays):
        lookup = PostQuantisationTransformCoeffBoundsLookup(coeff_arrays, -100, 1000)
        assert lookup[2, "LH", 0, 0] == (
            maximum_dequantised_magnitude(-100),
            maximum_dequantised_magnitude(1000),
        )
    
    def test_includes_error_terms(self, coeff_arrays):
        lookup = PostQuantisationTransformCoeffBoundsLookup(coeff_arrays, -100, 1000)
        assert lookup[2, "HL", 0, 0] == (
            maximum_dequantised_magnitude(-50.5),
            maximum_dequantised_magnitude(500.5),
        )
    
    def test_caching(self, coeff_arrays):
        lookup = PostQuantisationTransformCoeffBoundsLookup(coeff_arrays, -100, 1000)
        
        assert len(lookup._cache) == 0
        
        assert lookup[2, "HH", 0, 0] == (
            maximum_dequantised_magnitude(-100),
            maximum_dequantised_magnitude(1000),
        )
        assert len(lookup._cache) == 1
        
        assert lookup[2, "HH", 2, 0] == (
            maximum_dequantised_magnitude(-100),
            maximum_dequantised_magnitude(1000),
        )
        assert lookup[2, "HH", 2, 1] == (
            maximum_dequantised_magnitude(-100),
            maximum_dequantised_magnitude(1000),
        )
        assert len(lookup._cache) == 1
        
        assert lookup[2, "HH", 1, 0] == (
            maximum_dequantised_magnitude(-50.5),
            maximum_dequantised_magnitude(500.5),
        )
        assert len(lookup._cache) == 2
        
        assert lookup[2, "HH", 3, 0] == (
            maximum_dequantised_magnitude(-50.5),
            maximum_dequantised_magnitude(500.5),
        )
        assert lookup[2, "HH", 3, 1] == (
            maximum_dequantised_magnitude(-50.5),
            maximum_dequantised_magnitude(500.5),
        )
        assert len(lookup._cache) == 2


coeff_arrays = make_symbol_coeff_arrays(2, 0)

@pytest.mark.parametrize("expr,exp_lower,exp_upper", [
    # Constant, no errors, no variables
    (0, 0, 0),
    (123, 123, 123),
    (-123, -123, -123),
    # Error term, no variables
    (affine_error_with_range(-123, 1234), -123, 1234),
    # Variable, no error term, no constants
    (coeff_arrays[2]["LH"][0, 0], -10, 100),
    (coeff_arrays[1]["HH"][2, 3]*-20, -20000, 2000),
    # Everything!
    (
        coeff_arrays[1]["HH"][2, 3] + 10 + affine_error_with_range(-5, 15),
        -100 + 10 - 5,
        1000 + 10 + 15,
    ),
])
def test_find_synthesis_filter_bounds(expr, exp_lower, exp_upper):
    # Mocked up
    coeff_value_ranges = {
        (2, "LH", 0, 0): (-10, 100),
        (1, "HH", 2, 3): (-100, 1000),
    }
    
    assert synthesis_filter_bounds(expr, coeff_value_ranges) == (exp_lower, exp_upper)


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
    
    input_lower, input_upper = -512, 511
    
    # Input signal bounds should be as specified
    assert analysis_filter_bounds(
        analysis_intermediate_values[(2, "Input")][0, 0],
        input_lower, input_upper,
    ) == (input_lower, input_upper)
    
    # Output of final analysis filter should require a greater depth (NB: for
    # the Haar transform it is the high-pass bands which gain the largest
    # signal range)
    analysis_output_lower, analysis_output_upper = analysis_filter_bounds(
        analysis_intermediate_values[(1, "H")][0, 0],
        input_lower, input_upper,
    )
    assert analysis_output_lower < input_lower
    assert analysis_output_upper > input_upper
    
    analysis_coeff_array_bounds = PostQuantisationTransformCoeffBoundsLookup(
        analysis_coeff_arrays,
        input_lower,
        input_upper,
    )
    
    # Signal range should shrink down by end of synthesis process but should
    # still be larger than the original signal
    final_output_lower, final_output_upper = synthesis_filter_bounds(
        synthesis_output[0, 0],
        analysis_coeff_array_bounds,
    )
    assert final_output_upper < analysis_output_upper
    assert final_output_lower > analysis_output_lower
    
    assert final_output_upper > input_upper
    assert final_output_lower < input_lower
