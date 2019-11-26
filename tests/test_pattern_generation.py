import pytest

from decode_and_quantise_test_utils import (
    encode_with_vc2,
    quantise_coeffs,
    decode_with_vc2,
)

from vc2_data_tables import LIFTING_FILTERS, WaveletFilters

from vc2_bit_widths.linexp import (
    LinExp,
    strip_affine_errors,
)

from vc2_bit_widths.infinite_arrays import SymbolArray

from vc2_bit_widths.vc2_filters import (
    make_symbol_coeff_arrays,
    analysis_transform,
    synthesis_transform,
)

from vc2_bit_widths.signal_bounds import (
    analysis_filter_bounds,
)

# NB: Can't have any class names starting with 'Test' in a Pytest test file!
from vc2_bit_widths.patterns import TestPattern as TP
from vc2_bit_widths.patterns import TestPatternSpecification as TPS
from vc2_bit_widths.patterns import OptimisedTestPatternSpecification as OTPS

from vc2_bit_widths.pattern_generation import (
    get_maximising_inputs,
    make_analysis_maximising_pattern,
    make_synthesis_maximising_pattern,
)


@pytest.mark.parametrize("expression,exp", [
    # Empty/constant expression
    (0, {}),
    (123, {}),
    (LinExp(123), {}),
    # Error terms should be omitted
    (LinExp.new_affine_error_symbol(), {}),
    (LinExp.new_affine_error_symbol() + 123, {}),
    # Terms should be extracted correctly
    (LinExp(("p", 0, 0)), {("p", 0, 0): 1}),
    # Only the signs of the coefficients should be preserved
    (-4*LinExp(("p", 0, 0)), {("p", 0, 0): -1}),
    # Combination of everything
    (
        2*LinExp(("p", 0, 0)) - 3*LinExp(("p", 1, 2)) + LinExp.new_affine_error_symbol() + 123,
        {("p", 0, 0): 1, ("p", 1, 2): -1},
    ),
])
def test_get_maximising_inputs(expression, exp):
    assert get_maximising_inputs(expression) == exp


class TestMakeAnalysisMaximisingSignal(object):
    
    @pytest.fixture(scope="class")
    def input_array(self):
        return SymbolArray(2)
    
    @pytest.fixture(scope="class")
    def wavelet_index(self):
        return WaveletFilters.le_gall_5_3
    
    @pytest.fixture(scope="class")
    def wavelet_index_ho(self):
        return WaveletFilters.le_gall_5_3
    
    @pytest.fixture(scope="class")
    def dwt_depth(self):
        return 1
    
    @pytest.fixture(scope="class")
    def dwt_depth_ho(self):
        return 1
    
    @pytest.fixture(scope="class")
    def analysis_transform_output(self, input_array,
                                   wavelet_index, wavelet_index_ho,
                                   dwt_depth, dwt_depth_ho):
        h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
        v_filter_params = LIFTING_FILTERS[wavelet_index]
        
        return analysis_transform(
            h_filter_params, v_filter_params,
            dwt_depth, dwt_depth_ho,
            input_array,
        )
    
    @pytest.fixture(scope="class")
    def transform_coeffs(self, analysis_transform_output):
        return analysis_transform_output[0]
    
    @pytest.fixture(scope="class")
    def intermediate_arrays(self, analysis_transform_output):
        return analysis_transform_output[1]
    
    def test_always_positive_pixel_coords(self, input_array, intermediate_arrays):
        for (level, name), target_array in intermediate_arrays.items():
            for tx in range(target_array.period[0]):
                for ty in range(target_array.period[1]):
                    ts = make_analysis_maximising_pattern(
                        input_array,
                        target_array,
                        tx, ty,
                    )
                    
                    new_tx, new_ty = ts.target
                    assert new_tx >= 0
                    assert new_ty >= 0
                    
                    # Ensure that, as promised, the returned test patterns
                    # don't use any negative pixel coordinates
                    assert ts.pattern.origin[0] >= 0
                    assert ts.pattern.origin[1] >= 0
                    
                    # Also ensure that the returned test pattern is as close to
                    # the edge of the picture as possible (that is, moving one
                    # multiple left or up moves us off the edge of the picture)
                    mx, my = ts.pattern_translation_multiple
                    assert ts.pattern.origin[0] < my
                    assert ts.pattern.origin[1] < mx
    
    def test_translation_is_valid(self, input_array, intermediate_arrays):
        for (level, name), target_array in intermediate_arrays.items():
            for tx in range(target_array.period[0]):
                for ty in range(target_array.period[1]):
                    ts = make_analysis_maximising_pattern(
                        input_array,
                        target_array,
                        tx, ty,
                    )
                    
                    new_tx, new_ty = ts.target
                    mx, my = ts.pattern_translation_multiple
                    tmx, tmy = ts.target_translation_multiple
                    
                    # Check that the translated values really are processed by
                    # an equivalent filter to the offset passed in
                    dx = (new_tx - tx) * mx // tmx
                    dy = (new_ty - ty) * my // tmy
                    specified_filter_translated = target_array[tx, ty].subs({
                        (prefix, x, y): (prefix, x + dx, y + dy)
                        for prefix, x, y in get_maximising_inputs(target_array[tx, ty])
                    })
                    returned_filter = target_array[new_tx, new_ty]
                    
                    delta = strip_affine_errors(specified_filter_translated - returned_filter)
                    assert delta == 0
                    
                    # Check that an offset of (tmx, tmy) offsets the input
                    # array by (mx, my)
                    target_filter = target_array[tx + 2*tmx, ty + 3*tmy]
                    translated_filter = target_array[tx, ty].subs({
                        (prefix, x, y): (prefix, x + 2*mx, y + 3*my)
                        for prefix, x, y in get_maximising_inputs(target_array[tx, ty])
                    })
                    
                    delta = strip_affine_errors(target_filter - translated_filter)
                    assert delta == 0
    
    def test_plausible_maximisation(self, input_array, transform_coeffs,
                                    wavelet_index, wavelet_index_ho,
                                    dwt_depth, dwt_depth_ho):
        # Check that the test pattern does in fact appear to maximise the filter
        # output value within the limits of the affine arithmetic bounds
        value_min = -512
        value_max = 255
        signal_range = {"signal_min": value_min, "signal_max": value_max}
        
        for level, orients in transform_coeffs.items():
            for orient, target_array in orients.items():
                for tx in range(target_array.period[0]):
                    for ty in range(target_array.period[1]):
                        # Produce a test pattern
                        ts = make_analysis_maximising_pattern(
                            input_array,
                            target_array,
                            tx, ty,
                        )
                        
                        # Find the expected bounds for values in the targeted
                        # transform coefficient
                        new_tx, new_ty = ts.target
                        target_filter = target_array[new_tx, new_ty]
                        lower_bound, upper_bound = analysis_filter_bounds(target_filter)
                        lower_bound = lower_bound.subs(signal_range).constant
                        upper_bound = upper_bound.subs(signal_range).constant
                        
                        # Create a test picture and encode it with the VC-2
                        # pseudocode
                        test_pattern_picture, _ = ts.pattern.as_picture_and_slice(value_min, value_max)
                        height, width = test_pattern_picture.shape
                        test_pattern_picture = test_pattern_picture.tolist()
                        
                        value_maximised = encode_with_vc2(
                            test_pattern_picture,
                            width, height,
                            wavelet_index, wavelet_index_ho,
                            dwt_depth, dwt_depth_ho,
                        )[level][orient][new_ty][new_tx]
                        
                        # Check we get within 1% of the upper bound
                        assert upper_bound*0.99 <= value_maximised <= upper_bound


class TestMakeSynthesisMaximisingSignal(object):
    
    @pytest.fixture(scope="class")
    def wavelet_index(self):
        return WaveletFilters.haar_with_shift
    
    @pytest.fixture(scope="class")
    def wavelet_index_ho(self):
        return WaveletFilters.haar_with_shift
    
    @pytest.fixture(scope="class")
    def dwt_depth(self):
        return 1
    
    @pytest.fixture(scope="class")
    def dwt_depth_ho(self):
        return 1
    
    @pytest.fixture(scope="class")
    def analysis_input_array(self):
        return SymbolArray(2)
    
    @pytest.fixture(scope="class")
    def analysis_transform_output(self, analysis_input_array,
                                  wavelet_index, wavelet_index_ho,
                                  dwt_depth, dwt_depth_ho):
        h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
        v_filter_params = LIFTING_FILTERS[wavelet_index]
        
        return analysis_transform(
            h_filter_params, v_filter_params,
            dwt_depth, dwt_depth_ho,
            analysis_input_array,
        )
    
    @pytest.fixture(scope="class")
    def analysis_transform_coeff_arrays(self, analysis_transform_output):
        return analysis_transform_output[0]
    
    @pytest.fixture(scope="class")
    def synthesis_input_arrays(self, dwt_depth, dwt_depth_ho):
        return make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho)
    
    @pytest.fixture(scope="class")
    def synthesis_transform_output(self, synthesis_input_arrays,
                                   wavelet_index, wavelet_index_ho,
                                   dwt_depth, dwt_depth_ho):
        h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
        v_filter_params = LIFTING_FILTERS[wavelet_index]
        
        return synthesis_transform(
            h_filter_params, v_filter_params,
            dwt_depth, dwt_depth_ho,
            synthesis_input_arrays,
        )
    
    @pytest.fixture(scope="class")
    def synthesis_output_array(self, synthesis_transform_output):
        return synthesis_transform_output[0]
    
    @pytest.fixture(scope="class")
    def synthesis_intermediate_arrays(self, synthesis_transform_output):
        return synthesis_transform_output[1]
    
    def test_always_positive_pixel_coords(
        self, analysis_input_array, analysis_transform_coeff_arrays,
        synthesis_output_array, synthesis_intermediate_arrays,
    ):
        for (level, name), synthesis_target_array in synthesis_intermediate_arrays.items():
            for tx in range(synthesis_target_array.period[0]):
                for ty in range(synthesis_target_array.period[1]):
                    ts = make_synthesis_maximising_pattern(
                        analysis_input_array,
                        analysis_transform_coeff_arrays,
                        synthesis_target_array,
                        synthesis_output_array,
                        tx, ty,
                    )
                    new_tx, new_ty = ts.target
                    assert new_tx >= 0
                    assert new_ty >= 0
                    
                    # Ensure that, as promised, the returned test patterns
                    # don't use any negative pixel coordinates
                    assert ts.pattern.origin[0] >= 0
                    assert ts.pattern.origin[1] >= 0
                    
                    # Also ensure that the returned test pattern is as close to
                    # the edge of the picture as possible (that is, moving one
                    # multiple left or up moves us off the edge of the picture)
                    # or if not, that the target value is as close as possible
                    # to the edge in that dimension.
                    mx, my = ts.pattern_translation_multiple
                    tmx, tmy = ts.target_translation_multiple
                    assert ts.pattern.origin[0] < my or new_tx < my
                    assert ts.pattern.origin[1] < mx or new_ty < mx
    
    def test_translation_is_valid(
        self, analysis_input_array, analysis_transform_coeff_arrays,
        synthesis_output_array, synthesis_intermediate_arrays,
    ):
        for (level, name), synthesis_target_array in synthesis_intermediate_arrays.items():
            for tx in range(synthesis_target_array.period[0]):
                for ty in range(synthesis_target_array.period[1]):
                    ts = make_synthesis_maximising_pattern(
                        analysis_input_array,
                        analysis_transform_coeff_arrays,
                        synthesis_target_array,
                        synthesis_output_array,
                        tx, ty,
                    )
                    new_tx, new_ty = ts.target
                    mx, my = ts.pattern_translation_multiple
                    tmx, tmy = ts.target_translation_multiple
                    
                    def full_filter(synthesis_expression):
                        """
                        Given a synthesis filter expression in terms of
                        transform coefficients, substitute in the analysis
                        transform to return the filter expression in terms of
                        the input pixels.
                        """
                        return synthesis_expression.subs({
                            ((prefix, l, o), x, y): analysis_transform_coeff_arrays[l][o][x, y]
                            for (prefix, l, o), x, y in get_maximising_inputs(synthesis_expression)
                        })
                    
                    # Check that the translated values really are processed by
                    # an equivalent filter to the offset passed in
                    dx = (new_tx - tx) * mx // tmx
                    dy = (new_ty - ty) * my // tmy
                    full_target_value = full_filter(synthesis_target_array[tx, ty])
                    specified_filter_translated = full_target_value.subs({
                        (prefix, x, y): (prefix, x + dx, y + dy)
                        for prefix, x, y in get_maximising_inputs(full_target_value)
                    })
                    returned_filter = full_filter(synthesis_target_array[new_tx, new_ty])
                    
                    delta = strip_affine_errors(specified_filter_translated - returned_filter)
                    assert delta == 0
                    
                    # Check that an offset of (tmx, tmy) offsets the input
                    # array by (mx, my)
                    target_filter = full_filter(synthesis_target_array[tx + 2*tmx, ty + 3*tmy])
                    full_translated_value = full_filter(synthesis_target_array[tx, ty])
                    translated_filter = full_translated_value.subs({
                        (prefix, x, y): (prefix, x + 2*mx, y + 3*my)
                        for prefix, x, y in get_maximising_inputs(full_translated_value)
                    })
                    
                    delta = strip_affine_errors(target_filter - translated_filter)
                    assert delta == 0
    
    def test_plausible_maximisation_no_quantisation(
        self, analysis_input_array, analysis_transform_coeff_arrays,
        synthesis_output_array, synthesis_intermediate_arrays,
        wavelet_index, wavelet_index_ho,
        dwt_depth, dwt_depth_ho
    ):
        # Check that the test patterns do, in fact, appear to maximise the
        # filter output for quantisation-free filtering
        value_min = -512
        value_max = 255
        
        for tx in range(synthesis_output_array.period[0]):
            for ty in range(synthesis_output_array.period[1]):
                # Produce a test pattern
                ts = make_synthesis_maximising_pattern(
                    analysis_input_array,
                    analysis_transform_coeff_arrays,
                    synthesis_output_array,
                    synthesis_output_array,
                    tx, ty,
                )
                
                # Create a test picture and encode/decode it with the VC-2
                # pseudocode (with no quantisaton)
                test_pattern_picture, _ = ts.pattern.as_picture_and_slice(value_min, value_max)
                height, width = test_pattern_picture.shape
                test_pattern_picture = test_pattern_picture.tolist()
                
                new_tx, new_ty = ts.target
                kwargs = {
                    "width": width,
                    "height": height,
                    "wavelet_index": wavelet_index,
                    "wavelet_index_ho": wavelet_index_ho,
                    "dwt_depth": dwt_depth,
                    "dwt_depth_ho": dwt_depth_ho,
                }
                value_maximised = decode_with_vc2(
                    encode_with_vc2(test_pattern_picture, **kwargs),
                    **kwargs
                )[new_ty][new_tx]
                
                # Check the value was in fact maximised
                assert value_maximised == value_max
    
    def test_plausible_maximisation_with_quantisation(
        self, analysis_input_array, analysis_transform_coeff_arrays,
        synthesis_output_array, synthesis_intermediate_arrays,
        wavelet_index, wavelet_index_ho,
        dwt_depth, dwt_depth_ho
    ):
        # Check that the test patterns do, in fact, appear to make larger values
        # post-quantisation than would appear with a more straight-forward
        # scheme
        value_min = -512
        value_max = 511
        
        num_periods_made_larger_by_test_pattern = 0
        for tx in range(synthesis_output_array.period[0]):
            for ty in range(synthesis_output_array.period[1]):
                # Produce a test pattern
                ts = make_synthesis_maximising_pattern(
                    analysis_input_array,
                    analysis_transform_coeff_arrays,
                    synthesis_output_array,
                    synthesis_output_array,
                    tx, ty,
                )
                
                test_pattern_picture, _ = ts.pattern.as_picture_and_slice(value_min, value_max)
                height, width = test_pattern_picture.shape
                test_pattern_picture = test_pattern_picture.tolist()
                
                # Create picture where just the target pixel is set
                new_tx, new_ty = ts.target
                just_target_picture = [
                    [
                        int((x, y) == (new_tx, new_ty)) * value_max
                        for x in range(width)
                    ]
                    for y in range(height)
                ]
                
                kwargs = {
                    "width": width,
                    "height": height,
                    "wavelet_index": wavelet_index,
                    "wavelet_index_ho": wavelet_index_ho,
                    "dwt_depth": dwt_depth,
                    "dwt_depth_ho": dwt_depth_ho,
                }
                
                # Encode with VC-2 pseudocode
                test_pattern_coeffs = encode_with_vc2(test_pattern_picture, **kwargs)
                just_target_coeffs = encode_with_vc2(just_target_picture, **kwargs)
                
                # Try different quantisation levels and record the worst-case
                # effect on the target pixel
                test_pattern_value_worst_case = 0
                just_target_value_worst_case = 0
                for qi in range(64):
                    test_pattern_value = decode_with_vc2(quantise_coeffs(test_pattern_coeffs, qi), **kwargs)[new_ty][new_tx]
                    if abs(test_pattern_value) > abs(test_pattern_value_worst_case):
                        test_pattern_value_worst_case = test_pattern_value
                    
                    just_target_value = decode_with_vc2(quantise_coeffs(just_target_coeffs, qi), **kwargs)[new_ty][new_tx]
                    if abs(just_target_value) > abs(just_target_value_worst_case):
                        just_target_value_worst_case = just_target_value
                
                # Check the value was in fact made bigger in the worst case by
                # the new test pattern
                if abs(test_pattern_value_worst_case) > abs(just_target_value_worst_case):
                    num_periods_made_larger_by_test_pattern += 1

        # Check that, in the common case, make the worst-case values worse with
        # the test pattern than a single hot pixel would.
        num_periods = synthesis_output_array.period[0] * synthesis_output_array.period[1]
        assert num_periods_made_larger_by_test_pattern > (num_periods / 2.0)
