import pytest

import numpy as np

from decode_and_quantise_test_utils import (
    encode_with_vc2,
    quantise_coeffs,
    decode_with_vc2,
)

from vc2_data_tables import LIFTING_FILTERS, WaveletFilters, QUANTISATION_MATRICES

from vc2_bit_widths.linexp import (
    LinExp,
    strip_affine_errors,
)

from vc2_bit_widths.infinite_arrays import SymbolArray

from vc2_bit_widths.vc2_filters import (
    make_symbol_coeff_arrays,
    make_variable_coeff_arrays,
    analysis_transform,
    synthesis_transform,
)

from vc2_bit_widths.signal_bounds import (
    analysis_filter_bounds,
    synthesis_filter_bounds,
    evaluate_analysis_filter_bounds,
    evaluate_synthesis_filter_bounds,
    signed_integer_range,
)

from vc2_bit_widths.fast_partial_analyse_quantise_synthesise import (
    FastPartialAnalyseQuantiseSynthesise,
)

# NB: Can't have any class names starting with 'Test' in a Pytest test file!
from vc2_bit_widths.pattern_generation import TestPatternSpecification as TPS
from vc2_bit_widths.pattern_optimisation import OptimisedTestPatternSpecification as OTPS

from vc2_bit_widths.pattern_optimisation import (
    optimise_synthesis_maximising_test_pattern,
)

from vc2_bit_widths.pattern_generation import (
    invert_test_pattern_specification,
    get_maximising_inputs,
    make_analysis_maximising_signal,
    make_synthesis_maximising_signal,
    convert_test_pattern_to_array_and_slice,
    evaluate_analysis_test_pattern_output,
    evaluate_synthesis_test_pattern_output,
)


class TestInvertTestPatternSpecification(object):
    
    def test_test_pattern(self):
        ts = TPS(
            target=(1, 2),
            pattern={
                (3, 4): +1,
                (5, 6): -1,
            },
            pattern_translation_multiple=(7, 8),
            target_translation_multiple=(9, 10),
        )
        
        its = invert_test_pattern_specification(ts)
        
        assert its == TPS(
            target=(1, 2),
            pattern={
                (3, 4): -1,
                (5, 6): +1,
            },
            pattern_translation_multiple=(7, 8),
            target_translation_multiple=(9, 10),
        )
    
    def test_optimised_test_pattern(self):
        ts = OTPS(
            target=(1, 2),
            pattern={
                (3, 4): +1,
                (5, 6): -1,
            },
            pattern_translation_multiple=(7, 8),
            target_translation_multiple=(9, 10),
            quantisation_index=11,
            decoded_value=12,
            num_search_iterations=13,
        )
        
        its = invert_test_pattern_specification(ts)
        
        assert its == OTPS(
            target=(1, 2),
            pattern={
                (3, 4): -1,
                (5, 6): +1,
            },
            pattern_translation_multiple=(7, 8),
            target_translation_multiple=(9, 10),
            quantisation_index=11,
            decoded_value=12,
            num_search_iterations=13,
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
                    ts = make_analysis_maximising_signal(
                        input_array,
                        target_array,
                        tx, ty,
                    )
                    
                    new_tx, new_ty = ts.target
                    assert new_tx >= 0
                    assert new_ty >= 0
                    
                    # Ensure that, as promised, the returned test patterns
                    # don't use any negative pixel coordinates
                    assert all(x >= 0 and y >= 0 for (x, y) in ts.pattern)
                    
                    # Also ensure that the returned test pattern is as close to
                    # the edge of the picture as possible (that is, moving one
                    # multiple left or up moves us off the edge of the picture)
                    mx, my = ts.pattern_translation_multiple
                    assert any(x < mx for (x, y) in ts.pattern)
                    assert any(y < my for (x, y) in ts.pattern)
    
    def test_translation_is_valid(self, input_array, intermediate_arrays):
        for (level, name), target_array in intermediate_arrays.items():
            for tx in range(target_array.period[0]):
                for ty in range(target_array.period[1]):
                    ts = make_analysis_maximising_signal(
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
                        ts = make_analysis_maximising_signal(
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
                        xs, ys = zip(*ts.pattern)
                        width = max(xs) + 1
                        height = max(ys) + 1
                        test_pattern_picture = [
                            [
                                0
                                if (x, y) not in ts.pattern else
                                value_max
                                if ts.pattern[(x, y)] > 0 else
                                value_min
                                for x in range(width)
                            ]
                            for y in range(height)
                        ]
                        
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
                    ts = make_synthesis_maximising_signal(
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
                    assert all(x >= 0 and y >= 0 for (x, y) in ts.pattern)
                    
                    # Also ensure that the returned test pattern is as close to
                    # the edge of the picture as possible (that is, moving one
                    # multiple left or up moves us off the edge of the picture)
                    # or if not, that the target value is as close as possible
                    # to the edge in that dimension.
                    mx, my = ts.pattern_translation_multiple
                    tmx, tmy = ts.target_translation_multiple
                    assert any(x < mx or new_tx < tmx for (x, y) in ts.pattern)
                    assert any(y < my or new_ty < tmy for (x, y) in ts.pattern)
    
    def test_translation_is_valid(
        self, analysis_input_array, analysis_transform_coeff_arrays,
        synthesis_output_array, synthesis_intermediate_arrays,
    ):
        for (level, name), synthesis_target_array in synthesis_intermediate_arrays.items():
            for tx in range(synthesis_target_array.period[0]):
                for ty in range(synthesis_target_array.period[1]):
                    ts = make_synthesis_maximising_signal(
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
                ts = make_synthesis_maximising_signal(
                    analysis_input_array,
                    analysis_transform_coeff_arrays,
                    synthesis_output_array,
                    synthesis_output_array,
                    tx, ty,
                )
                
                # Create a test picture and encode/decode it with the VC-2
                # pseudocode (with no quantisaton)
                xs, ys = zip(*ts.pattern)
                width = max(xs) + 1
                height = max(ys) + 1
                test_pattern_picture = [
                    [
                        0
                        if (x, y) not in ts.pattern else
                        value_max
                        if ts.pattern[(x, y)] > 0 else
                        value_min
                        for x in range(width)
                    ]
                    for y in range(height)
                ]
                
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
                ts = make_synthesis_maximising_signal(
                    analysis_input_array,
                    analysis_transform_coeff_arrays,
                    synthesis_output_array,
                    synthesis_output_array,
                    tx, ty,
                )
                
                xs, ys = zip(*ts.pattern)
                width = max(xs) + 1
                height = max(ys) + 1
                
                # Create test pattern picture
                test_pattern_picture = [
                    [
                        0
                        if (x, y) not in ts.pattern else
                        value_max
                        if ts.pattern[(x, y)] > 0 else
                        value_min
                        for x in range(width)
                    ]
                    for y in range(height)
                ]
                
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


class TestConvertTestPatternToArrayAndSlice(object):
    
    def test_conversion_to_array(self):
        test_pattern, search_slice = convert_test_pattern_to_array_and_slice(
            # NB: Coordinates are (x, y) here
            test_pattern={
                (2, 3): +1,
                (4, 5): -1,
            },
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
        test_pattern, search_slice = convert_test_pattern_to_array_and_slice(
            # NB: Coordinates are (x, y) here
            test_pattern={
                (6, 6): +1,
            },
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
                    num_bits=picture_bit_width,
                )
                
                # Create a test pattern
                test_pattern = make_analysis_maximising_signal(
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
                test_pattern = make_synthesis_maximising_signal(
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
