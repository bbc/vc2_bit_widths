import pytest

import numpy as np

from vc2_data_tables import LIFTING_FILTERS, WaveletFilters, QUANTISATION_MATRICES

from vc2_conformance.state import State

from vc2_conformance.picture_encoding import forward_wavelet_transform
from vc2_conformance.picture_decoding import inverse_wavelet_transform

from vc2_bit_widths.quantisation import forward_quant, inverse_quant

from vc2_bit_widths.linexp import LinExp

import vc2_bit_widths.affine_arithmetic as aa

from vc2_bit_widths.infinite_arrays import SymbolArray

from vc2_bit_widths.vc2_filters import (
    make_coeff_arrays,
    analysis_transform,
    synthesis_transform,
)

from vc2_bit_widths.signal_bounds import analysis_filter_bounds

from vc2_bit_widths.signal_generation import (
    get_maximising_inputs,
    make_analysis_maximising_signal,
    make_synthesis_maximising_signal,
    partial_analysis_to_matrix,
    partial_synthesis_to_matrix,
    make_quantisation_factor_sweep,
    apply_quantisation_sweep,
    find_quantisation_index_with_greatest_output_magnitude,
    greedy_stochastic_search,
    improve_synthesis_maximising_signal,
)


@pytest.mark.parametrize("expression,exp", [
    # Empty/constant expression
    (0, {}),
    (123, {}),
    (LinExp(123), {}),
    # Error terms should be omitted
    (aa.new_error_symbol(), {}),
    (aa.new_error_symbol() + 123, {}),
    # Terms should be extracted correctly
    (LinExp(("p", 0, 0)), {("p", 0, 0): 1}),
    # Only the signs of the coefficients should be preserved
    (-4*LinExp(("p", 0, 0)), {("p", 0, 0): -1}),
    # Combination of everything
    (
        2*LinExp(("p", 0, 0)) - 3*LinExp(("p", 1, 2)) + aa.new_error_symbol() + 123,
        {("p", 0, 0): 1, ("p", 1, 2): -1},
    ),
])
def test_get_maximising_inputs(expression, exp):
    assert get_maximising_inputs(expression) == exp


def encode_with_vc2(picture, width, height,
                    wavelet_index, wavelet_index_ho, dwt_depth, dwt_depth_ho):
    state = State(
        luma_width=width,
        luma_height=height,
        color_diff_width=0,
        color_diff_height=0,
        
        wavelet_index=wavelet_index,
        wavelet_index_ho=wavelet_index_ho,
        dwt_depth=dwt_depth,
        dwt_depth_ho=dwt_depth_ho,
    )
    
    current_picture = {
        "Y": picture,
        "C1": [],
        "C2": [],
    }
    
    forward_wavelet_transform(state, current_picture)
    
    return state["y_transform"]


def quantise_coeffs(transform_coeffs, qi, quantisation_matrix={}):
    return {
        level: {
            orient: [
                [
                    inverse_quant(
                        forward_quant(
                            value,
                            max(0, qi - quantisation_matrix.get(level, {}).get(orient, 0)),
                        ),
                        max(0, qi - quantisation_matrix.get(level, {}).get(orient, 0)),
                    )
                    for value in row
                ]
                for row in array
            ]
            for orient, array in orients.items()
        }
        for level, orients in transform_coeffs.items()
    }


def decode_with_vc2(transform_coeffs, width, height,
                    wavelet_index, wavelet_index_ho, dwt_depth, dwt_depth_ho):
    state = State(
        luma_width=width,
        luma_height=height,
        color_diff_width=0,
        color_diff_height=0,
        
        wavelet_index=wavelet_index,
        wavelet_index_ho=wavelet_index_ho,
        dwt_depth=dwt_depth,
        dwt_depth_ho=dwt_depth_ho,
        
        y_transform=transform_coeffs,
        c1_transform=transform_coeffs,  # Ignored
        c2_transform=transform_coeffs,  # Ignored
        
        current_picture={},
    )
    
    inverse_wavelet_transform(state)
    
    return state["current_picture"]["Y"]


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
                    test_signal, new_tx, new_ty, mx, my, tmx, tmy = make_analysis_maximising_signal(
                        input_array,
                        target_array,
                        tx, ty,
                    )
                    
                    assert new_tx >= 0
                    assert new_ty >= 0
                    
                    # Ensure that, as promised, the returned test patterns
                    # don't use any negative pixel coordinates
                    assert all(x >= 0 and y >= 0 for (x, y) in test_signal)
                    
                    # Also ensure that the returned test pattern is as close to
                    # the edge of the picture as possible (that is, moving one
                    # multiple left or up moves us off the edge of the picture)
                    assert any(x < mx for (x, y) in test_signal)
                    assert any(y < my for (x, y) in test_signal)
    
    def test_translation_is_valid(self, input_array, intermediate_arrays):
        for (level, name), target_array in intermediate_arrays.items():
            for tx in range(target_array.period[0]):
                for ty in range(target_array.period[1]):
                    test_signal, new_tx, new_ty, mx, my, tmx, tmy = make_analysis_maximising_signal(
                        input_array,
                        target_array,
                        tx, ty,
                    )
                    # Check that the translated values really are processed by
                    # an equivalent filter to the offset passed in
                    dx = (new_tx - tx) * mx // tmx
                    dy = (new_ty - ty) * my // tmy
                    specified_filter_translated = target_array[tx, ty].subs({
                        (prefix, x, y): (prefix, x + dx, y + dy)
                        for prefix, x, y in get_maximising_inputs(target_array[tx, ty])
                    })
                    returned_filter = target_array[new_tx, new_ty]
                    
                    delta = aa.strip_error_terms(specified_filter_translated - returned_filter)
                    assert delta == 0
                    
                    # Check that an offset of (tmx, tmy) offsets the input
                    # array by (mx, my)
                    target_filter = target_array[tx + 2*tmx, ty + 3*tmy]
                    translated_filter = target_array[tx, ty].subs({
                        (prefix, x, y): (prefix, x + 2*mx, y + 3*my)
                        for prefix, x, y in get_maximising_inputs(target_array[tx, ty])
                    })
                    
                    delta = aa.strip_error_terms(target_filter - translated_filter)
                    assert delta == 0
    
    def test_plausible_maximisation(self, input_array, transform_coeffs,
                                    wavelet_index, wavelet_index_ho,
                                    dwt_depth, dwt_depth_ho):
        # Check that the test signal does in fact appear to maximise the filter
        # output value within the limits of the affine arithmetic bounds
        value_min = -512
        value_max = 255
        
        for level, orients in transform_coeffs.items():
            for orient, target_array in orients.items():
                for tx in range(target_array.period[0]):
                    for ty in range(target_array.period[1]):
                        # Produce a test pattern
                        test_signal, new_tx, new_ty, mx, my, tmx, tmy = make_analysis_maximising_signal(
                            input_array,
                            target_array,
                            tx, ty,
                        )
                        
                        # Find the expected bounds for values in the targeted
                        # transform coefficient
                        target_filter = target_array[new_tx, new_ty]
                        lower_bound, upper_bound = analysis_filter_bounds(target_filter, value_min, value_max)
                        
                        # Create a test picture and encode it with the VC-2
                        # pseudocode
                        xs, ys = zip(*test_signal)
                        width = max(xs) + 1
                        height = max(ys) + 1
                        test_signal_picture = [
                            [
                                0
                                if (x, y) not in test_signal else
                                value_max
                                if test_signal[(x, y)] > 0 else
                                value_min
                                for x in range(width)
                            ]
                            for y in range(height)
                        ]
                        
                        value_maximised = encode_with_vc2(
                            test_signal_picture,
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
        return make_coeff_arrays(dwt_depth, dwt_depth_ho)
    
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
                    test_signal, new_tx, new_ty, mx, my, tmx, tmy = make_synthesis_maximising_signal(
                        analysis_input_array,
                        analysis_transform_coeff_arrays,
                        synthesis_target_array,
                        synthesis_output_array,
                        tx, ty,
                    )
                    assert new_tx >= 0
                    assert new_ty >= 0
                    
                    # Ensure that, as promised, the returned test patterns
                    # don't use any negative pixel coordinates
                    assert all(x >= 0 and y >= 0 for (x, y) in test_signal)
                    
                    # Also ensure that the returned test pattern is as close to
                    # the edge of the picture as possible (that is, moving one
                    # multiple left or up moves us off the edge of the picture)
                    # or if not, that the target value is as close as possible
                    # to the edge in that dimension.
                    assert any(x < mx or new_tx < tmx for (x, y) in test_signal)
                    assert any(y < my or new_ty < tmy for (x, y) in test_signal)
    
    def test_translation_is_valid(
        self, analysis_input_array, analysis_transform_coeff_arrays,
        synthesis_output_array, synthesis_intermediate_arrays,
    ):
        for (level, name), synthesis_target_array in synthesis_intermediate_arrays.items():
            for tx in range(synthesis_target_array.period[0]):
                for ty in range(synthesis_target_array.period[1]):
                    test_signal, new_tx, new_ty, mx, my, tmx, tmy = make_synthesis_maximising_signal(
                        analysis_input_array,
                        analysis_transform_coeff_arrays,
                        synthesis_target_array,
                        synthesis_output_array,
                        tx, ty,
                    )
                    
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
                    
                    delta = aa.strip_error_terms(specified_filter_translated - returned_filter)
                    assert delta == 0
                    
                    # Check that an offset of (tmx, tmy) offsets the input
                    # array by (mx, my)
                    target_filter = full_filter(synthesis_target_array[tx + 2*tmx, ty + 3*tmy])
                    full_translated_value = full_filter(synthesis_target_array[tx, ty])
                    translated_filter = full_translated_value.subs({
                        (prefix, x, y): (prefix, x + 2*mx, y + 3*my)
                        for prefix, x, y in get_maximising_inputs(full_translated_value)
                    })
                    
                    delta = aa.strip_error_terms(target_filter - translated_filter)
                    assert delta == 0
    
    def test_plausible_maximisation_no_quantisation(
        self, analysis_input_array, analysis_transform_coeff_arrays,
        synthesis_output_array, synthesis_intermediate_arrays,
        wavelet_index, wavelet_index_ho,
        dwt_depth, dwt_depth_ho
    ):
        # Check that the test signals do, in fact, appear to maximise the
        # filter output for quantisation-free filtering
        value_min = -512
        value_max = 255
        
        for tx in range(synthesis_output_array.period[0]):
            for ty in range(synthesis_output_array.period[1]):
                # Produce a test pattern
                test_signal, new_tx, new_ty, mx, my, tmx, tmy = make_synthesis_maximising_signal(
                    analysis_input_array,
                    analysis_transform_coeff_arrays,
                    synthesis_output_array,
                    synthesis_output_array,
                    tx, ty,
                )
                
                # Create a test picture and encode/decode it with the VC-2
                # pseudocode (with no quantisaton)
                xs, ys = zip(*test_signal)
                width = max(xs) + 1
                height = max(ys) + 1
                test_signal_picture = [
                    [
                        0
                        if (x, y) not in test_signal else
                        value_max
                        if test_signal[(x, y)] > 0 else
                        value_min
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
                value_maximised = decode_with_vc2(
                    encode_with_vc2(test_signal_picture, **kwargs),
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
        # Check that the test signals do, in fact, appear to make larger values
        # post-quantisation than would appear with a more straight-forward
        # scheme
        value_min = -512
        value_max = 511
        
        num_periods_made_larger_by_test_signal = 0
        for tx in range(synthesis_output_array.period[0]):
            for ty in range(synthesis_output_array.period[1]):
                # Produce a test pattern
                test_signal, new_tx, new_ty, mx, my, tmx, tmy = make_synthesis_maximising_signal(
                    analysis_input_array,
                    analysis_transform_coeff_arrays,
                    synthesis_output_array,
                    synthesis_output_array,
                    tx, ty,
                )
                
                xs, ys = zip(*test_signal)
                width = max(xs) + 1
                height = max(ys) + 1
                
                # Create test pattern picture
                test_signal_picture = [
                    [
                        0
                        if (x, y) not in test_signal else
                        value_max
                        if test_signal[(x, y)] > 0 else
                        value_min
                        for x in range(width)
                    ]
                    for y in range(height)
                ]
                
                # Create picture where just the target pixel is set
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
                test_signal_coeffs = encode_with_vc2(test_signal_picture, **kwargs)
                just_target_coeffs = encode_with_vc2(just_target_picture, **kwargs)
                
                # Try different quantisation levels and record the worst-case
                # effect on the target pixel
                test_signal_value_worst_case = 0
                just_target_value_worst_case = 0
                for qi in range(64):
                    test_signal_value = decode_with_vc2(quantise_coeffs(test_signal_coeffs, qi), **kwargs)[new_ty][new_tx]
                    if abs(test_signal_value) > abs(test_signal_value_worst_case):
                        test_signal_value_worst_case = test_signal_value
                    
                    just_target_value = decode_with_vc2(quantise_coeffs(just_target_coeffs, qi), **kwargs)[new_ty][new_tx]
                    if abs(just_target_value) > abs(just_target_value_worst_case):
                        just_target_value_worst_case = just_target_value
                
                # Check the value was in fact made bigger in the worst case by
                # the new test signal
                if abs(test_signal_value_worst_case) > abs(just_target_value_worst_case):
                    num_periods_made_larger_by_test_signal += 1

        # Check that, in the common case, make the worst-case values worse with
        # the test signal than a single hot pixel would.
        num_periods = synthesis_output_array.period[0] * synthesis_output_array.period[1]
        assert num_periods_made_larger_by_test_signal > (num_periods / 2.0)


def test_partial_analysis_to_matrix():
    input_symbols = ["p0", "p1", "p2"]
    transform_coeff_symbols = ["l", "h"]
    transform_coeff_expressions = {
        "l": (LinExp("p0") + LinExp("p1") + LinExp("p2"))/2,
        "h": LinExp("p0") - LinExp("p1"),
    }
    
    analysis_matrix = partial_analysis_to_matrix(
        input_symbols,
        transform_coeff_symbols,
        transform_coeff_expressions,
    )
    
    input_vector = np.array([100, -200, 300])
    
    matrix_result = dict(zip(
        transform_coeff_symbols,
        np.matmul(analysis_matrix, input_vector),
    ))
    
    direct_result = {
        sym: expr.subs(dict(zip(input_symbols, input_vector))).constant
        for sym, expr in transform_coeff_expressions.items()
    }
    
    assert matrix_result == direct_result


def test_partial_synthesis_to_matrix():
    transform_coeff_symbols = ["l", "h"]
    synthesis_expression = LinExp("l") - (2*LinExp("h"))
    
    synthesis_matrix = partial_synthesis_to_matrix(
        transform_coeff_symbols,
        synthesis_expression,
    )
    
    transform_coeffs = np.array([100, -200])
    
    matrix_result = np.matmul(synthesis_matrix, transform_coeffs)
    
    direct_result = synthesis_expression.subs(
        dict(zip(transform_coeff_symbols, transform_coeffs))
    ).constant
    
    assert matrix_result == direct_result


def test_make_quantisation_factor_sweep():
    quantisation_indices = [0, 4, 8]  # Factors 1, 2, 4
    
    quantisation_matrix = {0: {"L": 0}, 1: {"H": 4}}
    
    transform_coeff_symbols = [
        (("coeff", 0, "L"), 0, 0),
        (("coeff", 1, "H"), 0, 0),
        (("coeff", 0, "L"), 1, 1),
        (("coeff", 1, "H"), 1, 1),
    ]
    
    quantisation_factor_matrix = make_quantisation_factor_sweep(
        quantisation_indices,
        quantisation_matrix,
        transform_coeff_symbols,
    )
    
    assert np.array_equal(quantisation_factor_matrix, np.array([
        [1, 2, 4],
        [1, 1, 2],
        [1, 2, 4],
        [1, 1, 2],
    ]))


def test_apply_quantisation_sweep():
    quantisation_factor_matrix = np.array([
        [1, 2, 4],
        [1, 1, 2],
        [1, 2, 4],
        [1, 1, 2],
    ])
    
    transform_coeff_vector = np.array([
        128,
        1024,
        -128,
        0
    ])
    
    transform_coeff_matrix = apply_quantisation_sweep(
        quantisation_factor_matrix,
        transform_coeff_vector,
    )
    
    assert np.array_equal(transform_coeff_matrix, np.array([
        [ 128,  129,  130],
        [1024, 1024, 1025],
        [-128, -129, -130],
        [   0,    0,    0],
    ]))


def test_find_quantisation_index_with_greatest_output_magnitude():
    input_vector = [100, -200, 300, -400]
    
    analysis_matrix = np.array([
        [-1, -1, -1, -1],
        [0,   1, -1,  0],
    ])
    
    # Transform coeffs = [
    #    200,
    #    -500,
    # ]
    
    quantisation_indices = [0, 4, 8]
    quantisation_factor_matrix = np.array([
        [1, 2, 4],
        [1, 1, 2],
    ])
    
    # Quantised transform coeffs = [
    #    [ 200,  201,  202].
    #    [-500, -500, -501],
    # ]
    
    synthesis_matrix = np.array([[-1, 1]])
    
    # Decoded values = [[
    #    -700,
    #    -701,
    #    -703,
    # ]]
    
    decoded_value, qi = find_quantisation_index_with_greatest_output_magnitude(
        input_vector,
        analysis_matrix,
        quantisation_indices,
        quantisation_factor_matrix,
        synthesis_matrix,
    )
    
    assert decoded_value == -703
    assert qi == 8


def test_greedy_stochastic_search():
    # This test is fairly crude. It just tests that the search terminates, does
    # not crash and improves on the initial input signal, however slightly.
    
    # The initial filter and inputs used here are essentially arbitrary and
    # chosen randomly.
    input_min = -512
    input_max = 255
    input_vector = np.array([input_max, input_min, input_max, input_min])
    analysis_matrix = np.array([
        [-1, 1, 1, -1],
        [ 0, 1,-1,  0],
    ])
    quantisation_indices = [0, 4, 8, 12, 16]
    quantisation_factor_matrix = np.array([
        [1, 2, 4, 8, 16],
        [1, 1, 2, 4,  8],
    ])
    synthesis_matrix = np.array([[-1, 1]])
    
    # Determine the initial input performance (without random search)
    base_decoded_value, base_qi = find_quantisation_index_with_greatest_output_magnitude(
        input_vector,
        analysis_matrix,
        quantisation_indices,
        quantisation_factor_matrix,
        synthesis_matrix,
    )
    
    kwargs = {
        "starting_input_vector": input_vector,
        "input_min": input_min,
        "input_max": input_max,
        "analysis_matrix": analysis_matrix,
        "quantisation_indices": quantisation_indices,
        "quantisation_factor_matrix": quantisation_factor_matrix,
        "synthesis_matrix": synthesis_matrix,
        "random_state": np.random.RandomState(1),
        "added_corruptions_per_iteration": 1,
        "removed_corruptions_per_iteration": 1,
        "added_iterations_per_improvement": 50,
    }
    
    # Check that a zero-iteration search produces identical results
    new_input_vector, new_decoded_value, new_qi = greedy_stochastic_search(
        base_iterations=0,
        **kwargs
    )
    assert np.array_equal(new_input_vector, input_vector)
    assert new_decoded_value == base_decoded_value
    assert new_qi == base_qi
    
    # Check that when run for some time we get an improved result
    new_input_vector, new_decoded_value, new_qi = greedy_stochastic_search(
        base_iterations=100,
        **kwargs
    )
    assert not np.array_equal(new_input_vector, input_vector)
    assert abs(new_decoded_value) > abs(base_decoded_value)
    assert new_qi in quantisation_indices
    
    # Check that the result is in range
    for value in new_input_vector:
        assert input_min <= value <= input_max


def test_improve_synthesis_maximising_signal():
    # This test will simply attempt to maximise a real filter test signal and
    # verify that the procedure appears to produce an improved result.
    
    # Create filter descriptions (of an arbtirary, smallish filter)
    wavelet_index = WaveletFilters.le_gall_5_3
    filter_params = LIFTING_FILTERS[wavelet_index]
    dwt_depth = 1
    dwt_depth_ho = 1
    
    # Arbitrary choice of 10 bit signal range
    input_min = -512
    input_max = 511
    
    max_quantisation_index = 63
    quantisation_matrix = QUANTISATION_MATRICES[(
        wavelet_index,
        wavelet_index,
        dwt_depth,
        dwt_depth_ho,
    )]
    
    analysis_input_array = SymbolArray(2)
    analysis_coeff_arrays, analysis_intermediate_values = analysis_transform(
        filter_params, filter_params,
        dwt_depth, dwt_depth_ho,
        analysis_input_array,
    )
    
    synthesis_input_arrays = make_coeff_arrays(dwt_depth, dwt_depth_ho)
    synthesis_output_array, synthesis_intermediate_arrays = synthesis_transform(
        filter_params, filter_params,
        dwt_depth, dwt_depth_ho,
        synthesis_input_arrays,
    )
    
    # Make an arbitrary choice of target to maximise
    synthesis_target_array = synthesis_output_array
    
    # Run against all filter phases as some phases may happen to be maximised
    # by the test signal anyway
    num_improved_phases = 0
    for tx in range(synthesis_target_array.period[0]):
        for ty in range(synthesis_target_array.period[1]):
            # Produce test signal
            test_signal, new_tx, new_ty, mx, my, tmx, tmy = make_synthesis_maximising_signal(
                analysis_input_array,
                analysis_coeff_arrays,
                synthesis_target_array,
                synthesis_output_array,
                tx, ty,
            )
            
            synthesis_target_expression = synthesis_target_array[new_tx, new_ty]
            
            kwargs = {
                "test_signal": test_signal,
                "analysis_transform_coeff_arrays": analysis_coeff_arrays,
                "synthesis_target_expression": synthesis_target_expression,
                "input_min": input_min,
                "input_max": input_max,
                "max_quantisation_index": max_quantisation_index,
                "quantisation_matrix": quantisation_matrix,
                "random_state": np.random.RandomState(1),
                "added_corruptions_per_iteration": (len(test_signal)+19)//20,  # 5%
                "removed_corruptions_per_iteration": (len(test_signal)+8)//5,  # 20%
                "added_iterations_per_improvement": 50,
            }
            
            # Run without any greedy searches to get 'baseline' figure
            base_test_signal, base_decoded_value, base_qi = improve_synthesis_maximising_signal(
                number_of_searches=1,
                base_iterations=0,
                **kwargs
            )
            
            # Run with greedy search to verify better result
            imp_test_signal, imp_decoded_value, imp_qi = improve_synthesis_maximising_signal(
                number_of_searches=10,
                base_iterations=100,
                **kwargs
            )
            
            # Ensure new test signal is in range
            for value in imp_test_signal.values():
                assert input_min <= value <= input_max
            
            if abs(imp_decoded_value) > abs(base_decoded_value):
                num_improved_phases += 1
            
            # Check to see if estimate of target value is plausible
            xs, ys = zip(*imp_test_signal)
            width = max(xs) + 1
            height = max(ys) + 1
            imp_test_signal_picture = [
                [
                    imp_test_signal.get((x, y), 0)
                    for x in range(width)
                ]
                for y in range(height)
            ]
            kwargs = {
                "width": width,
                "height": height,
                "wavelet_index": wavelet_index,
                "wavelet_index_ho": wavelet_index,
                "dwt_depth": dwt_depth,
                "dwt_depth_ho": dwt_depth_ho,
            }
            actual_decoded_value = decode_with_vc2(
                quantise_coeffs(
                    encode_with_vc2(
                        imp_test_signal_picture,
                        **kwargs
                    ),
                    imp_qi,
                    quantisation_matrix,
                ),
                **kwargs
            )[new_ty][new_tx]
            
            assert np.isclose(imp_decoded_value, actual_decoded_value, rtol=0.9)
            
    # Consider the procedure to work if at least half of the filter phases are
    # improved by the greedy search
    num_phases = float(synthesis_target_array.period[0] * synthesis_target_array.period[1])
    assert num_improved_phases / num_phases > 0.5
