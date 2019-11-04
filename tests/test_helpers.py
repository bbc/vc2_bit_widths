import pytest

import numpy as np

from collections import defaultdict, OrderedDict

from vc2_data_tables import WaveletFilters

from vc2_conformance.state import State

from vc2_conformance.picture_encoding import dwt
from vc2_conformance.picture_decoding import idwt

from vc2_bit_widths.quantisation import forward_quant, inverse_quant

from vc2_bit_widths.helpers import (
    static_filter_analysis,
    evaluate_filter_bounds,
    quantisation_index_bound,
    optimise_synthesis_test_signals,
    evaluate_test_signal_outputs,
    add_omitted_synthesis_values,
)


# This set of tests are largely simplistic integration tests and basically feed
# an arbitrary but small filter into the functions and check the output against
# the behaviour of the VC-2 pseudocode.


def test_static_filter_analysis_and_evaluate_filter_bounds():
    wavelet_index = WaveletFilters.haar_with_shift
    wavelet_index_ho = WaveletFilters.le_gall_5_3
    dwt_depth = 1
    dwt_depth_ho = 0
    picture_bit_width = 10
    
    input_min = -512
    input_max = 511
    
    (
        analysis_signal_bounds,
        synthesis_signal_bounds,
        analysis_test_signals,
        synthesis_test_signals,
    ) = static_filter_analysis(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
    )
    
    (
        concrete_analysis_signal_bounds,
        concrete_synthesis_signal_bounds,
    ) = evaluate_filter_bounds(
        analysis_signal_bounds,
        synthesis_signal_bounds,
        picture_bit_width,
    )
    
    # Analysis/synthesis intermediate arrays should be consistent...
    assert set(analysis_signal_bounds) == set(analysis_test_signals) == set(concrete_analysis_signal_bounds)
    assert set(synthesis_signal_bounds) == set(synthesis_test_signals) == set(concrete_synthesis_signal_bounds)
    
    # As a sanity check, ensure that the analysis test signals provided reach
    # the ranges expected when encoded by the pseudocode encoder
    state = State(
        wavelet_index=wavelet_index,
        wavelet_index_ho=wavelet_index_ho,
        dwt_depth=dwt_depth,
        dwt_depth_ho=dwt_depth_ho,
    )
    for level, array_name in [(0, "LL"), (1, "LH"), (1, "HL"), (1, "HH")]:
        ts = analysis_test_signals[(level or 1, array_name, 0, 0)]
        exp_lower, exp_upper = concrete_analysis_signal_bounds[(level or 1, array_name, 0, 0)]
        
        xs, ys = zip(*ts.picture)
        
        # NB: Pad the width/height to the appropriate picture size for this
        # transform
        x_multiple = 2**(dwt_depth + dwt_depth_ho)
        y_multiple = 2**(dwt_depth)
        width = (((max(xs) + 1) + x_multiple - 1) // x_multiple) * x_multiple
        height = (((max(ys) + 1) + y_multiple - 1) // y_multiple) * y_multiple
        
        picture = np.zeros((height, width), dtype=int)
        picture[(ys, xs)] = list(ts.picture.values())
        
        min_picture = picture.copy()
        min_picture[picture==+1] = input_min
        min_picture[picture==-1] = input_max
        
        max_picture = picture.copy()
        max_picture[picture==+1] = input_max
        max_picture[picture==-1] = input_min
        
        tx, ty = ts.target
        actual_lower = dwt(state, min_picture.tolist())[level][array_name][ty][tx]
        actual_upper = dwt(state, max_picture.tolist())[level][array_name][ty][tx]
        
        assert np.isclose(actual_lower, exp_lower, rtol=0.1)
        assert np.isclose(actual_upper, exp_upper, rtol=0.1)
    
    # As a sanity check, ensure that the synthesis test signals provided
    # maximise the decoder output (when no quantisation is used) in the
    # pseudocode encoder/decoder
    for level, array_name, x, y in synthesis_test_signals:
        # Only check the final decoder output (as the pseudocode doesn't
        # provide access to other arrays)
        if level != dwt_depth + dwt_depth_ho or array_name != "Output":
            continue
        
        ts = synthesis_test_signals[(level, array_name, x, y)]
        
        xs, ys = zip(*ts.picture)
        
        # NB: Pad the width/height to the appropriate picture size for this
        # transform
        x_multiple = 2**(dwt_depth + dwt_depth_ho)
        y_multiple = 2**(dwt_depth)
        width = (((max(xs) + 1) + x_multiple - 1) // x_multiple) * x_multiple
        height = (((max(ys) + 1) + y_multiple - 1) // y_multiple) * y_multiple
        
        picture = np.zeros((height, width), dtype=int)
        picture[(ys, xs)] = list(ts.picture.values())
        
        min_picture = picture.copy()
        min_picture[picture==+1] = input_min
        min_picture[picture==-1] = input_max
        
        max_picture = picture.copy()
        max_picture[picture==+1] = input_max
        max_picture[picture==-1] = input_min
        
        tx, ty = ts.target
        actual_lower = idwt(state, dwt(state, min_picture.tolist()))[ty][tx]
        actual_upper = idwt(state, dwt(state, max_picture.tolist()))[ty][tx]
        
        assert actual_lower == input_min
        assert actual_upper == input_max


def test_quantisation_index_bound():
    wavelet_index = WaveletFilters.haar_with_shift
    wavelet_index_ho = WaveletFilters.haar_with_shift
    dwt_depth = 1
    dwt_depth_ho = 0
    
    quantisation_matrix = {
        0: {"LL": 10},
        1: {"LH": 10, "HL": 10, "HH": 10},
    }
    
    picture_bit_width = 10
    
    (
        analysis_signal_bounds,
        synthesis_signal_bounds,
        analysis_test_signals,
        synthesis_test_signals,
    ) = static_filter_analysis(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
    )
    
    (
        concrete_analysis_signal_bounds,
        concrete_synthesis_signal_bounds,
    ) = evaluate_filter_bounds(
        analysis_signal_bounds,
        synthesis_signal_bounds,
        picture_bit_width,
    )
    
    max_qi = quantisation_index_bound(
        concrete_analysis_signal_bounds,
        quantisation_matrix,
    )
    
    max_coeff_magnitude = max(
        max(abs(lower), abs(upper))
        for lower, upper in concrete_analysis_signal_bounds.values()
    )
    
    assert forward_quant(max_coeff_magnitude, max_qi - 10) == 0
    assert forward_quant(max_coeff_magnitude, max_qi - 11) != 0


def test_optimise_synthesis_test_signals():
    wavelet_index = WaveletFilters.haar_with_shift
    wavelet_index_ho = WaveletFilters.le_gall_5_3
    dwt_depth = 1
    dwt_depth_ho = 0
    
    quantisation_matrix = {
        0: {"LL": 0},
        1: {"LH": 0, "HL": 0, "HH": 0},
    }
    
    picture_bit_width = 10
    input_min = -512
    input_max = 511
    
    (
        analysis_signal_bounds,
        synthesis_signal_bounds,
        analysis_test_signals,
        synthesis_test_signals,
    ) = static_filter_analysis(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
    )
    
    (
        concrete_analysis_signal_bounds,
        concrete_synthesis_signal_bounds,
    ) = evaluate_filter_bounds(
        analysis_signal_bounds,
        synthesis_signal_bounds,
        picture_bit_width,
    )
    
    max_quantisation_index = quantisation_index_bound(
        concrete_analysis_signal_bounds,
        quantisation_matrix,
    )
    
    random_state = np.random.RandomState(1)
    number_of_searches = 2
    terminate_early = None
    added_corruption_rate = 0.5
    removed_corruption_rate = 0.0
    base_iterations = 10
    added_iterations_per_improvement = 1
    
    optimised_synthesis_test_signals = optimise_synthesis_test_signals(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        picture_bit_width,
        synthesis_test_signals,
        max_quantisation_index,
        random_state,
        number_of_searches,
        terminate_early,
        added_corruption_rate,
        removed_corruption_rate,
        base_iterations,
        added_iterations_per_improvement,
    )
    
    # As a sanity check, ensure that the test signals provided do produce the
    # values they say they do at the quantisation indices claimed
    state = State(
        wavelet_index=wavelet_index,
        wavelet_index_ho=wavelet_index_ho,
        dwt_depth=dwt_depth,
        dwt_depth_ho=dwt_depth_ho,
    )
    for level, array_name, x, y in optimised_synthesis_test_signals:
        # Only check the final decoder output (as the pseudocode doesn't
        # provide access to other arrays)
        if level != dwt_depth + dwt_depth_ho or array_name != "Output":
            continue
        
        ts = optimised_synthesis_test_signals[(level, array_name, x, y)]
        
        xs, ys = zip(*ts.picture)
        
        # NB: Pad the width/height to the appropriate picture size for this
        # transform
        x_multiple = 2**(dwt_depth + dwt_depth_ho)
        y_multiple = 2**(dwt_depth)
        width = (((max(xs) + 1) + x_multiple - 1) // x_multiple) * x_multiple
        height = (((max(ys) + 1) + y_multiple - 1) // y_multiple) * y_multiple
        
        picture = np.zeros((height, width), dtype=int)
        picture[(ys, xs)] = list(ts.picture.values())
        picture[picture==+1] = input_max
        picture[picture==-1] = input_min
        
        tx, ty = ts.target
        coeffs = dwt(state, picture.tolist())
        
        qi = ts.quantisation_index
        quantised_coeffs = {
            level: {
                orient: [
                    [
                        inverse_quant(forward_quant(value, qi), qi)
                        for value in row
                    ]
                    for row in rows
                ]
                for orient, rows in orients.items()
            }
            for level, orients in coeffs.items()
        }
        
        decoded_picture = idwt(state, quantised_coeffs)
        
        assert decoded_picture[ty][tx] == ts.decoded_value


@pytest.mark.parametrize("wavelet_index,wavelet_index_ho,dwt_depth,dwt_depth_ho", [
    # No-shift (2D and HO)
    (WaveletFilters.haar_no_shift, WaveletFilters.haar_no_shift, 2, 0),
    (WaveletFilters.haar_no_shift, WaveletFilters.haar_no_shift, 0, 2),
    # A multi-level filter, testing 2D and 1D transforms
    (WaveletFilters.haar_with_shift, WaveletFilters.le_gall_5_3, 1, 1),
    # A filter which has a different number of lifting stages to usual
    (WaveletFilters.haar_with_shift, WaveletFilters.daubechies_9_7, 0, 1),
])
def test_add_omitted_synthesis_values(
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
):
    quantisation_matrix = defaultdict(lambda: defaultdict(lambda: 0))
    picture_bit_width = 10
    max_quantisation_index = 0
    random_state = np.random.RandomState()
    
    (_, _, _, test_signals) = static_filter_analysis(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
    )
    
    _, evaluated_test_signals = evaluate_test_signal_outputs(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        picture_bit_width,
        quantisation_matrix,
        max_quantisation_index,
        {},
        test_signals,
    )
    
    # No-op optimisation to work out which arrays are retained
    optimised_test_signals = optimise_synthesis_test_signals(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        picture_bit_width,
        test_signals,
        max_quantisation_index,
        random_state,
        number_of_searches=1,
        terminate_early=0,
        added_corruption_rate=0,
        removed_corruption_rate=0,
        base_iterations=0,
        added_iterations_per_improvement=0,
    )
    
    # Make a copy of the synthesis test signals with only those produced by the
    # optimiser retained
    subset_evaluated_test_signals = OrderedDict(
        (key, evaluated_test_signals[key])
        for key in optimised_test_signals
    )
    
    # Re-add the remaining entries
    desubsetted_evaluated_test_signals = add_omitted_synthesis_values(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        subset_evaluated_test_signals,
    )
    
    assert evaluated_test_signals == desubsetted_evaluated_test_signals


def test_evaluate_test_signal_outputs():
    wavelet_index = WaveletFilters.haar_with_shift
    wavelet_index_ho = WaveletFilters.le_gall_5_3
    dwt_depth = 1
    dwt_depth_ho = 0
    
    quantisation_matrix = {
        0: {"LL": 0},
        1: {"LH": 1, "HL": 2, "HH": 3},
    }
    
    picture_bit_width = 10
    input_min = -512
    input_max = 511
    
    (
        analysis_signal_bounds,
        synthesis_signal_bounds,
        analysis_test_signals,
        synthesis_test_signals,
    ) = static_filter_analysis(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
    )
    
    (
        concrete_analysis_signal_bounds,
        concrete_synthesis_signal_bounds,
    ) = evaluate_filter_bounds(
        analysis_signal_bounds,
        synthesis_signal_bounds,
        picture_bit_width,
    )
    
    max_quantisation_index = quantisation_index_bound(
        concrete_analysis_signal_bounds,
        quantisation_matrix,
    )
    
    # Run a null-search to determine the actual decoded output of the synthesis
    # test patterns generated... (bodge)
    random_state = np.random.RandomState(1)
    number_of_searches = 1
    terminate_early = None
    added_corruption_rate = 0
    removed_corruption_rate = 0.0
    base_iterations = 0
    added_iterations_per_improvement = 0
    optimised_synthesis_test_signals = optimise_synthesis_test_signals(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        picture_bit_width,
        synthesis_test_signals,
        max_quantisation_index,
        random_state,
        number_of_searches,
        terminate_early,
        added_corruption_rate,
        removed_corruption_rate,
        base_iterations,
        added_iterations_per_improvement,
    )
    
    (
        analysis_test_signal_outputs,
        synthesis_test_signal_outputs,
    ) = evaluate_test_signal_outputs(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        picture_bit_width,
        quantisation_matrix,
        max_quantisation_index,
        analysis_test_signals,
        optimised_synthesis_test_signals,
    )
    
    # Analysis values should be similar to the bounds
    for (level, array_name, x, y), (minimum, maximum) in analysis_test_signal_outputs.items():
        lower_bound, upper_bound = concrete_analysis_signal_bounds[(level, array_name, x, y)]
        
        assert np.isclose(minimum, lower_bound, 0.01)
        assert lower_bound <= minimum
        
        assert np.isclose(maximum, upper_bound, 0.01)
        assert maximum <= upper_bound
    
    # Synthesis values should be equal to the optimiser's values
    for (
        (level, array_name, x, y),
        ((minimum, min_qi), (maximum, max_qi)),
    ) in synthesis_test_signal_outputs.items():
        ts = optimised_synthesis_test_signals[(level, array_name, x, y)]
        
        assert ts.decoded_value == maximum
        assert ts.quantisation_index == max_qi
