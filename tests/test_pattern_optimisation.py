import pytest

import numpy as np

from decode_and_quantise_test_utils import (
    encode_with_vc2,
    quantise_coeffs,
    decode_with_vc2,
)

from vc2_bit_widths.pyexp import Argument

from vc2_bit_widths.infinite_arrays import SymbolArray

from vc2_bit_widths.vc2_filters import (
    make_symbol_coeff_arrays,
    make_variable_coeff_arrays,
    analysis_transform,
    synthesis_transform,
)

from vc2_bit_widths.fast_partial_analyse_quantise_synthesise import (
    FastPartialAnalyseQuantiseSynthesise,
)

from vc2_data_tables import LIFTING_FILTERS, WaveletFilters, QUANTISATION_MATRICES

from vc2_bit_widths.pattern_generation import TestPatternSpecification as TPS

from vc2_bit_widths.pattern_generation import (
    make_synthesis_maximising_signal,
)

from vc2_bit_widths.pattern_optimisation import OptimisedTestPatternSpecification as OTPS

from vc2_bit_widths.pattern_optimisation import (
    choose_random_indices_of,
    greedy_stochastic_search,
    optimise_synthesis_maximising_test_pattern,
)


def test_choose_random_indices_of():
    rand = np.random.RandomState(1)
    array = np.empty((2, 3, 4))
    
    ia = choose_random_indices_of(array, 1000, rand)
    
    assert len(ia) == 3
    
    assert len(ia[0]) == 1000
    assert len(ia[1]) == 1000
    assert len(ia[2]) == 1000
    
    assert set(ia[0]) == set(range(2))
    assert set(ia[1]) == set(range(3))
    assert set(ia[2]) == set(range(4))


def test_greedy_stochastic_search():
    # This test is fairly crude. It just tests that the search terminates, does
    # not crash and improves on the initial input signal, however slightly.
    
    wavelet_index = WaveletFilters.le_gall_5_3
    wavelet_index_ho = WaveletFilters.le_gall_5_3
    dwt_depth = 1
    dwt_depth_ho = 0
    
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    
    quantisation_matrix = QUANTISATION_MATRICES[(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
    )]
    
    quantisation_indices = list(range(64))

    input_min = -511
    input_max = 256

    width = 16
    height = 8
    
    # Arbitrary test pattern
    rand = np.random.RandomState(1)
    input_pattern = rand.choice((input_min, input_max), (height, width))
    
    # Restrict search-space to bottom-right three quarters only
    search_slice = (slice(height//4, None), slice(width//4, None))
    
    # Arbitrary output pixel
    output_array, intermediate_arrays = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        make_variable_coeff_arrays(dwt_depth, dwt_depth_ho),
    )
    synthesis_pyexp = output_array[width//2, height//2]
    
    codec = FastPartialAnalyseQuantiseSynthesise(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        quantisation_indices,
        synthesis_pyexp,
    )
    
    kwargs = {
        "starting_pattern": input_pattern.copy(),
        "search_slice": search_slice,
        "input_min": input_min,
        "input_max": input_max,
        "codec": codec,
        "random_state": np.random.RandomState(1),
        "added_corruptions_per_iteration": 1,
        "removed_corruptions_per_iteration": 1,
        "added_iterations_per_improvement": 50,
    }
    
    # Get the baseline after no searches performed
    base_input_pattern, base_decoded_value, base_qi, decoded_values = greedy_stochastic_search(
        base_iterations=0,
        **kwargs
    )
    assert decoded_values == []
    
    # Check that when run for some time we get an improved result
    new_input_pattern, new_decoded_value, new_qi, decoded_values = greedy_stochastic_search(
        base_iterations=100,
        **kwargs
    )
    assert not np.array_equal(new_input_pattern, base_input_pattern)
    assert abs(new_decoded_value) > abs(base_decoded_value)
    assert new_qi in quantisation_indices
    assert len(decoded_values) > 100
    assert decoded_values[-1] == new_decoded_value
    
    # Check haven't mutated the supplied starting pattern argument
    assert np.array_equal(kwargs["starting_pattern"], input_pattern)
    
    # Check that only the specified slice was modified
    before = input_pattern.copy()
    before[search_slice] = 0
    after = new_input_pattern.copy()
    after[search_slice] = 0
    assert np.array_equal(before, after)
    
    # Check that all values are in range
    assert np.all((new_input_pattern >= input_min) & (new_input_pattern <= input_max))


class TestOptimiseSynthesisMaximisingSignal(object):
    
    @pytest.fixture
    def wavelet_index(self):
        return WaveletFilters.le_gall_5_3
    
    @pytest.fixture
    def filter_params(self, wavelet_index):
        return LIFTING_FILTERS[wavelet_index]
    
    @pytest.fixture
    def dwt_depth(self):
        return 1
    
    @pytest.fixture
    def dwt_depth_ho(self):
        return 1
    
    @pytest.fixture
    def quantisation_matrix(self, wavelet_index, dwt_depth, dwt_depth_ho):
        return QUANTISATION_MATRICES[(
            wavelet_index,
            wavelet_index,
            dwt_depth,
            dwt_depth_ho,
        )]
    
    @pytest.fixture
    def analysis_input_linexp_array(self):
        return SymbolArray(2)
    
    @pytest.fixture
    def analysis_coeff_linexp_arrays(
        self,
        filter_params,
        dwt_depth,
        dwt_depth_ho,
        analysis_input_linexp_array,
    ):
        return analysis_transform(
            filter_params, filter_params,
            dwt_depth, dwt_depth_ho,
            analysis_input_linexp_array,
        )[0]
    
    @pytest.fixture
    def synthesis_input_linexp_arrays(self, dwt_depth, dwt_depth_ho):
        return make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho)
    
    @pytest.fixture
    def synthesis_output_linexp_array(
        self,
        filter_params,
        dwt_depth,
        dwt_depth_ho,
        synthesis_input_linexp_arrays,
    ):
        return synthesis_transform(
            filter_params, filter_params,
            dwt_depth, dwt_depth_ho,
            synthesis_input_linexp_arrays,
        )[0]
    
    @pytest.fixture
    def synthesis_output_linexp_array(
        self,
        filter_params,
        dwt_depth,
        dwt_depth_ho,
        synthesis_input_linexp_arrays,
    ):
        return synthesis_transform(
            filter_params, filter_params,
            dwt_depth, dwt_depth_ho,
            synthesis_input_linexp_arrays,
        )[0]
    
    @pytest.fixture
    def synthesis_transform_output(
        self,
        filter_params,
        dwt_depth,
        dwt_depth_ho,
    ):
        synthesis_input_pyexp_arrays = make_variable_coeff_arrays(dwt_depth, dwt_depth_ho)
        return synthesis_transform(
            filter_params, filter_params,
            dwt_depth, dwt_depth_ho,
            synthesis_input_pyexp_arrays,
        )
    
    @pytest.fixture
    def synthesis_output_pyexp_array(self, synthesis_transform_output):
        return synthesis_transform_output[0]
    
    @pytest.fixture
    def synthesis_intermediate_pyexps(self, synthesis_transform_output):
        return synthesis_transform_output[1]
    
    def test_correctness(
        self,
        wavelet_index,
        filter_params,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        analysis_input_linexp_array,
        analysis_coeff_linexp_arrays,
        synthesis_output_linexp_array,
        synthesis_output_pyexp_array,
    ):
        # This test will simply attempt to maximise a real test pattern and verify
        # that the procedure appears to produce an improved result.
        
        max_quantisation_index = 63
        
        input_min = -512
        input_max = 511
        
        # Make an arbitrary choice of target to maximise
        synthesis_target_linexp_array = synthesis_output_linexp_array
        synthesis_target_pyexp_array = synthesis_output_pyexp_array
        
        # Run against all filter phases as some phases may happen to be maximised
        # by the test pattern anyway
        num_improved_phases = 0
        for tx in range(synthesis_target_linexp_array.period[0]):
            for ty in range(synthesis_target_linexp_array.period[1]):
                # Produce test pattern
                ts = make_synthesis_maximising_signal(
                    analysis_input_linexp_array,
                    analysis_coeff_linexp_arrays,
                    synthesis_target_linexp_array,
                    synthesis_output_linexp_array,
                    tx, ty,
                )
                
                new_tx, new_ty = ts.target
                synthesis_pyexp = synthesis_target_pyexp_array[new_tx, new_ty]
                
                kwargs = {
                    "h_filter_params": filter_params,
                    "v_filter_params": filter_params,
                    "dwt_depth": dwt_depth,
                    "dwt_depth_ho": dwt_depth_ho,
                    "quantisation_matrix": quantisation_matrix,
                    "synthesis_pyexp": synthesis_pyexp,
                    "test_pattern": ts,
                    "input_min": input_min,
                    "input_max": input_max,
                    "max_quantisation_index": max_quantisation_index,
                    "random_state": np.random.RandomState(1),
                    "added_corruptions_per_iteration": (len(ts.pattern)+19)//20,  # 5%
                    "removed_corruptions_per_iteration": (len(ts.pattern)+8)//5,  # 20%
                    "added_iterations_per_improvement": 50,
                    "terminate_early": None,
                }
                
                # Run without any greedy searches to get 'baseline' figure
                base_ts = optimise_synthesis_maximising_test_pattern(
                    number_of_searches=1,
                    base_iterations=0,
                    **kwargs
                )
                
                # Verify unrelated test pattern parameters passed through
                assert base_ts.target == ts.target
                assert base_ts.pattern_translation_multiple == ts.pattern_translation_multiple
                assert base_ts.target_translation_multiple == ts.target_translation_multiple
                
                # Run with greedy search to verify better result
                imp_ts = optimise_synthesis_maximising_test_pattern(
                    number_of_searches=3,
                    base_iterations=100,
                    **kwargs
                )
                assert imp_ts.num_search_iterations >= 3 * 100
                
                # Ensure new test pattern is normalised to polarities
                for value in imp_ts.pattern.values():
                    assert value in (-1, +1)
                
                # Should have improved over the test pattern alone
                if abs(imp_ts.decoded_value) > abs(base_ts.decoded_value):
                    num_improved_phases += 1
                
                # Check to see if decoded value matches what the pseudocode decoder
                # would produce
                xs, ys = zip(*imp_ts.pattern)
                width = max(xs) + 1
                height = max(ys) + 1
                imp_test_pattern_picture = [
                    [
                        input_min
                        if imp_ts.pattern.get((x, y), 0) < 0 else
                        input_max
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
                            imp_test_pattern_picture,
                            **kwargs
                        ),
                        imp_ts.quantisation_index,
                        quantisation_matrix,
                    ),
                    **kwargs
                )[new_ty][new_tx]
                
                assert imp_ts.decoded_value == actual_decoded_value
                
        # Consider the procedure to work if some of the phases are improved
        assert num_improved_phases >= 1
    
    def test_terminate_early(
        self,
        synthesis_intermediate_pyexps,
        filter_params,
    ):
        input_min = -512
        input_max = 511
        
        ts = TPS(
            target=(0, 0),
            pattern={(0, 0): -1},
            pattern_translation_multiple=(0, 0),
            target_translation_multiple=(0, 0),
        )
        
        synthesis_pyexp = Argument("coeffs")[0]["LL"][0, 0]
        
        kwargs = {
            "h_filter_params": filter_params,
            "v_filter_params": filter_params,
            "dwt_depth": 0,
            "dwt_depth_ho": 0,
            "quantisation_matrix": {0: {"LL": 0}},
            "synthesis_pyexp": synthesis_pyexp,
            "test_pattern": ts,
            "input_min": input_min,
            "input_max": input_max,
            "max_quantisation_index": 1,
            "random_state": np.random.RandomState(1),
            "base_iterations": 100,
            "added_corruptions_per_iteration": 1,
            "removed_corruptions_per_iteration": 0,
            "added_iterations_per_improvement": 50,
            "number_of_searches": 10,
        }
        
        # Check that with terminate early disabled the search runs the full
        # base set of iterations but finds no improvement
        new_ts = optimise_synthesis_maximising_test_pattern(
            terminate_early=None,
            **kwargs
        )
        assert new_ts.pattern == ts.pattern
        assert new_ts.num_search_iterations == 100 * 10
        
        # Check terminate early allows early termination after the specified
        # number of iterations
        new_ts = optimise_synthesis_maximising_test_pattern(
            terminate_early=3,
            **kwargs
        )
        assert new_ts.pattern == ts.pattern
        assert new_ts.num_search_iterations == 100 * 3
        
        # Should run all iterations if an improvement is found
        kwargs["test_pattern"].pattern[(0, 0)] = +1
        new_ts = optimise_synthesis_maximising_test_pattern(
            terminate_early=3,
            **kwargs
        )
        assert new_ts.num_search_iterations >= 100 * 10


