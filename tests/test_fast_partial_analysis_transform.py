import pytest

import numpy as np

import vc2_data_tables as tables

from vc2_conformance.state import State

from vc2_conformance.picture_encoding import dwt

from vc2_conformance.picture_decoding import SYNTHESIS_LIFTING_FUNCTION_TYPES

from vc2_bit_widths.linexp import strip_affine_errors

from vc2_bit_widths.infinite_arrays import SymbolArray

from vc2_bit_widths.vc2_filters import (
    analysis_transform,
    convert_between_synthesis_and_analysis,
)

from vc2_bit_widths.fast_partial_analysis_transform import (
    compute_lifting_stage_slices,
    h_stage,
    fast_partial_analysis_transform,
)


class TestComputeLiftingStageSlices(object):
    
    def test_haar(self):
        # The Haar transform has the special property of never requiring any
        # signal values outside of the input signal. As such it makes a good
        # simple test case.
        
        filter_params = tables.LIFTING_FILTERS[tables.WaveletFilters.haar_no_shift]
        
        # Haar synthesis stage 0:
        #
        #  Used by tap:         @       @     ...   @       @       @
        #
        #                   0   1   2   3              n-4 n-3 n-2 n-1 
        #                 +---+---+---+---+--       --+---+---+---+---+
        #  Signal:        |   |   |   |   | ...   ... |   |   |   |   |
        #                 +---+---+---+---+--       --+---+---+---+---+
        #                   +---'   +---'               +---'   +---'
        #                   |       |                   |       |
        #                 +-------+-------+--       --+-------+-------+
        #  Accumulator:   |       |       | ...   ... |       |       |
        #                 +-------+-------+--       --+-------+-------+
        #                     0       1                 n/2-2   n/2-1
        slices = compute_lifting_stage_slices(filter_params.stages[0])
        
        assert slices.signal_update_slice == slice(0, -1, 2)
        assert slices.accumulator_slice == slice(0, None)
        assert slices.tap_signal_slices == [
            slice(1, None, 2),
        ]
        
        # Haar synthesis stage 1:
        #
        #  Used by tap:     @       @         ...       @       @
        #
        #                   0   1   2   3              n-4 n-3 n-2 n-1 
        #                 +---+---+---+---+--       --+---+---+---+---+
        #  Signal:        |   |   |   |   | ...   ... |   |   |   |   |
        #                 +---+---+---+---+--       --+---+---+---+---+
        #                   '---+   '---+               '---+   '---+
        #                       |       |                   |       |
        #                 +-------+-------+--       --+-------+-------+
        #  Accumulator:   |       |       | ...   ... |       |       |
        #                 +-------+-------+--       --+-------+-------+
        #                     0       1                 n/2-2   n/2-1
        slices = compute_lifting_stage_slices(filter_params.stages[1])
        
        assert slices.signal_update_slice == slice(1, None, 2)
        assert slices.accumulator_slice == slice(0, None)
        assert slices.tap_signal_slices == [
            slice(0, -1, 2),
        ]
    
    def test_deslauriers_dubuc_13_7(self):
        # As a more general test of filters which overshoot the end of the
        # signal, Deslauriers Dubuc (13, 7) is used since it overshoots both
        # ends of the signal by either 1 or 2 samples. It is also the running
        # example used in the code's documentation so a good case to verify!
        
        filter_params = tables.LIFTING_FILTERS[tables.WaveletFilters.deslauriers_dubuc_13_7]
        
        # Deslauriers Dubuc synthesis stage 0: (not the running example)
        #
        #  Used by tap 0:       @       @       @       @       @    ...    @       @
        #  Used by tap 1:               @       @       @       @    ...    @       @       @
        #  Used by tap 2:                       @       @       @    ...    @       @       @       @
        #  Used by tap 3:                               @       @    ...    @       @       @       @       @
        #
        #                   0   1   2   3   4   5   6   7   8              n-9 n-8 n-7 n-6 n-5 n-4 n-3 n-2 n-1 
        #                 +---+---+---+---+---+---+---+---+---+--       --+---+---+---+---+---+---+---+---+---+
        #  Signal:        |   |   |   |   |   |   |   |   |   | ...   ... |   |   |   |   |   |   |   |   |   |
        #                 +---+---+---+---+---+---+---+---+---+--       --+---+---+---+---+---+---+---+---+---+
        #                       '-------'---+---'-------'           '-------'---+---'-------'
        #                               '-------'---+---'-------'           '-------'---+---'-------'
        #                                   |       |                           |   '-------'---+---'-------'
        #                                   |       |                           |       |       |
        #                 +-------+-------+-------+-------+------       ------+-------+-------+-------+-------+
        #  Accumulator:   |XXXXXXX|XXXXXXX|       |       |     ...   ...     |       |       |       |XXXXXXX|
        #                 +-------+-------+-------+-------+------       ------+-------+-------+-------+-------+
        #                     0       1       2       3       4         n/2-5   n/2-4   n/2-3   n/2-2   n/2-1
        slices = compute_lifting_stage_slices(filter_params.stages[0])
        
        assert slices.signal_update_slice == slice(4, -3, 2)
        assert slices.accumulator_slice == slice(2, -1)
        assert slices.tap_signal_slices == [
            slice(1, -6, 2),
            slice(3, -4, 2),
            slice(5, -2, 2),
            slice(7, None, 2),
        ]
        
        # Deslauriers Dubuc synthesis stage 1: (the running example)
        #
        #  Used by tap 0:   @       @       @       @       @    ...    @       @
        #  Used by tap 1:           @       @       @       @    ...    @       @       @
        #  Used by tap 2:                   @       @       @    ...    @       @       @       @
        #  Used by tap 3:                           @       @    ...    @       @       @       @       @
        #
        #                   0   1   2   3   4   5   6   7   8              n-9 n-8 n-7 n-6 n-5 n-4 n-3 n-2 n-1 
        #                 +---+---+---+---+---+---+---+---+---+--       --+---+---+---+---+---+---+---+---+---+
        #  Signal:        |   |   |   |   |   |   |   |   |   | ...   ... |   |   |   |   |   |   |   |   |   |
        #                 +---+---+---+---+---+---+---+---+---+--       --+---+---+---+---+---+---+---+---+---+
        #                   '-------'---+---'-------'                   '-------'---+---'-------'
        #                           '-------'---+---'-------'                   '-------'---+---'-------'
        #                               |   '-------'---+---'-------'               |       |
        #                               |       |       |                           |       |
        #                 +-------+-------+-------+-------+------       ------+-------+-------+-------+-------+
        #  Accumulator:   |XXXXXXX|       |       |       |     ...   ...     |       |       |XXXXXXX|XXXXXXX|
        #                 +-------+-------+-------+-------+------       ------+-------+-------+-------+-------+
        #                     0       1       2       3       4         n/2-5   n/2-4   n/2-3   n/2-2   n/2-1
        slices = compute_lifting_stage_slices(filter_params.stages[1])
        
        assert slices.signal_update_slice == slice(3, -4, 2)
        assert slices.accumulator_slice == slice(1, -2)
        assert slices.tap_signal_slices == [
            slice(0, -7, 2),
            slice(2, -5, 2),
            slice(4, -3, 2),
            slice(6, -1, 2),
        ]


@pytest.mark.parametrize("stage", [
    stage
    for filter_params in tables.LIFTING_FILTERS.values()
    for stage in convert_between_synthesis_and_analysis(filter_params).stages
])
def test_h_stage(stage):
    # Check that the array operation produces the same results as the
    # pseudocode lifting operations do for all lifting steps used in the
    # standard.
    stage_slices = compute_lifting_stage_slices(stage)
    
    width = 32
    height = 4
    
    rand = np.random.RandomState(1)
    data = rand.randint(-512, 511, (height, width))
    
    # Process with pseudocode
    lift = SYNTHESIS_LIFTING_FUNCTION_TYPES[stage.lift_type]
    pseudocode_out = data.copy()
    for row in pseudocode_out:
        lift(row, stage.L, stage.D, stage.taps, stage.S)
    
    # Process with matrix lifting
    matrix_out = data.copy()
    acc1 = np.empty_like(data[:, ::2])
    acc2 = np.empty_like(data[:, ::2])
    h_stage(matrix_out, acc1, acc2, stage, stage_slices)
    
    # Compare values free from edge-effects (in updated samples
    assert np.array_equal(
        pseudocode_out[:, stage_slices.signal_update_slice],
        matrix_out[:, stage_slices.signal_update_slice],
    )
    
    # Compare values in samples which shouldn't have changed
    unmodified_start = 0 if (
        stage.lift_type == tables.LiftingFilterTypes.odd_add_even or
        stage.lift_type == tables.LiftingFilterTypes.odd_subtract_even
    ) else 1
    assert np.array_equal(
        pseudocode_out[:, unmodified_start::2],
        matrix_out[:, unmodified_start::2],
    )


@pytest.mark.parametrize("wavelet_index,wavelet_index_ho,dwt_depth,dwt_depth_ho", [
    # Check shift is taken from the correct transform
    (tables.WaveletFilters.haar_no_shift, tables.WaveletFilters.haar_no_shift, 1, 0),
    (tables.WaveletFilters.haar_no_shift, tables.WaveletFilters.haar_with_shift, 1, 0),
    (tables.WaveletFilters.haar_with_shift, tables.WaveletFilters.haar_no_shift, 1, 0),
    (tables.WaveletFilters.haar_with_shift, tables.WaveletFilters.haar_with_shift, 1, 0),
    # Check depths work independently
    (tables.WaveletFilters.haar_no_shift, tables.WaveletFilters.haar_no_shift, 0, 1),
    (tables.WaveletFilters.haar_no_shift, tables.WaveletFilters.haar_no_shift, 1, 0),
    (tables.WaveletFilters.haar_no_shift, tables.WaveletFilters.haar_no_shift, 1, 2),
    # A comparatively less simple filter which includes some
    # non-edge-effect-free outputs
    (tables.WaveletFilters.le_gall_5_3, tables.WaveletFilters.le_gall_5_3, 1, 0),
    # Check asymmetric transforms work
    (tables.WaveletFilters.haar_no_shift, tables.WaveletFilters.le_gall_5_3, 1, 2),
])
def test_fast_partial_analysis_transform(wavelet_index, wavelet_index_ho, dwt_depth, dwt_depth_ho):
    # This test verifies the analysis transform produces identical results to
    # the pseudocode in the case where no edge effects are encountered.
    
    width = 32
    height = 8
    
    rand = np.random.RandomState(1)
    signal = rand.randint(-512, 511, (height, width))
    
    # Process using pseudocode
    state = State(
        wavelet_index=wavelet_index,
        wavelet_index_ho=wavelet_index_ho,
        dwt_depth=dwt_depth,
        dwt_depth_ho=dwt_depth_ho,
    )
    pseudocode_out = dwt(state, signal.tolist())
    
    # Process using matrix transform
    h_filter_params = tables.LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = tables.LIFTING_FILTERS[wavelet_index]
    matrix_out = fast_partial_analysis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        signal.copy(),
    )
    
    # Using a symbolic representation of the transform operation and use this
    # to create masks identifying the coefficients which are edge-effect free.
    symbolic_out, _ = analysis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        SymbolArray(2),
    )
    edge_effect_free_pixel_mask = {
        level: {
            orient: np.array([
                [
                    all(
                        0 <= sym[1] < width and 0 <= sym[2] < height
                        for sym in strip_affine_errors(symbolic_out[level][orient][col, row]).symbols()
                        if sym is not None
                    )
                    for col in range(matrix_out[level][orient].shape[1])
                ]
                for row in range(matrix_out[level][orient].shape[0])
            ])
            for orient in matrix_out[level]
        }
        for level in matrix_out
    }
    
    # Sanity check: Ensure that in every transform subband there is at least
    # one edge-effect free value (otherwise the test needs to be modified to
    # use a larger input picture.
    assert all(
        np.any(mask)
        for level, orients in edge_effect_free_pixel_mask.items()
        for orient, mask in orients.items()
    )
    
    # Compare the two outputs and ensure all edge-effect free pixels are
    # identical
    assert set(pseudocode_out) == set(matrix_out)
    for level in matrix_out:
        assert set(pseudocode_out[level]) == set(matrix_out[level])
        for orient in matrix_out[level]:
            pseudocode_array = np.array(pseudocode_out[level][orient])
            matrix_array = matrix_out[level][orient]
            mask = edge_effect_free_pixel_mask[level][orient]
            assert np.array_equal(pseudocode_array[mask], matrix_array[mask])
