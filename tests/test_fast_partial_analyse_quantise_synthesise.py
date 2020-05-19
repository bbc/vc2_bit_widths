import pytest

import numpy as np

from itertools import cycle, islice

from vc2_data_tables import (
    LiftingFilterParameters,
    WaveletFilters,
    LIFTING_FILTERS,
    QUANTISATION_MATRICES,
)

from vc2_conformance.pseudocode.state import State
from vc2_conformance.pseudocode.picture_encoding import dwt
from vc2_conformance.pseudocode.picture_decoding import idwt

from vc2_bit_widths.pyexp import Argument

from vc2_bit_widths.quantisation import (
    forward_quant,
    inverse_quant,
)

from vc2_bit_widths.vc2_filters import (
    synthesis_transform,
    make_variable_coeff_arrays,
)

from vc2_bit_widths.fast_partial_analysis_transform import (
    fast_partial_analysis_transform,
)

from vc2_bit_widths.fast_partial_analyse_quantise_synthesise import (
    get_transform_coeffs_used_by_synthesis_exp,
    exp_coeff_nested_dicts_to_list,
    to_interleaved_transform_coord,
    transform_coeffs_to_index_array,
    make_quantisation_factor_sweep,
    apply_quantisation_sweep,
    FastPartialAnalyseQuantiseSynthesise,
)


def test_get_transform_coeffs_used_by_synthesis_exp():
    a = Argument("a")
    
    exp = (
        a[0]["L"][(1, 2)] +
        a[1]["H"][(2, 3)] +
        a[2]["LH"][(3, 4)] +
        123
    )
    
    assert set(get_transform_coeffs_used_by_synthesis_exp(exp)) == set([
        (0, "L", 1, 2),
        (1, "H", 2, 3),
        (2, "LH", 3, 4),
    ])


def test_exp_coeff_nested_dicts_to_list():
    a = Argument("a")
    
    dict_exp = (
        a[0]["L"][(1, 2)] -
        a[1]["H"][(2, 3)] +
        123
    )
    
    list_exp, transform_coeffs_used = exp_coeff_nested_dicts_to_list(dict_exp)
    
    coeffs_dict = {
        0: {"L": {(1, 2): 100}},
        1: {"H": {(2, 3): 1000}},
    }
    coeffs_list = [
        coeffs_dict[level][orient][x, y]
        for level, orient, x, y in transform_coeffs_used
    ]
    
    dict_f = dict_exp.make_function()
    list_f = list_exp.make_function()
    
    assert dict_f(a=coeffs_dict) == list_f(a=coeffs_list)


@pytest.mark.parametrize("dwt_depth,dwt_depth_ho", [
    # 2D only
    (1, 0),
    (3, 0),
    # H only
    (0, 1),
    (0, 3),
    # Mixed
    (1, 1),
    (2, 2),
])
def test_to_interleaved_transform_coord(dwt_depth, dwt_depth_ho):
    width = 32
    height = 16
    
    # Create a signal with unique values in every pixel
    interleaved = np.arange(width*height).reshape((height, width))
    
    # As a somewhat dirty trick we use the fast_partial_analysis_transform
    # (with a null filter) to produce a set of array views of a correctly
    # interleaved input array.
    null_filter = LiftingFilterParameters(filter_bit_shift=0, stages=[])
    subband_views = fast_partial_analysis_transform(
        null_filter,
        null_filter,
        dwt_depth,
        dwt_depth_ho,
        interleaved,
    )
    
    for level in subband_views:
        for orient in subband_views[level]:
            subband_view = subband_views[level][orient]
            for sb_y in range(subband_view.shape[0]):
                for sb_x in range(subband_view.shape[1]):
                    il_x, il_y = to_interleaved_transform_coord(
                        dwt_depth,
                        dwt_depth_ho,
                        level,
                        orient,
                        sb_x,
                        sb_y,
                    )
                    
                    assert interleaved[il_y, il_x] == subband_view[sb_y, sb_x]


def test_transform_coeffs_to_index_array():
    dwt_depth = 1
    dwt_depth_ho = 0
    
    transform_coeffs = [
        (0, "LL", 100, 1000),
        (1, "HL", 100, 1000),
        (1, "LH", 100, 1000),
        (1, "HH", 100, 1000),
    ]
    
    index_array = transform_coeffs_to_index_array(
        dwt_depth,
        dwt_depth_ho,
        transform_coeffs,
    )
    
    assert list(zip(*index_array)) == [
        (2000, 200),
        (2000, 201),
        (2001, 200),
        (2001, 201),
    ]
    
    assert index_array[0].dtype == np.intp
    assert index_array[1].dtype == np.intp


def test_make_quantisation_factor_sweep():
    quantisation_indices = [0, 4, 8]  # Factors 1, 2, 4
    
    quantisation_matrix = {0: {"L": 0}, 1: {"H": 4}}
    
    transform_coeff = [
        (0, "L", 0, 0),
        (1, "H", 0, 0),
        (0, "L", 1, 1),
        (1, "H", 1, 1),
    ]
    
    quantisation_factor_matrix, quantisation_offset_matrix = make_quantisation_factor_sweep(
        quantisation_indices,
        quantisation_matrix,
        transform_coeff,
    )
    
    assert np.array_equal(quantisation_factor_matrix, np.array([
        [4, 4, 4, 4],
        [8, 4, 8, 4],
        [16, 8, 16, 8],
    ]))
    
    assert np.array_equal(quantisation_offset_matrix, np.array([
        [1, 1, 1, 1],
        [4, 1, 4, 1],
        [8, 4, 8, 4],
    ]))


def test_apply_quantisation_sweep():
    quantisation_factor_matrix = np.array([
        [1, 1, 1, 1],
        [2, 1, 2, 1],
        [4, 2, 4, 2],
    ])
    quantisation_offset_matrix = np.array([
        [1, 1, 1, 1],
        [4, 1, 4, 1],
        [8, 4, 8, 4],
    ])
    
    transform_coeff_vector = np.array([128, 1024, -128, 0])
    
    transform_coeff_matrix = apply_quantisation_sweep(
        quantisation_factor_matrix,
        quantisation_offset_matrix,
        transform_coeff_vector,
    )
    
    assert transform_coeff_matrix.dtype == quantisation_factor_matrix.dtype
    assert np.array_equal(transform_coeff_matrix, np.array([
        [128, 1024, -128, 0],
        [129, 1024, -129, 0],
        [130, 1025, -130, 0],
    ]))


def quant_roundtrip(value, qi):
    return inverse_quant(forward_quant(value, qi), qi)


def test_quantisation_integration():
    dwt_depth = 1
    dwt_depth_ho = 2
    
    # This test checks that make_quantisation_factor_sweep and
    # apply_quantisation_sweep together produce answers consistent with the
    # behaviour of the pseudocode quantiser
    quantisation_matrix = QUANTISATION_MATRICES[(
        WaveletFilters.le_gall_5_3,
        WaveletFilters.le_gall_5_3,
        dwt_depth,
        dwt_depth_ho,
    )]
    
    quantisation_indices = list(range(64))
    
    # Some random values to quantise
    num_values = 100
    rand = np.random.RandomState(1)
    values = rand.randint(-10000, 10000, num_values)
    
    # Assign an arbitrary subband to every value
    transform_coeffs = list(islice(cycle([
        (
            level,
            (
                "L" if level == 0 else
                "H" if level <= dwt_depth_ho else
                orient_2d
            ),
            0,
            0,
        )
        for level in range(dwt_depth + dwt_depth_ho + 1)
        for orient_2d in ["HH", "HL", "LH"]
    ]), num_values))
    
    # Quantise the values using the pseudocode
    ref_quantised_values = np.array([
        [
            quant_roundtrip(
                value,
                max(0, qi - quantisation_matrix[level][orient]),
            )
            for value, (level, orient, x, y) in zip(
                values,
                transform_coeffs,
            )
        ]
        for qi in quantisation_indices
    ])
    
    # Quantise using the functions under test
    qf_matrix, qo_matrix = make_quantisation_factor_sweep(
        quantisation_indices,
        quantisation_matrix,
        transform_coeffs,
    )
    fut_quantised_values = apply_quantisation_sweep(
        qf_matrix,
        qo_matrix,
        values,
    )
    
    # Verify that the two outputs are identical
    assert np.array_equal(ref_quantised_values, fut_quantised_values)


@pytest.mark.parametrize("wavelet_index,wavelet_index_ho", [
    # The two combinations below ensure that the arguments haven't got swapped
    # or duplicated anywhere in the implementation
    (WaveletFilters.haar_no_shift, WaveletFilters.haar_with_shift),
    (WaveletFilters.haar_with_shift, WaveletFilters.haar_no_shift),
])
def test_fast_partial_analyse_quantise_synthesise(wavelet_index, wavelet_index_ho):
    # For this test we compare the output of the pseudocode encoder with the
    # partial decoder and check that they agree. To simplify this test, the
    # Haar transform is used which is edge effect free and therefore any/every
    # pixel decoded by this implementation should exactly match to pseudocode.
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    dwt_depth = 1
    dwt_depth_ho = 2
    
    # We're using the wrong matrix here (since the default matrices don't
    # include this type of asymmetric filter) but this is unimportant since any
    # matrix will do...
    quantisation_matrix = QUANTISATION_MATRICES[(
        wavelet_index,
        wavelet_index,
        dwt_depth,
        dwt_depth_ho,
    )]
    
    # This should be enough to get to the point where all transform
    # coefficients are zero in this test
    quantisation_indices = list(range(64))
    
    # Create a test image
    width = 16
    height = 4
    rand = np.random.RandomState(1)
    picture = rand.randint(-512, 511, (height, width))
    
    # Encode the picture using the pseudocode
    state = State(
        wavelet_index=wavelet_index,
        wavelet_index_ho=wavelet_index_ho,
        dwt_depth=dwt_depth,
        dwt_depth_ho=dwt_depth_ho,
    )
    pseudocode_coeffs = dwt(state, picture.tolist())
    
    # Quantise/Decode using the pseudocode at each quantisation level in turn
    # to generate 'model answers' for every pixel
    pseudocode_decoded_pictures = []
    for qi in quantisation_indices:
        pseudocode_decoded_pictures.append(idwt(state, {
            level: {
                orient: [
                    [
                        quant_roundtrip(
                            value,
                            max(0, qi - quantisation_matrix[level][orient]),
                        )
                        for value in row
                    ]
                    for row in array
                ]
                for orient, array in orients.items()
            }
            for level, orients in pseudocode_coeffs.items()
        }))
    
    # Create decoder function expressions
    synthesis_expressions, _ = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        make_variable_coeff_arrays(dwt_depth, dwt_depth_ho)
    )
    
    # Check decoding of every pixel individually matches the pseudocode at
    # every quantisation level.
    for y in range(height):
        for x in range(width):
            codec = FastPartialAnalyseQuantiseSynthesise(
                h_filter_params,
                v_filter_params,
                dwt_depth,
                dwt_depth_ho,
                quantisation_matrix,
                quantisation_indices,
                synthesis_expressions[x, y],
            )
            decoded_values = codec.analyse_quantise_synthesise(picture.copy())
            for reference_decoded_picture, decoded_value in zip(
                pseudocode_decoded_pictures,
                decoded_values,
            ):
                assert decoded_value == reference_decoded_picture[y][x]
