import pytest

import numpy as np

from collections import namedtuple

import vc2_data_tables as tables

from vc2_conformance.decoder.transform_data_syntax import (
    quant_factor,
    inverse_quant,
)

from vc2_conformance.state import State

from vc2_conformance.picture_encoding import dwt
from vc2_conformance.picture_decoding import idwt

from vc2_bit_widths.linexp import LinExp, AAError

from vc2_bit_widths.pyexp import PyExp, Argument

from vc2_bit_widths.infinite_arrays import (
    SymbolArray,
    VariableArray,
)

from vc2_bit_widths.vc2_filters import (
    analysis_transform,
    synthesis_transform,
    make_coeff_arrays,
    make_symbol_coeff_arrays,
    make_variable_coeff_arrays,
)

class TestAnalysisAndSynthesisTransforms(object):

    @pytest.mark.parametrize("wavelet_index,wavelet_index_ho,dwt_depth,dwt_depth_ho", [
        # Check that when no shifting occurs, no constant error exists
        # (since all other rounding errors should cancel out due to lifting)
        (
            tables.WaveletFilters.haar_no_shift,
            tables.WaveletFilters.haar_no_shift,
            1,
            2,
        ),
        # Check:
        # * Asymmetric transforms
        # * Differing horizontal and vertical transform depths
        # * Bit shift is taken from the correct filter
        # * Remaining error terms which are normally cancelled during
        #   bit-shifts are within the predicted bounds
        (
            tables.WaveletFilters.haar_no_shift,  # Vertical only
            tables.WaveletFilters.le_gall_5_3,  # Horizontal only
            1,
            2,
        ),
        # Checks a filter which uses all four lifting types
        (
            tables.WaveletFilters.daubechies_9_7,
            tables.WaveletFilters.daubechies_9_7,
            1,
            0,
        ),
    ])
    def test_filters_invert_eachother(self, wavelet_index, wavelet_index_ho, dwt_depth, dwt_depth_ho):
        # Test that the analysis and synthesis filters invert each-other as a
        # check of consistency (and, indirectly, the correctness of the
        # analysis implementation and convert_between_synthesis_and_analysis)
        
        h_filter_params = tables.LIFTING_FILTERS[wavelet_index_ho]
        v_filter_params = tables.LIFTING_FILTERS[wavelet_index]
        
        input_picture = SymbolArray(2, "p")
        
        transform_coeffs, _ = analysis_transform(
            h_filter_params,
            v_filter_params,
            dwt_depth,
            dwt_depth_ho,
            input_picture,
        )
        output_picture, _ = synthesis_transform(
            h_filter_params,
            v_filter_params,
            dwt_depth,
            dwt_depth_ho,
            transform_coeffs,
        )
        
        # In this example, no quantisation is applied between the two filters.
        # As a consequence the only error terms arise from rounding errors in
        # the analysis and synthesis filters. Since this implementation does
        # not account for divisions of the same numbers producing the same
        # rounding errors, these rounding errors do not cancel out here.
        # However, aside from these terms, the input and output of the filters
        # should be identical.
        rounding_errors = output_picture[0, 0] - input_picture[0, 0]
        assert all(isinstance(sym, AAError) for sym in rounding_errors.symbols())
    
    @pytest.mark.parametrize("wavelet_index,wavelet_index_ho", [
        (tables.WaveletFilters.haar_no_shift, tables.WaveletFilters.haar_no_shift),
        (tables.WaveletFilters.haar_no_shift, tables.WaveletFilters.haar_with_shift),
        (tables.WaveletFilters.haar_with_shift, tables.WaveletFilters.haar_no_shift),
        (tables.WaveletFilters.haar_with_shift, tables.WaveletFilters.haar_with_shift),
    ])
    def test_filters_match_pseudocode(self, wavelet_index, wavelet_index_ho):
        # This test checks that the filters implement the same behaviour as the
        # VC-2 pseudocode, including compatible operation ordering. This test is
        # carried out on a relatively small Haar transform because::
        #
        # * The Haar transform is free from edge effects making the
        #   InfiniteArray implementation straight-forwardly equivalent to the
        #   pseudocode behaviour in all cases (not just non-edge cases)
        # * The Haar transform is available in a form with and without the bit
        #   shift so we can check that the bit shift parameter is used
        #   correctly and taken from the correct wavelet index.
        # * Using large transform depths or filters produces very large
        #   functions for analysis transforms under PyExp which can crash
        #   Python interpreters. (In practice they'll only ever be generated
        #   for synthesis transforms which produce small code even for large
        #   transforms)
        
        width = 16
        height = 8
        
        dwt_depth = 1
        dwt_depth_ho = 2
        
        # Create a random picture to analyse
        rand = np.random.RandomState(1)
        random_input_picture = rand.randint(-512, 511, (height, width))
        
        # Analyse using pseudocode
        state = State(
            wavelet_index=wavelet_index,
            wavelet_index_ho=wavelet_index_ho,
            dwt_depth=dwt_depth,
            dwt_depth_ho=dwt_depth_ho,
        )
        pseudocode_coeffs = dwt(state, random_input_picture.tolist())
        
        # Analyse using InfiniteArrays
        h_filter_params = tables.LIFTING_FILTERS[wavelet_index_ho]
        v_filter_params = tables.LIFTING_FILTERS[wavelet_index]
        ia_coeffs, _ = analysis_transform(
            h_filter_params,
            v_filter_params,
            dwt_depth,
            dwt_depth_ho,
            VariableArray(2, Argument("picture")),
        )
        
        # Compare analysis results
        for level in pseudocode_coeffs:
            for orient in pseudocode_coeffs[level]:
                pseudocode_data = pseudocode_coeffs[level][orient]
                for row, row_data in enumerate(pseudocode_data):
                    for col, pseudocode_value in enumerate(row_data):
                        # Create and call a function to compute this value via
                        # InfiniteArrays/PyExp
                        expr = ia_coeffs[level][orient][col, row]
                        f = expr.make_function()
                        # NB: Array is transposed to support (x, y) indexing
                        ia_value = f(random_input_picture.T)
                        
                        assert ia_value == pseudocode_value
        
        # Synthesise using pseudocode
        pseudocode_picture = idwt(state, pseudocode_coeffs)
        
        # Synthesise using InfiniteArrays
        ia_picture, _ = synthesis_transform(
            h_filter_params,
            v_filter_params,
            dwt_depth,
            dwt_depth_ho,
            make_variable_coeff_arrays(dwt_depth, dwt_depth_ho),
        )
        
        # Create numpy-array based coeff data for use by
        # InfiniteArray-generated functions (NB: arrays are transposed to
        # support (x, y) indexing.
        ia_coeffs_data = {
            level: {
                orient: np.array(array, dtype=int).T
                for orient, array in orients.items()
            }
            for level, orients in pseudocode_coeffs.items()
        }
        
        # Compare synthesis results
        
        for row, row_data in enumerate(pseudocode_picture):
            for col, pseudocode_value in enumerate(row_data):
                # Create and call a function to compute this value via
                # InfiniteArrays/PyExp
                expr = ia_picture[col, row]
                f = expr.make_function()
                # NB: Arrays are transposed to support (x, y) indexing
                ia_value = f(ia_coeffs_data)
                
                assert ia_value == pseudocode_value
    
    @pytest.mark.parametrize("dwt_depth,dwt_depth_ho", [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (2, 2),
    ])
    def test_analysis_intermediate_steps_as_expected(self, dwt_depth, dwt_depth_ho):
        filter_params = tables.LIFTING_FILTERS[tables.WaveletFilters.haar_with_shift]
        
        input_picture = SymbolArray(2, "p")
        
        _, intermediate_values = analysis_transform(
            filter_params,
            filter_params,
            dwt_depth,
            dwt_depth_ho,
            input_picture,
        )
        
        # 2D stages have all expected values
        for level in range(dwt_depth_ho + 1, dwt_depth + dwt_depth_ho + 1):
            names = set(n for l, n in intermediate_values if l == level)
            assert names == set([
                "Input",
                "DC",
                "DC'",
                "DC''",
                "L",
                "L'",
                "L''",
                "H",
                "H'",
                "H''",
                "LL",
                "LH",
                "HL",
                "HH",
            ])
        
        # HO stages have all expected values
        for level in range(1, dwt_depth_ho + 1):
            names = set(n for l, n in intermediate_values if l == level)
            assert names == set([
                "Input",
                "DC",
                "DC'",
                "DC''",
                "L",
                "H",
            ])
    
    @pytest.mark.parametrize("dwt_depth,dwt_depth_ho", [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (2, 2),
    ])
    def test_synthesis_intermediate_steps_as_expected(self, dwt_depth, dwt_depth_ho):
        filter_params = tables.LIFTING_FILTERS[tables.WaveletFilters.haar_with_shift]
        
        transform_coeffs = make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho)
        
        _, intermediate_values = synthesis_transform(
            filter_params,
            filter_params,
            dwt_depth,
            dwt_depth_ho,
            transform_coeffs,
        )
        
        # 2D stages have all expected values
        for level in range(dwt_depth_ho + 1, dwt_depth + dwt_depth_ho + 1):
            names = set(n for l, n in intermediate_values if l == level)
            assert names == set([
                "LL",
                "LH",
                "HL",
                "HH",
                "L''",
                "L'",
                "L",
                "H''",
                "H'",
                "H",
                "DC''",
                "DC'",
                "DC",
                "Output",
            ])
        
        # HO stages have all expected values
        for level in range(1, dwt_depth_ho + 1):
            names = set(n for l, n in intermediate_values if l == level)
            assert names == set([
                "L",
                "H",
                "DC''",
                "DC'",
                "DC",
                "Output",
            ])


class TestMakeCoeffArrays(object):
    
    @pytest.fixture
    def FakeArray(self):
        return namedtuple("FakeArray", "level,orient")
    
    def test_2d_only(self, FakeArray):
        coeff_arrays = make_coeff_arrays(2, 0, FakeArray)
        
        assert set(coeff_arrays) == set([0, 1, 2])
        
        assert set(coeff_arrays[0]) == set(["LL"])
        assert set(coeff_arrays[1]) == set(["LH", "HL", "HH"])
        assert set(coeff_arrays[2]) == set(["LH", "HL", "HH"])
        
        assert coeff_arrays[0]["LL"] == FakeArray(0, "LL")
        
        assert coeff_arrays[1]["LH"] == FakeArray(1, "LH")
        assert coeff_arrays[1]["HL"] == FakeArray(1, "HL")
        assert coeff_arrays[1]["HH"] == FakeArray(1, "HH")
        
        assert coeff_arrays[2]["LH"] == FakeArray(2, "LH")
        assert coeff_arrays[2]["HL"] == FakeArray(2, "HL")
        assert coeff_arrays[2]["HH"] == FakeArray(2, "HH")
    
    def test_2d_and_1d(self, FakeArray):
        coeff_arrays = make_coeff_arrays(1, 2, FakeArray)
        
        assert set(coeff_arrays) == set([0, 1, 2, 3])
        
        assert set(coeff_arrays[0]) == set(["L"])
        assert set(coeff_arrays[1]) == set(["H"])
        assert set(coeff_arrays[2]) == set(["H"])
        assert set(coeff_arrays[3]) == set(["LH", "HL", "HH"])
        
        assert coeff_arrays[0]["L"] == FakeArray(0, "L")
        assert coeff_arrays[1]["H"] == FakeArray(1, "H")
        assert coeff_arrays[2]["H"] == FakeArray(2, "H")
        
        assert coeff_arrays[3]["LH"] == FakeArray(3, "LH")
        assert coeff_arrays[3]["HL"] == FakeArray(3, "HL")
        assert coeff_arrays[3]["HH"] == FakeArray(3, "HH")


def test_make_symbol_coeff_arrays():
    coeff_arrays = make_symbol_coeff_arrays(2, 0, "foobar")
    
    assert isinstance(coeff_arrays[1]["HL"][2, 3], LinExp)
    assert coeff_arrays[1]["HL"][2, 3].symbol == (("foobar", 1, "HL"), 2, 3)


def test_make_variable_coeff_arrays():
    coeff_arrays = make_variable_coeff_arrays(2, 0, Argument("foobar"))
    
    assert isinstance(coeff_arrays[1]["HL"][2, 3], PyExp)
    assert coeff_arrays[1]["HL"][2, 3] == Argument("foobar")[1]["HL"][2, 3]
    
    # Check default argument works too since it is a somewhat exciting type...
    assert make_variable_coeff_arrays(2, 0)[1]["HL"][2, 3] == Argument("coeffs")[1]["HL"][2, 3]
