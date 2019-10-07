import pytest

from fractions import Fraction

import vc2_data_tables as tables

from vc2_conformance.decoder.transform_data_syntax import (
    quant_factor,
    inverse_quant,
)

from vc2_bit_widths.linexp import LinExp, AAError

from vc2_bit_widths.infinite_arrays import (
    SymbolArray,
)

from vc2_bit_widths.vc2_filters import (
    analysis_transform,
    synthesis_transform,
    make_coeff_arrays,
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
        
        transform_coeffs = make_coeff_arrays(dwt_depth, dwt_depth_ho)
        
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
    
    def test_2d_only(self):
        coeff_arrays = make_coeff_arrays(2, 0, "foo")
        
        assert set(coeff_arrays) == set([0, 1, 2])
        
        assert set(coeff_arrays[0]) == set(["LL"])
        assert set(coeff_arrays[1]) == set(["LH", "HL", "HH"])
        assert set(coeff_arrays[2]) == set(["LH", "HL", "HH"])
        
        assert coeff_arrays[0]["LL"][10, 20] == LinExp((("foo", 0, "LL"), 10, 20))
        
        assert coeff_arrays[1]["LH"][10, 20] == LinExp((("foo", 1, "LH"), 10, 20))
        assert coeff_arrays[1]["HL"][10, 20] == LinExp((("foo", 1, "HL"), 10, 20))
        assert coeff_arrays[1]["HH"][10, 20] == LinExp((("foo", 1, "HH"), 10, 20))
        
        assert coeff_arrays[2]["LH"][10, 20] == LinExp((("foo", 2, "LH"), 10, 20))
        assert coeff_arrays[2]["HL"][10, 20] == LinExp((("foo", 2, "HL"), 10, 20))
        assert coeff_arrays[2]["HH"][10, 20] == LinExp((("foo", 2, "HH"), 10, 20))
    
    def test_2d_and_1d(self):
        coeff_arrays = make_coeff_arrays(1, 2, "foo")
        
        assert set(coeff_arrays) == set([0, 1, 2, 3])
        
        assert set(coeff_arrays[0]) == set(["L"])
        assert set(coeff_arrays[1]) == set(["H"])
        assert set(coeff_arrays[2]) == set(["H"])
        assert set(coeff_arrays[3]) == set(["LH", "HL", "HH"])
        
        assert coeff_arrays[0]["L"][10, 20] == LinExp((("foo", 0, "L"), 10, 20))
        assert coeff_arrays[1]["H"][10, 20] == LinExp((("foo", 1, "H"), 10, 20))
        assert coeff_arrays[2]["H"][10, 20] == LinExp((("foo", 2, "H"), 10, 20))
        
        assert coeff_arrays[3]["LH"][10, 20] == LinExp((("foo", 3, "LH"), 10, 20))
        assert coeff_arrays[3]["HL"][10, 20] == LinExp((("foo", 3, "HL"), 10, 20))
        assert coeff_arrays[3]["HH"][10, 20] == LinExp((("foo", 3, "HH"), 10, 20))

