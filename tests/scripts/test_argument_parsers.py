import pytest

from vc2_data_tables import WaveletFilters, QUANTISATION_MATRICES

from vc2_bit_widths.scripts.argument_parsers import (
    parse_quantisation_matrix_argument,
)


class TestParseQuantisationMatrixArgument(object):
    
    def test_use_default_matrix(self):
        assert parse_quantisation_matrix_argument(
            None,
            WaveletFilters.le_gall_5_3,
            WaveletFilters.le_gall_5_3,
            4,
            0,
        ) == QUANTISATION_MATRICES[(
            WaveletFilters.le_gall_5_3,
            WaveletFilters.le_gall_5_3,
            4,
            0,
        )]
    
    def test_default_matrix_not_available_and_non_specified(self):
        with pytest.raises(SystemExit):
            parse_quantisation_matrix_argument(
                None,
                WaveletFilters.le_gall_5_3,
                WaveletFilters.le_gall_5_3,
                100,
                0,
            )
    
    def test_custom_matrix_supplied(self):
        assert parse_quantisation_matrix_argument(
            "0 L 10 1 H 20".split(),
            WaveletFilters.le_gall_5_3,
            WaveletFilters.le_gall_5_3,
            0,
            1,
        ) == {
            0: {"L": 10},
            1: {"H": 20},
        }
    
    @pytest.mark.parametrize("argument", [
        # Wrong length
        "1",
        "1 L",
        "0 L 10 1",
        "0 L 10 1 H",
        # Non-integer values where there should be
        "x L 10 1 H 20",
        "0 L xx 1 H 20",
    ])
    def test_bad_custom_matrix(self, argument):
        with pytest.raises(SystemExit):
            parse_quantisation_matrix_argument(
                argument.split(),
                WaveletFilters.le_gall_5_3,
                WaveletFilters.le_gall_5_3,
                0,
                1,
            )
