import pytest

import json

import shlex

from vc2_data_tables import WaveletFilters

from vc2_bit_widths.patterns import (
    OptimisedTestPatternSpecification,
)

from vc2_bit_widths.json_serialisations import (
    deserialise_test_pattern_specifications,
    deserialise_quantisation_matrix,
)

from vc2_bit_widths.scripts.vc2_static_filter_analysis import main as sfa
from vc2_bit_widths.scripts.vc2_optimise_synthesis_test_patterns import main as osfts


def test_sanity(tmpdir, capsys):
    # Just a simple check to see that optimisation achieves *something*
    
    static_file = str(tmpdir.join("static_file.json"))
    unoptimised_file = str(tmpdir.join("unoptimised_file.json"))
    optimised_file = str(tmpdir.join("optimised_file.json"))
    
    # vc2-static-filter-analysis
    assert sfa(shlex.split("-w le_gall_5_3 -D 1 -o") + [static_file]) == 0
    
    # Run with no iterations of random search
    # vc2-optimise-synthesis-filter-test-patterns
    assert osfts(
        [static_file]
        + shlex.split("-b 10 -i 0 -o")
        + [unoptimised_file]
    ) == 0
    
    # Run with 100 iterations of random search
    # vc2-optimise-synthesis-filter-test-patterns
    assert osfts(
        [static_file]
        + shlex.split("-b 10 -N 1 -a 0.05 -r 0.0 -i 100 -I 0 -o")
        + [optimised_file]
    ) == 0
    
    with open(optimised_file, "r") as f:
        optimised = json.load(f)
    with open(unoptimised_file, "r") as f:
        unoptimised = json.load(f)
    
    # Check all wavelet parameters are passed through as expected
    assert optimised["wavelet_index"] == WaveletFilters.le_gall_5_3
    assert optimised["wavelet_index_ho"] == WaveletFilters.le_gall_5_3
    assert optimised["dwt_depth"] == 0
    assert optimised["dwt_depth_ho"] == 1
    assert optimised["picture_bit_width"] == 10
    assert deserialise_quantisation_matrix(optimised["quantisation_matrix"]) == {
        0: {"L": 2},
        1: {"H": 0}
    }
    
    assert unoptimised["wavelet_index"] == WaveletFilters.le_gall_5_3
    assert unoptimised["wavelet_index_ho"] == WaveletFilters.le_gall_5_3
    assert unoptimised["dwt_depth"] == 0
    assert unoptimised["dwt_depth_ho"] == 1
    assert unoptimised["picture_bit_width"] == 10
    assert deserialise_quantisation_matrix(unoptimised["quantisation_matrix"]) == {
        0: {"L": 2},
        1: {"H": 0}
    }
    
    # Check optimised version does achieve some benefit over non-optimised
    # version
    optimised_test_patterns = deserialise_test_pattern_specifications(
        OptimisedTestPatternSpecification,
        optimised["optimised_synthesis_test_patterns"],
    )
    unoptimised_test_patterns = deserialise_test_pattern_specifications(
        OptimisedTestPatternSpecification,
        unoptimised["optimised_synthesis_test_patterns"],
    )
    
    assert set(optimised_test_patterns) == set(unoptimised_test_patterns)
    assert any(
        abs(optimised_test_patterns[key].decoded_value) >
        abs(unoptimised_test_patterns[key].decoded_value)
        for key in optimised_test_patterns
    )
