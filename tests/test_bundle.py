import pytest

import json

from vc2_data_tables import (
    WaveletFilters,
)

from vc2_bit_widths.bundle import (
    bundle_create_from_serialised_dicts,
    bundle_index,
    bundle_get_static_filter_analysis,
    bundle_get_optimised_synthesis_test_patterns,
)

from vc2_bit_widths.helpers import (
    static_filter_analysis,
)

from vc2_bit_widths.scripts import (
    vc2_static_filter_analysis,
    vc2_optimise_synthesis_test_patterns,
)


def test_empty(tmpdir):
    filename = str(tmpdir.join("bundle.zip"))
    bundle_create_from_serialised_dicts(filename)
    
    assert bundle_index(filename) == {
        "static_filter_analyses": [],
        "optimised_synthesis_test_patterns": [],
    }
    
    with pytest.raises(KeyError):
        bundle_get_static_filter_analysis(
            filename,
            WaveletFilters.haar_no_shift,
            WaveletFilters.haar_no_shift,
            1,
            2,
        )
    
    with pytest.raises(KeyError):
        bundle_get_optimised_synthesis_test_patterns(
            filename,
            WaveletFilters.haar_no_shift,
            WaveletFilters.haar_no_shift,
            0,
            0,
            {},
            10,
        )


def test_bundle_get_static_filter_analysis(tmpdir):
    bundle_filename = str(tmpdir.join("bundle.zip"))
    analysis_1_filename = str(tmpdir.join("analysis1.json"))
    analysis_2_filename = str(tmpdir.join("analysis2.json"))
    
    vc2_static_filter_analysis.main([
        "-w", "haar_no_shift",
        "-W", "haar_with_shift",
        "-d", "1",
        "-D", "2",
        "-o", analysis_1_filename,
    ])
    analysis_1_json = json.load(open(analysis_1_filename, "rb"))
    
    vc2_static_filter_analysis.main([
        "-w", "haar_no_shift",
        "-d", "1",
        "-o", analysis_2_filename,
    ])
    analysis_2_json = json.load(open(analysis_2_filename, "rb"))
    
    bundle_create_from_serialised_dicts(
        bundle_filename,
        serialised_static_filter_analyses=[
            analysis_1_json,
            analysis_2_json,
        ],
    )
    
    out = bundle_get_static_filter_analysis(
        bundle_filename,
        WaveletFilters.haar_no_shift,
        WaveletFilters.haar_with_shift,
        1,
        2,
    )
    expected = static_filter_analysis(
        WaveletFilters.haar_no_shift,
        WaveletFilters.haar_with_shift,
        1,
        2,
    )
    assert out == expected
    
    out = bundle_get_static_filter_analysis(
        bundle_filename,
        WaveletFilters.haar_no_shift,
        WaveletFilters.haar_no_shift,
        1,
        0,
    )
    expected = static_filter_analysis(
        WaveletFilters.haar_no_shift,
        WaveletFilters.haar_no_shift,
        1,
        0,
    )
    assert out == expected
    
    with pytest.raises(KeyError):
        bundle_get_static_filter_analysis(
            bundle_filename,
            WaveletFilters.haar_no_shift,
            WaveletFilters.haar_no_shift,
            2,
            0,
        )


def test_bundle_get_optimised_synthesis_test_patterns(tmpdir):
    bundle_filename = str(tmpdir.join("bundle.zip"))
    analysis_filename = str(tmpdir.join("analysis.json"))
    optimised_filename = str(tmpdir.join("optimised.json"))
    
    vc2_static_filter_analysis.main([
        "-w", "haar_no_shift",
        "-W", "haar_with_shift",
        "-d", "0",
        "-D", "1",
        "-o", analysis_filename,
    ])
    analysis_json = json.load(open(analysis_filename, "rb"))
    
    vc2_optimise_synthesis_test_patterns.main([
        analysis_filename,
        "-b", "10",
        "-q", "0", "L", "0", "1", "H", "3",
        "-o", optimised_filename,
        # Effectively disable the search for speed reasons
        "--base-iterations", "0",
        "--added-corruption-rate", "0",
    ])
    optimised_json = json.load(open(optimised_filename, "rb"))
    
    bundle_create_from_serialised_dicts(
        bundle_filename,
        serialised_optimised_synthesis_test_patterns=[
            optimised_json,
        ],
    )
    
    # Check it looks like a (deserialised) optimised pattern
    out = bundle_get_optimised_synthesis_test_patterns(
        bundle_filename,
        WaveletFilters.haar_no_shift,
        WaveletFilters.haar_with_shift,
        0,
        1,
        {0: {"L": 0}, 1: {"H": 3}},
        10,
    )
    assert out[(1, "L", 0, 0)].num_search_iterations == 0
    
    with pytest.raises(KeyError):
        bundle_get_optimised_synthesis_test_patterns(
            bundle_filename,
            WaveletFilters.haar_no_shift,
            WaveletFilters.haar_with_shift,
            0,
            0,
            {0: {"LL": 0}},
            8,
        )
