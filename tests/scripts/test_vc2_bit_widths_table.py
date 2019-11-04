import pytest

import csv

import shlex

from collections import OrderedDict

from vc2_data_tables import WaveletFilters, LIFTING_FILTERS

from vc2_bit_widths.infinite_arrays import (
    SymbolArray,
)

from vc2_bit_widths.vc2_filters import (
    make_symbol_coeff_arrays,
    analysis_transform,
    synthesis_transform,
)

from vc2_bit_widths.scripts.vc2_static_filter_analysis import main as sfa
from vc2_bit_widths.scripts.vc2_bit_widths_table import main as bwt

from vc2_bit_widths.scripts.vc2_bit_widths_table import (
    dict_aggregate,
    combine_bounds,
)


@pytest.mark.parametrize("dict_type", [
    dict,
    OrderedDict,
])
def test_dict_aggregate(dict_type):
    d = dict_type([
        ((1, 10), 10),
        ((2, 20), 20),
        ((2, 200), 200),
        ((3, 30), 30),
        ((3, 300), 300),
        ((3, 3000), 3000),
    ])
    
    def key_map(t):
        return t[0]
    
    def value_reduce(a, b):
        return a + b
    
    out = dict_aggregate(d, key_map, value_reduce)
    
    assert isinstance(out, dict_type)
    
    assert out == {
        1: 10,
        2: 220,
        3: 3330,
    }


@pytest.mark.parametrize("a,b,exp", [
    # All same
    ((10, 100), (10, 100), (10, 100)),
    # Left wider
    ((5, 100), (10, 100), (5, 100)),
    ((10, 110), (10, 100), (10, 110)),
    ((5, 110), (10, 100), (5, 110)),
    # Right wider
    ((10, 100), (5, 100), (5, 100)),
    ((10, 100), (10, 110), (10, 110)),
    ((10, 100), (5, 110), (5, 110)),
    # Overlapping
    ((10, 100), (50, 1000), (10, 1000)),
    # Non-overlapping
    ((10, 100), (1000, 10000), (10, 10000)),
])
def combine_bounds(a, b, exp):
    assert combine_bounds(a, b) == exp


def test_bit_widths(tmpdir, capsys):
    # Just a simple sanity check that the command accepts the bit widths
    # specified
    
    f = str(tmpdir.join("file.json"))
    
    # vc2-static-filter-analysis
    assert sfa(shlex.split("-w haar_with_shift -d 1 -o {}".format(f))) == 0
    
    # vc2-bit-widths-table
    assert bwt(shlex.split("{} -b 10".format(f))) == 0
    
    csv_rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    
    # Check all columns are present as expected
    assert csv_rows[0] == (
        ["type", "level", "array_name"] +
        ["lower_bound", "test_signal_min", "test_signal_max", "upper_bound", "bits"]
    )
    
    # Check correct bit widths were read from arguments
    csv_rows[1] == [
        "analysis", 1, "Input",
        -512, -512, 511, 511, 10,
    ]


@pytest.mark.parametrize("arg,exp_phases", [
    ("", False),
    ("--show-all-filter-phases", True),
])
def test_aggregation_flag(tmpdir, capsys, arg, exp_phases):
    # Check that aggregation of filter phases works
    
    f = str(tmpdir.join("file.json"))
    
    # vc2-static-filter-analysis
    assert sfa(shlex.split("-w haar_with_shift -d 1 -o {}".format(f))) == 0
    
    # vc2-bit-widths-table
    assert bwt(shlex.split("{} -b 10 {}".format(f, arg))) == 0
    
    csv_rows = list(csv.reader(capsys.readouterr().out.splitlines()))
    
    columns = csv_rows[0][:-5]
    
    # Check all phase columns are present as expected
    if exp_phases:
        assert columns == ["type", "level", "array_name", "x", "y"]
    else:
        assert columns == ["type", "level", "array_name"]
    
    # Check the rows are as expected
    row_headers = [
        tuple(row[:-5])
        for row in csv_rows[1:]
    ]
    
    # ...by comparing with the intermediate arrays expected for this filter...
    h_filter_params = LIFTING_FILTERS[WaveletFilters.haar_with_shift]
    v_filter_params = LIFTING_FILTERS[WaveletFilters.haar_with_shift]
    dwt_depth = 1
    dwt_depth_ho = 0
    
    _, analysis_intermediate_arrays = analysis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        SymbolArray(2),
    )
    _, synthesis_intermediate_arrays = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho),
    )
    
    if exp_phases:
        assert row_headers == [
            (type_name, str(level), array_name, str(x), str(y))
            for type_name, intermediate_arrays in [
                ("analysis", analysis_intermediate_arrays),
                ("synthesis", synthesis_intermediate_arrays),
            ]
            for (level, array_name), array in intermediate_arrays.items()
            for x in range(array.period[0])
            for y in range(array.period[1])
        ]
    else:
        assert row_headers == [
            (type_name, str(level), array_name)
            for type_name, intermediate_arrays in [
                ("analysis", analysis_intermediate_arrays),
                ("synthesis", synthesis_intermediate_arrays),
            ]
            for level, array_name in intermediate_arrays
        ]


def test_aggregation_correct(tmpdir, capsys):
    # Check on manually computed example
    
    f = str(tmpdir.join("file.json"))
    
    # vc2-static-filter-analysis
    assert sfa(shlex.split("-w haar_with_shift -d 1 -o {}".format(f))) == 0
    
    # vc2-bit-widths-table
    assert bwt(shlex.split("{} -b 10".format(f))) == 0
    csv_aggregated = list(csv.reader(capsys.readouterr().out.splitlines()))
    
    # vc2-bit-widths-table
    assert bwt(shlex.split("{} -b 10 -p".format(f))) == 0
    csv_raw = list(csv.reader(capsys.readouterr().out.splitlines()))
    
    aggregated_values = {
        tuple(row[:-5]): tuple(row[-5:])
        for row in csv_aggregated
    }
    
    raw_values = {
        tuple(row[:-5]): tuple(row[-5:])
        for row in csv_raw
    }
    
    # Hand checked example...
    assert raw_values[("analysis", "1", "L''", "0", "0")] == ("-1025", "-1024", "1022", "1024", "11-12")
    assert raw_values[("analysis", "1", "L''", "0", "1")] == ("-2048", "-2046", "2046", "2049", "12-13")
    
    assert aggregated_values[("analysis", "1", "L''")] == ("-2048", "-2046", "2046", "2049", "12-13")
