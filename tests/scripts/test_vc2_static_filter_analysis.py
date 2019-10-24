import pytest

import json

import shlex

from vc2_data_tables import WaveletFilters

from vc2_bit_widths.signal_generation import TestSignalSpecification as TSS

from vc2_bit_widths.json_serialisations import (
    deserialise_signal_bounds,
    deserialise_test_signals,
)

from vc2_bit_widths.scripts.vc2_static_filter_analysis import main


def test_sanity(tmpdir):
    # Just a simple sanity check that the correct options are passed through
    # and processed as expected
    
    filename = str(tmpdir.join("file.json"))
    
    assert main(shlex.split("-w 4 -W daubechies_9_7 -d 1 -D 0 -o {}".format(
        filename,
    ))) == 0
    
    with open(filename, "r") as f:
        d = json.load(f)
    
    assert d["wavelet_index"] == WaveletFilters.haar_with_shift
    assert d["wavelet_index_ho"] == WaveletFilters.daubechies_9_7
    assert d["dwt_depth"] == 1
    assert d["dwt_depth_ho"] == 0
    
    analysis_signal_bounds = deserialise_signal_bounds(d["analysis_signal_bounds"])
    synthesis_signal_bounds = deserialise_signal_bounds(d["synthesis_signal_bounds"])
    
    analysis_test_signals = deserialise_test_signals(TSS, d["analysis_test_signals"])
    synthesis_test_signals = deserialise_test_signals(TSS, d["synthesis_test_signals"])
    
    # Check self consistent
    assert set(analysis_signal_bounds) == set(analysis_test_signals)
    assert set(synthesis_signal_bounds) == set(synthesis_test_signals)
    
    # Check first level is a 2D transform
    assert (1, "LL", 0, 0) in analysis_signal_bounds
    assert (1, "LL", 0, 0) in synthesis_signal_bounds
    
    # Check second level doesn't exist (i.e. we did a 1-level transform)
    assert (2, "Input", 0, 0) not in analysis_signal_bounds
    assert (2, "Output", 0, 0) not in synthesis_signal_bounds
    
    # Check horizontally we're doing the Daubechies 9/7 transform (which has 4
    # lifting stages) and vertically, the haar (which has 2 lifting stages)
    assert (1, "DC'", 0, 0) in analysis_signal_bounds  # Horizontal Stage 1
    assert (1, "DC''", 0, 0) in analysis_signal_bounds  # Horizontal Stage 2
    assert (1, "DC'''", 0, 0) in analysis_signal_bounds  # Horizontal Stage 3
    assert (1, "DC''''", 0, 0) in analysis_signal_bounds  # Horizontal Stage 4
    assert (1, "L'", 0, 0) in analysis_signal_bounds  # Vertical Stage 1
    assert (1, "L''", 0, 0) in analysis_signal_bounds  # Vertical Stage 2
    assert (1, "L'''", 0, 0) not in analysis_signal_bounds  # No vertical Stage 3!
    
    assert (1, "DC'", 0, 0) in synthesis_signal_bounds  # Horizontal Stage 1
    assert (1, "DC''", 0, 0) in synthesis_signal_bounds  # Horizontal Stage 2
    assert (1, "DC'''", 0, 0) in synthesis_signal_bounds  # Horizontal Stage 3
    assert (1, "DC''''", 0, 0) in synthesis_signal_bounds  # Horizontal Stage 4
    assert (1, "L'", 0, 0) in synthesis_signal_bounds  # Vertical Stage 1
    assert (1, "L''", 0, 0) in synthesis_signal_bounds  # Vertical Stage 2
    assert (1, "L'''", 0, 0) not in synthesis_signal_bounds  # No vertical Stage 3!