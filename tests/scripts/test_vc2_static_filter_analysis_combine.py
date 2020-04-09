import pytest

import json
import shlex

from vc2_bit_widths.patterns import TestPatternSpecification as TPS

from vc2_bit_widths.json_serialisations import (
    deserialise_signal_bounds,
    deserialise_test_pattern_specifications,
)

from vc2_bit_widths.scripts.vc2_static_filter_analysis import main as analysis_main
from vc2_bit_widths.scripts.vc2_static_filter_analysis_combine import main as combine_main

from vc2_bit_widths.scripts.vc2_static_filter_analysis_combine import interleave


@pytest.mark.parametrize("iterators,expected", [
    # No iterators
    (tuple(), []),
    # Single iterator
    (([1], ), [1]),
    (([1, 2, 3], ), [1, 2, 3]),
    # Multiple iterators (same length)
    (([1], [2], [3]), [1, 2, 3]),
    (([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6]),
    # Different lengths
    (([1], [2], []), [1, 2]),
    (([1], [], [2]), [1, 2]),
    (([], [1], [2]), [1, 2]),
    (([1, 4], [2, 5], [3]), [1, 2, 3, 4, 5]),
    (([1, 4], [2], [3, 5]), [1, 2, 3, 4, 5]),
    (([1, 4, 5], [2], [3]), [1, 2, 3, 4, 5]),
])
def test_interleave(iterators, expected):
    assert list(interleave(*iterators)) == expected


@pytest.fixture
def wavelet_args():
    return "-w le_gall_5_3 -d 1"


def deserialise_static_analysis(data):
    """
    Deserialise contents in place.
    """
    data["analysis_signal_bounds"] = deserialise_signal_bounds(
        data["analysis_signal_bounds"],
    )
    data["synthesis_signal_bounds"] = deserialise_signal_bounds(
        data["synthesis_signal_bounds"],
    )
    data["analysis_test_patterns"] = deserialise_test_pattern_specifications(
        TPS,
        data["analysis_test_patterns"],
    )
    data["synthesis_test_patterns"] = deserialise_test_pattern_specifications(
        TPS,
        data["synthesis_test_patterns"],
    )
    return data


@pytest.fixture
def model_answer(tmpdir, wavelet_args):
    filename = str(tmpdir.join("file_model.json"))
    
    assert analysis_main(shlex.split("{} -o {}".format(
        wavelet_args,
        filename,
    ))) == 0
    
    return deserialise_static_analysis(json.load(open(filename)))

@pytest.fixture
def batched_filenames(tmpdir, wavelet_args):
    filenames = []
    
    num_batches = 3
    for batch_num in range(num_batches):
        filename = str(tmpdir.join("file_{}.json".format(batch_num)))
        
        assert analysis_main(shlex.split("{} -o {} -B {} -b {}".format(
            wavelet_args,
            filename,
            num_batches,
            batch_num,
        ))) == 0
        
        filenames.append(filename)
    
    return filenames

@pytest.mark.parametrize("reverse_order", [False, True])
def test_combine(tmpdir, batched_filenames, model_answer, reverse_order):
    # Check that the combined file contains exactly what the all-at-once
    # generated file contains.
    
    filename = str(tmpdir.join("file.json"))
    assert combine_main(batched_filenames[::-1 if reverse_order else 1] + ["-o", filename]) == 0
    
    answer = deserialise_static_analysis(json.load(open(filename)))
    
    assert answer == model_answer

def test_combine_missing_batches(batched_filenames):
    assert combine_main(batched_filenames[:-1]) != 0

def test_combine_duplicate_batches(batched_filenames):
    assert combine_main(batched_filenames + batched_filenames[:1]) != 0

@pytest.mark.parametrize("unrelated_batch_args", [
    # Different wavelet/depth
    "-w haar_no_shift -d 1 -B 3 -b 0",
    "-w le_gall_5_3 -W haar_no_shift -d 1 -B 3 -b 0",
    "-w le_gall_5_3 -d 0 -B 3 -b 0",
    "-w le_gall_5_3 -d 1 -D 1 -B 3 -b 0",
    # Wrong number of batches
    "-w le_gall_5_3 -d 1 -B 2 -b 0",
    # Not batched
    "-w le_gall_5_3 -d 1",
])
def test_combine_unrelated_batches(tmpdir, batched_filenames, unrelated_batch_args):
    filename = str(tmpdir.join("file_unrelated.json"))
    assert analysis_main(shlex.split("{} -o {}".format(
        unrelated_batch_args,
        filename,
    ))) == 0
    
    assert combine_main(batched_filenames[1:] + [filename]) != 0
