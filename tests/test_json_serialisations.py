import pytest

import json

from collections import namedtuple

from vc2_bit_widths.linexp import LinExp

from vc2_bit_widths.signal_generation import (
    OptimisedTestSignalSpecification,
)

from vc2_bit_widths.json_serialisations import (
    serialise_intermediate_value_dictionary,
    deserialise_intermediate_value_dictionary,
    serialise_signal_bounds,
    deserialise_signal_bounds,
    serialise_concrete_signal_bounds,
    deserialise_concrete_signal_bounds,
    serialise_namedtuple,
    deserialise_namedtuple,
    serialise_test_signals,
    deserialise_test_signals,
)


def json_roundtrip(value):
    return json.loads(json.dumps(value))


def test_serialise_intermediate_value_dictionary():
    before = {
        (3, "HL", 2, 1): {"foo": "bar"},
        (6, "LH", 5, 4): {"baz": "quz"},
    }
    after = serialise_intermediate_value_dictionary(before)
    after = json_roundtrip(after)
    
    assert after == [
        {
            "level": 3,
            "array_name": "HL",
            "phase": [2, 1],
            "foo": "bar",
        },
        {
            "level": 6,
            "array_name": "LH",
            "phase": [5, 4],
            "baz": "quz",
        },
    ]
    
    assert deserialise_intermediate_value_dictionary(after) == before


def test_serialise_signal_bounds():
    before = {
        (1, "LH", 2, 3): (
            LinExp("foo")/2 + 1,
            LinExp("bar")/4 + 2,
        ),
    }
    after = serialise_signal_bounds(before)
    after = json_roundtrip(after)
    
    assert after == [
        {
            "level": 1,
            "array_name": "LH",
            "phase": [2, 3],
            "lower_bound": [
                {"symbol": "foo", "numer": "1", "denom": "2"},
                {"symbol": None, "numer": "1", "denom": "1"},
            ],
            "upper_bound": [
                {"symbol": "bar", "numer": "1", "denom": "4"},
                {"symbol": None, "numer": "2", "denom": "1"},
            ],
        },
    ]
    
    assert deserialise_signal_bounds(after) == before


def test_serialise_concrete_signal_bounds():
    before = {
        (1, "LH", 2, 3): (-512, 511),
    }
    after = serialise_concrete_signal_bounds(before)
    after = json_roundtrip(after)
    
    assert after == [
        {
            "level": 1,
            "array_name": "LH",
            "phase": [2, 3],
            "lower_bound": -512,
            "upper_bound": 511,
        },
    ]
    
    assert deserialise_concrete_signal_bounds(after) == before


def test_serialise_namedtuple():
    Point = namedtuple('Point', ['x', 'y'])
    
    before = Point(x=10, y=20)
    after = serialise_namedtuple(Point, before)
    after = json_roundtrip(after)
    
    assert after == {
        "x": 10,
        "y": 20,
    }
    
    assert deserialise_namedtuple(Point, after) == before
    assert type(deserialise_namedtuple(Point, after)) == type(before)


def test_serialise_test_signals():
    before = {
        (1, "LH", 2, 3): OptimisedTestSignalSpecification(
            target=(4, 5),
            picture={(10, 20): 1},
            picture_translation_multiple=(6, 7),
            target_translation_multiple=(8, 9),
            quantisation_index=30,
            decoded_value=40,
            num_search_iterations=50,
        ),
    }
    after = serialise_test_signals(OptimisedTestSignalSpecification, before)
    after = json_roundtrip(after)
    
    assert after == [
        {
            "level": 1,
            "array_name": "LH",
            "phase": [2, 3],
            "target": [4, 5],
            "picture": [[10, 20, 1]],
            "picture_translation_multiple": [6, 7],
            "target_translation_multiple": [8, 9],
            "quantisation_index": 30,
            "decoded_value": 40,
            "num_search_iterations": 50,
        },
    ]
    
    assert deserialise_test_signals(OptimisedTestSignalSpecification, after) == before
