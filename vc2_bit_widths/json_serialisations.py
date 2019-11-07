"""
Data file formatting utilities
==============================

This module contains utilities for serialising and deserialising bit width
analysis and test signal data as JSON. In particular it concentrates on the
data types returned by the functions in :py:mod:`vc2_bit_widths.helpers`
module.

Specifically:

:py:func:`vc2_bit_widths.helpers.static_filter_analysis`'s output can be
(de)serialised by:

.. autofunction:: serialise_signal_bounds

.. autofunction:: deserialise_signal_bounds

.. autofunction:: serialise_test_signals

.. autofunction:: deserialise_test_signals

:py:func:`vc2_bit_widths.helpers.evaluate_filter_bounds`'s output can be
(de)serialised by :py:func:`serialise_signal_bounds` and
:py:func:`deserialise_signal_bounds` too.

:py:func:`vc2_bit_widths.helpers.optimise_synthesis_test_signals`'s output can
be (de)serialised by :py:func:`serialise_test_signals` and
:py:func:`deserialise_test_signals` too.

"""

import numpy as np

from base64 import b64encode, b64decode

from collections import OrderedDict

from vc2_bit_widths.linexp import LinExp


def serialise_intermediate_value_dictionary(dictionary):
    """
    Given a dictionary whose keys are intermediate value index tuples ``(level,
    array_name, x, y)`` and whose values are dictionaries, return a
    JSON-serialisable equivalent form.
    
    For example::
    
        >>> before = {
        ...     (3, "HL", 2, 1): {"foo": "bar", "baz": "quz"},
        ...     ...
        ... }
        >>> serialise_intermediate_value_dictionary(before)
        [
            {
                "level": 3,
                "array_name": "HL",
                "phase": [2, 1],
                "foo": "bar",
                "baz": "qux",
            },
            ...
        ]
    """
    out = []
    for (level, array_name, x, y), value in dictionary.items():
        d = value.copy()
        d["level"] = level
        d["array_name"] = array_name
        d["phase"] = [x, y]
        out.append(d)
    return out


def deserialise_intermediate_value_dictionary(lst):
    """
    Inverse of :py:func:`serialise_intermediate_value_dictionary`.
    
    NB: Will mutate the dictionaries in the passed-in list.
    """
    out = OrderedDict()
    for d in lst:
        level = d.pop("level")
        array_name = d.pop("array_name")
        x, y = d.pop("phase")
        out[(level, array_name, x, y)] = d
    return out


def serialise_signal_bounds(signal_bounds):
    """
    Convert a dictionary of analysis or synthesis signal bounds expressions
    into a JSON-serialisable form.
    
    For example::
        >>> before = {
        ...     (1, "LH", 2, 3): (
        ...         LinExp("foo")/2 + 1,
        ...         LinExp("bar")/4 + 2,
        ...     ),
        ...     ...
        ... }
        >>> serialise_signal_bounds(before)
        [
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
            ...
        ]
    """
    return serialise_intermediate_value_dictionary({
        key: {
            "lower_bound": LinExp(lower_bound).to_json(),
            "upper_bound": LinExp(upper_bound).to_json(),
        }
        for key, (lower_bound, upper_bound) in signal_bounds.items()
    })


def deserialise_signal_bounds(signal_bounds):
    """
    Inverse of :py:func:`serialise_signal_bounds`.
    """
    return {
        key: (
            LinExp.from_json(d["lower_bound"]),
            LinExp.from_json(d["upper_bound"]),
        )
        for key, d in deserialise_intermediate_value_dictionary(signal_bounds).items()
    }


def serialise_concrete_signal_bounds(signal_bounds):
    """
    Convert a dictionary of concrete analysis or synthesis signal bounds
    expressions into a JSON-serialisable form.
    
    For example::
        >>> before = {
        ...     (1, "LH", 2, 3): (-512, 511),
        ...     ...
        ... }
        >>> serialise_signal_bounds(before)
        [
            {
                "level": 1,
                "array_name": "LH",
                "phase": [2, 3],
                "lower_bound": -512,
                "upper_bound": 511,
            },
            ...
        ]
    """
    return serialise_intermediate_value_dictionary({
        key: {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
        for key, (lower_bound, upper_bound) in signal_bounds.items()
    })


def deserialise_concrete_signal_bounds(signal_bounds):
    """
    Inverse of :py:func:`serialise_concrete_signal_bounds`.
    """
    return {
        key: (d["lower_bound"], d["upper_bound"])
        for key, d in deserialise_intermediate_value_dictionary(signal_bounds).items()
    }


def serialise_namedtuple(namedtuple_type, value):
    """
    Serialise a tuple into a dictionary of named values.
    
    For example::
    
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> before = Point(x=10, y=20)
        >>> serialise_namedtuple(Point, before)
        {
            "x": 10,
            "y": 20,
        }
    
    Parameters
    ==========
    namedtuple_type : :py:class:`~collections.namedtuple`
        The namedtuple class of the value to be serialised.
    value : tuple
    """
    return dict(zip(namedtuple_type._fields, value))


def deserialise_namedtuple(namedtuple_type, dictionary):
    """
    Inverse of :py:func:`serialise_namedtuple`.
    """
    return namedtuple_type(*(
        dictionary[n] for n in namedtuple_type._fields
    ))


def serialise_picture(picture):
    """
    Convert a test picture (a dictionary {(x, y): polarity, ...}) into a
    compact JSON-serialisable form.
    
    For example::
        >>> before = {
        ...     (4, 10): +1, (5, 10): -1, (6, 10): +1, (7, 10): -1,
        ...     (4, 11): -1, (5, 11): +1, (6, 11): -1, (7, 11): +1,
        ...     (4, 12): +1, (5, 12): -1, (6, 12): +1, (7, 12): -1,
        ...     (4, 13): -1, (5, 13): +1, (6, 13): -1, (7, 13): +1,
        ... }
        >>> serialise_picture(before)
        {
            'dx': 4,
            'dy': 10,
            'width': 4,
            'height': 4,
            'positive': 'paU=',
            'mask': '//8=',
        }
    
    The serialised format is based on a Base64 (RFC 3548) encoded bit-packed
    positive/mask representation of the picture.
    
    In a positive/mask representation, the input picture is represented as two
    binary arrays: 'positive' and 'mask'. When an entry in the mask is 1, a 1
    in the corresponding positive array means '+1' and a 0 means '-1'. When the
    mask entry is 0, the corresponding entry in the positive array is ignored
    and a value of '0' is assumed.
    
    The positive and mask arrays are defined as having their bottom left corner
    at (dx, dy) and having the width and height defined. All values outside of
    this region are zero.
    
    The positive and mask arrays are serialised in raster-scan order into a bit
    array. This bit array is packed into bytes with the first bit in each byte
    being the most significant.
    
    The byte-packed bit arrays are finally Base64 (RFC 3548) encoded into the
    'positive' and 'mask' fields.
    """
    xs, ys = map(np.array, zip(*picture))
    
    # Bottom-left corner of the sparse matrix
    dx = np.min(xs)
    dy = np.min(ys)
    
    # Active area of the sparse matrix
    width = np.max(xs) - dx + 1
    height = np.max(ys) - dy + 1
    
    # Convert to numpy array with bottom left at (dx, dy)
    polarities = np.zeros((height, width))
    polarities[(ys-dy, xs-dx)] = np.array(list(picture.values()))
    
    # Convert into positive + mask representation
    positive = polarities > 0
    mask = polarities != 0
    
    packed_positive = np.packbits(positive, bitorder="big").tobytes()
    packed_mask = np.packbits(mask, bitorder="big").tobytes()
    
    out = {
        "dx": int(dx),
        "dy": int(dy),
        "width": int(width),
        "height": int(height),
        "positive": b64encode(packed_positive).decode("ascii"),
        "mask": b64encode(packed_mask).decode("ascii"),
    }
    
    return out


def deserialise_picture(dictionary):
    """
    Inverse of :py:func:`serialise_namedtuple`.
    """
    dx = dictionary["dx"]
    dy = dictionary["dy"]
    width = dictionary["width"]
    height = dictionary["height"]
    
    packed_positive = np.frombuffer(
        b64decode(dictionary["positive"]),
        dtype=np.uint8,
    )
    packed_mask = np.frombuffer(
        b64decode(dictionary["mask"]),
        dtype=np.uint8,
    )
    
    count = width * height
    
    positive = np.unpackbits(
        packed_positive,
        count=count,
        bitorder="big",
    ).reshape((height, width))
    
    mask = np.unpackbits(
        packed_mask,
        count=count,
        bitorder="big",
    ).reshape((height, width))
    
    return {
        (x + dx, y + dy): +1 if positive[y, x] else -1
        for y, x in np.argwhere(mask)
    }


def serialise_test_signals(spec_namedtuple_type, test_signals):
    """
    Convert a dictionary of analysis or synthesis test signals into a
    JSON-serialisable form.
    
    For example::
        >>> before = {
        ...     (1, "LH", 2, 3): TestSignalSpecification(
        ...         target=(4, 5),
        ...         picture={(x, y): polarity, ...},
        ...         picture_translation_multiple=(6, 7),
        ...         target_translation_multiple=(8, 9),
        ...     ),
        ...     ...
        ... }
        >>> serialise_test_signals(TestSignalSpecification, before)
        [
            {
                "level": 1,
                "array_name": "LH",
                "phase": [2, 3],
                "target": [4, 5],
                "picture": {
                    "dx": ...,
                    "dy": ...,
                    "width": ...,
                    "height": ...,
                    "positive": ...,
                    "mask": ...,
                },
                "picture_translation_multiple": [6, 7],
                "target_translation_multiple": [8, 9],
            },
            ...
        ]
    
    See :py:func:`serialise_picture` for the serialisation used for the picture
    data.
    
    Parameters
    ==========
    spec_namedtuple_type : :py:class:`~collections.namedtuple` class
        The namedtuple used to hold the test signal specification.
    test_signals : {(level, array_name, x, y): (...), ...}
    """
    out = serialise_intermediate_value_dictionary({
        key: serialise_namedtuple(spec_namedtuple_type, value)
        for key, value in test_signals.items()
    })
    
    for d in out:
        d["picture"] = serialise_picture(d["picture"])
    
    return out


def deserialise_test_signals(spec_namedtuple_type, test_signals):
    """
    Inverse of :py:func:`serialise_test_signals`.
    """
    for d in test_signals:
        d["picture"] = deserialise_picture(d["picture"])
        d["target"] = tuple(d["target"])
        d["picture_translation_multiple"] = tuple(d["picture_translation_multiple"])
        d["target_translation_multiple"] = tuple(d["target_translation_multiple"])
    
    return {
        key: deserialise_namedtuple(spec_namedtuple_type, value)
        for key, value in deserialise_intermediate_value_dictionary(test_signals).items()
    }


def serialise_quantisation_matrix(quantisation_matrix):
    """
    Convert a quantisation matrix into JSON-serialisable form.
    
    For example::
        >>> before = {0: {"L": 2}, 1: {"H": 0}}
        >>> serialise_quantisation_matrix(before)
        {"0": {"L": 2}, "1": {"H": 0}}
    """
    return {
        str(level): orients
        for level, orients in quantisation_matrix.items()
    }


def deserialise_quantisation_matrix(quantisation_matrix):
    """
    Inverse of :py:func:`serialise_quantisation_matrix`.
    """
    return {
        int(level): orients
        for level, orients in quantisation_matrix.items()
    }
