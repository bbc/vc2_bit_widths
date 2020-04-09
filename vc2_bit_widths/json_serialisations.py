"""
:py:mod:`vc2_bit_widths.json_serialisations`: JSON Data File Serialisation/Deserialisation
==========================================================================================

The :py:mod:`vc2_bit_widths.json_serialisations` module provides functions for
serialising and deserialising parts of the JSON files produced by the various
command-line interfaces to :py:mod:`vc2_bit_widths` (see :ref:`cli`).

.. note::

    These functions do not serialise/deserialise JSON directly, instead they
    produce/accept standard Python data structures compatible with the Python
    :py:mod:`json` module.

Signal bounds
-------------

.. autofunction:: serialise_signal_bounds

.. autofunction:: deserialise_signal_bounds

.. autofunction:: serialise_linexp

.. autofunction:: deserialise_linexp


Test patterns
-------------

.. autofunction:: serialise_test_pattern_specifications

.. autofunction:: deserialise_test_pattern_specifications

.. autofunction:: serialise_test_pattern

.. autofunction:: deserialise_test_pattern

Quantisation matrices
---------------------

.. autofunction:: serialise_quantisation_matrix

.. autofunction:: deserialise_quantisation_matrix

"""

import numpy as np

from base64 import b64encode, b64decode

from collections import OrderedDict

from vc2_bit_widths.linexp import LinExp

from vc2_bit_widths.fast_fractions import Fraction

from vc2_bit_widths.patterns import TestPattern


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


def serialise_linexp(exp):
    """
    Serialise a restricted subset of :py:class:`~vc2_bit_widths.linexp.LinExp`
    s into JSON form.
    
    Restrictions:
    
    * Only supports expressions made up of rational values (i.e. no floating
      point coefficients/constants.
    * Symbols must all be strings (i.e. no tuples or
      :py:class:`~vc2_bit_widths.linexp.AAError` sybols)
    
    Example::
        >>> before = LinExp({
        ...     "a": Fraction(1, 5),
        ...     "b": Fraction(10, 1),
        ...     None: Fraction(-2, 3),
        ... })
        >>> serialise_linexp(before)
        [
            {"symbol": "a", "numer": "1", "denom": "5"},
            {"symbol": "b", "numer": "10", "denom": "1"},
            {"symbol": None, "numer": "-2", "denom": "3"},
        ]
    
    .. note::
        
        Numbers are encoded as strings to avoid floating point precision
        limitations.
    """
    return [
        {
            "symbol": sym,
            "numer": str(coeff.numerator),
            "denom": str(coeff.denominator),
        }
        for sym, coeff in exp
    ]

def deserialise_linexp(json):
    """
    Inverse of :py:func:`serialise_linexp`.
    """
    return LinExp({
        d["symbol"]: Fraction(int(d["numer"]), int(d["denom"]))
        for d in json
    })


def serialise_signal_bounds(signal_bounds):
    """
    Convert a dictionary of analysis or synthesis signal bounds expressions
    into a JSON-serialisable form.
    
    See :py:func:`serialise_linexp` for details of the ``"lower_bound"`` and
    ``"upper_bound"`` fields.
    
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
    return serialise_intermediate_value_dictionary(OrderedDict(
        (
            key,
            {
                "lower_bound": serialise_linexp(LinExp(lower_bound)),
                "upper_bound": serialise_linexp(LinExp(upper_bound)),
            },
        )
        for key, (lower_bound, upper_bound) in signal_bounds.items()
    ))


def deserialise_signal_bounds(signal_bounds):
    """
    Inverse of :py:func:`serialise_signal_bounds`.
    """
    return OrderedDict(
        (
            key,
            (
                deserialise_linexp(d["lower_bound"]),
                deserialise_linexp(d["upper_bound"]),
            )
        )
        for key, d in deserialise_intermediate_value_dictionary(signal_bounds).items()
    )


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
    return serialise_intermediate_value_dictionary(OrderedDict(
        (
            key,
            {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )
        for key, (lower_bound, upper_bound) in signal_bounds.items()
    ))


def deserialise_concrete_signal_bounds(signal_bounds):
    """
    Inverse of :py:func:`serialise_concrete_signal_bounds`.
    """
    return OrderedDict(
        (
            key,
            (d["lower_bound"], d["upper_bound"]),
        )
        for key, d in deserialise_intermediate_value_dictionary(signal_bounds).items()
    )


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
    namedtuple_type : :py:func:`~collections.namedtuple`
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


def serialise_test_pattern(test_pattern):
    """
    Convert a :py:class:`TestPattern` into a compact JSON-serialisable form.
    
    For example::
        >>> before = TestPattern({
        ...     (4, 10): +1, (5, 10): -1, (6, 10): +1, (7, 10): -1,
        ...     (4, 11): -1, (5, 11): +1, (6, 11): -1, (7, 11): +1,
        ...     (4, 12): +1, (5, 12): -1, (6, 12): +1, (7, 12): -1,
        ...     (4, 13): -1, (5, 13): +1, (6, 13): -1, (7, 13): +1,
        ... })
        >>> serialise_test_pattern(before)
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
    # Bottom-left corner of the sparse matrix
    dy, dx = test_pattern.origin
    
    # Active area of the sparse matrix
    height, width = test_pattern.polarities.shape
    
    # Convert into positive + mask representation
    positive = test_pattern.polarities > 0
    mask = test_pattern.polarities != 0
    
    packed_positive = np.packbits(positive).tobytes()
    packed_mask = np.packbits(mask).tobytes()
    
    out = {
        "dx": int(dx),
        "dy": int(dy),
        "width": int(width),
        "height": int(height),
        "positive": b64encode(packed_positive).decode("ascii"),
        "mask": b64encode(packed_mask).decode("ascii"),
    }
    
    return out


def deserialise_test_pattern(dictionary):
    """
    Inverse of :py:func:`serialise_namedtuple`.
    """
    origin = (dictionary["dy"], dictionary["dx"])
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
    )[:count].reshape((height, width)).astype(bool)
    
    mask = np.unpackbits(
        packed_mask,
    )[:count].reshape((height, width)).astype(bool)
    
    polarities = np.full((height, width), -1, dtype=np.int8)
    polarities[positive] = +1
    polarities[~mask] = 0
    
    return TestPattern(origin, polarities)


def serialise_test_pattern_specifications(spec_namedtuple_type, test_patterns):
    """
    Convert a dictionary of analysis or synthesis test pattern specifications into a
    JSON-serialisable form.
    
    See :py:func:`serialise_test_pattern` for the serialisation used for the
    pattern data.
    
    For example::
    
        >>> before = {
        ...     (1, "LH", 2, 3): TestPatternSpecification(
        ...         target=(4, 5),
        ...         pattern=TestPattern({(x, y): polarity, ...}),
        ...         pattern_translation_multiple=(6, 7),
        ...         target_translation_multiple=(8, 9),
        ...     ),
        ...     ...
        ... }
        >>> serialise_test_pattern_specifications(TestPatternSpecification, before)
        [
            {
                "level": 1,
                "array_name": "LH",
                "phase": [2, 3],
                "target": [4, 5],
                "pattern": {...},
                "pattern_translation_multiple": [6, 7],
                "target_translation_multiple": [8, 9],
            },
            ...
        ]
    
    Parameters
    ==========
    spec_namedtuple_type : :py:func:`~collections.namedtuple` class
        The namedtuple used to hold the test pattern specification. One of
        :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`
        or
        :py:class:`~vc2_bit_widths.patterns.OptimisedTestPatternSpecification`.
    test_patterns : {(level, array_name, x, y): (...), ...}
    """
    out = serialise_intermediate_value_dictionary(OrderedDict(
        (
            key,
            serialise_namedtuple(spec_namedtuple_type, value),
        )
        for key, value in test_patterns.items()
    ))
    
    for d in out:
        d["pattern"] = serialise_test_pattern(d["pattern"])
    
    return out


def deserialise_test_pattern_specifications(spec_namedtuple_type, test_patterns):
    """
    Inverse of :py:func:`serialise_test_pattern_specifications`.
    """
    for d in test_patterns:
        d["pattern"] = deserialise_test_pattern(d["pattern"])
        d["target"] = tuple(d["target"])
        d["pattern_translation_multiple"] = tuple(d["pattern_translation_multiple"])
        d["target_translation_multiple"] = tuple(d["target_translation_multiple"])
    
    return OrderedDict(
        (
            key,
            deserialise_namedtuple(spec_namedtuple_type, value),
        )
        for key, value in deserialise_intermediate_value_dictionary(test_patterns).items()
    )


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
