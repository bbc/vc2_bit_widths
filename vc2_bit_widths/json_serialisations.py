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
                "picture": [[x, y, polarity], ...],
                "picture_translation_multiple": [6, 7],
                "target_translation_multiple": [8, 9],
            },
            ...
        ]
    
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
        d["picture"] = [
            [x, y, value]
            for (x, y), value in d["picture"].items()
        ]
    
    return out


def deserialise_test_signals(spec_namedtuple_type, test_signals):
    """
    Inverse of :py:func:`serialise_test_signals`.
    """
    for d in test_signals:
        d["picture"] = {
            (x, y): value
            for x, y, value in d["picture"]
        }
        d["target"] = tuple(d["target"])
        d["picture_translation_multiple"] = tuple(d["picture_translation_multiple"])
        d["target_translation_multiple"] = tuple(d["target_translation_multiple"])
    
    return {
        key: deserialise_namedtuple(spec_namedtuple_type, value)
        for key, value in deserialise_intermediate_value_dictionary(test_signals).items()
    }
