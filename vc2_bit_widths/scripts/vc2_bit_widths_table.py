"""
Produce tables of theoretical signal bounds
===========================================

"""

import sys

import json

import csv

from collections import OrderedDict

from argparse import ArgumentParser, FileType

from vc2_bit_widths.linexp import LinExp

from vc2_bit_widths.signal_bounds import (
    twos_compliment_bits,
)

from vc2_bit_widths.helpers import (
    evaluate_filter_bounds,
)

from vc2_bit_widths.json_serialisations import (
    deserialise_signal_bounds,
)


def parse_args(args=None):
    parser = ArgumentParser(description="""
        Present computed bit width information in tabular form for input
        signals with a particular picture signal range.
    """)
    
    parser.add_argument(
        "static_filter_analysis",
        type=FileType("r"),
        help="""
            The static analysis JSON data produced by
            vc2-static-filter-analysis.
        """,
    )
    
    parser.add_argument(
        "--picture-bit-widths", "-b",
        type=int, nargs="+", default=[8, 10, 12, 16],
        help="""
            The number of bits in the picture signal. Multiple widths may be
            given and values will be computed for each.
        """,
    )
    
    parser.add_argument(
        "--show-all-filter-phases", "-p",
        action="store_true", default=False,
        help="""
            Show signal bounds broken down into individual filter phases.
            Without this option, the bounds are shown for all filter phases.
        """,
    )
    
    parser.add_argument(
        "--verbose", "-v", default=0, action="count",
        help="""
            Show more detailed status information during execution.
        """,
    )
    
    parser.add_argument(
        "--output", "-o",
        type=FileType("w"), default=sys.stdout,
        help="""
            The name of the file to write the CSV bit widths table to. Defaults
            to stdout.
        """
    )
    
    return parser.parse_args(args)


def dict_aggregate(dictionary, key_map, value_reduce):
    """
    Given a dictionary, combine entries together to aggregate the data set in
    some useful way.
    
    Parameters
    ==========
    dictionary : :py:class:`dict`
        The input dictionary which will be aggregated.
    key_map : fn(key) -> new_key
        Map keys in ``dictionary`` to new keys in the output. Typically several
        keys in the input dictionary will be mapped to the same key in the
        output dictionary.
    value_reduce : fn(value_a, value_b) -> value_combined
        Function which takes two dictionary values and combines/reduces them
        into a single value which is representative of the aggregate of both
        dictionaries.
    """
    out = type(dictionary)()
    
    for key, value in dictionary.items():
        new_key = key_map(key)
        
        if new_key not in out:
            out[new_key] = value
        else:
            out[new_key] = value_reduce(out[new_key], value)
    
    return out


def dict_join(dictionaries):
    """
    Perform a join over a set of dictionaries.
    
    Parameters
    ==========
    dictionaries : [dict, ...]
        A list of dictionaries which all have the same keys.
    
    Returns
    =======
    dictionary : {key, (dictionaries[0][key], dictionaries[1][key], ...), ...}
        A joined dictionary where every value is a tuple containing the
        corresponding values of the input dictionaries.
    """
    # Sanity check
    assert len(dictionaries) >= 1
    dict_keys = [set(d) for d in dictionaries]
    assert all(d == dict_keys[0] for d in dict_keys[1:])
    
    out = type(dictionaries[0])()
    
    for key in dictionaries[0]:
        out[key] = tuple(
            d[key]
            for d in dictionaries
        )
    
    return out


def strip_phase(key):
    level, array_name, x, y = key
    return (level, array_name)


def combine_bounds(a, b):
    lower_a, upper_a = a
    lower_b, upper_b = b
    return (min(lower_a, lower_b), max(upper_a, upper_b))


def main(args=None):
    args = parse_args(args)
    
    # Load precomputed signal bounds
    static_filter_analysis = json.load(args.static_filter_analysis)
    analysis_signal_bounds = deserialise_signal_bounds(
        static_filter_analysis["analysis_signal_bounds"]
    )
    synthesis_signal_bounds = deserialise_signal_bounds(
        static_filter_analysis["synthesis_signal_bounds"]
    )
    
    # Compute signal bounds for all specified bit widths
    #
    # analysis_bounds_dicts = [{(level, array_name, x, y): (lower_bound, upper_bound), ...}, ...]
    # synthesis_bounds_dicts = same as above
    analysis_bounds_dicts, synthesis_bounds_dicts = zip(*(
        evaluate_filter_bounds(
            analysis_signal_bounds,
            synthesis_signal_bounds,
            picture_bit_width,
        )
        for picture_bit_width in args.picture_bit_widths
    ))
    
    # Aggregate bounds to required level of detail (e.g. aggregate together all
    # filter phases)
    columns = ("level", "array_name", "x", "y")
    if not args.show_all_filter_phases:
        columns = ("level", "array_name")
        analysis_bounds_dicts = [
            dict_aggregate(d, strip_phase, combine_bounds)
            for d in analysis_bounds_dicts
        ]
        synthesis_bounds_dicts = [
            dict_aggregate(d, strip_phase, combine_bounds)
            for d in synthesis_bounds_dicts
        ]
    
    analysis_bounds = dict_join(analysis_bounds_dicts)
    synthesis_bounds = dict_join(synthesis_bounds_dicts)
    
    # Combine all bounds & bit widths
    all_bounds = OrderedDict()
    all_bounds.update((("analysis", ) + key, value) for key, value in analysis_bounds.items())
    all_bounds.update((("synthesis", ) + key, value) for key, value in synthesis_bounds.items())
    
    csv_writer = csv.writer(args.output)
    
    # Header
    csv_writer.writerow(
        ("type", ) +
        columns +
        (("lower", "upper", "bits")*len(args.picture_bit_widths))
    )
    
    # Data
    for key, bounds in all_bounds.items():
        bounds_and_bits = []
        for lower_bound, upper_bound in bounds:
            num_bits = max(
                twos_compliment_bits(lower_bound),
                twos_compliment_bits(upper_bound),
            )
            bounds_and_bits.append((lower_bound, upper_bound, num_bits))
        
        csv_writer.writerow(key + sum(bounds_and_bits, tuple()))
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
