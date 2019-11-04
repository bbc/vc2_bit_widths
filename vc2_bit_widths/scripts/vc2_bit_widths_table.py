"""
Produce tables of theoretical signal bounds
===========================================

"""

import sys

import json

import csv

from collections import OrderedDict

from argparse import ArgumentParser, FileType

from vc2_bit_widths.scripts.argument_parsers import (
    parse_quantisation_matrix_argument,
)

from vc2_bit_widths.linexp import LinExp

from vc2_bit_widths.signal_bounds import (
    twos_compliment_bits,
)

from vc2_bit_widths.signal_generation import (
    TestSignalSpecification,
    OptimisedTestSignalSpecification,
)

from vc2_bit_widths.helpers import (
    evaluate_filter_bounds,
    quantisation_index_bound,
    evaluate_test_signal_outputs,
    add_omitted_synthesis_values,
)

from vc2_bit_widths.json_serialisations import (
    deserialise_signal_bounds,
    deserialise_test_signals,
    deserialise_quantisation_matrix,
)


def parse_args(arg_strings=None):
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
        "optimised_synthesis_test_signals",
        type=FileType("r"), nargs="?",
        help="""
            A set of optimised synthesis test signals produced by
            vc2-optimise-synthesis-test-signals.
        """,
    )
    
    parser.add_argument(
        "--picture-bit-width", "-b",
        type=int,
        help="""
            The number of bits in the picture signal.
        """,
    )
    
    parser.add_argument(
        "--custom-quantisation-matrix", "-q",
        nargs="+",
        help="""
            Use a custom quantisation matrix. Optional except for filters
            without a default quantisation matrix defined. Should be specified
            as a series 3-argument tuples giving the level, orientation and
            quantisation matrix value for every entry in the quantisation
            matrix.
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
    
    args = parser.parse_args(arg_strings)
    
    if args.optimised_synthesis_test_signals is not None:
        if (
            args.picture_bit_width is not None or
            args.custom_quantisation_matrix is not None
        ):
            parser.error(
                "--picture-bit-width/-b and --custom-quantisation-matrix/-q "
                "must not be used when optimised synthesis test signals "
                "are provided"
            )
    else:
        if args.picture_bit_width is None:
            parser.error(
                "A --picture-bit-width/-b argument or a set of "
                "optimised synthesis test signals are required."
            )
    
    return args


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
    
    # Load precomputed test signals
    analysis_test_signals = deserialise_test_signals(
        TestSignalSpecification,
        static_filter_analysis["analysis_test_signals"]
    )
    synthesis_test_signals = deserialise_test_signals(
        TestSignalSpecification,
        static_filter_analysis["synthesis_test_signals"]
    )
    
    # Load optimised synthesis signal
    if args.optimised_synthesis_test_signals is not None:
        optimised_json = json.load(args.optimised_synthesis_test_signals)
        
        assert static_filter_analysis["wavelet_index"] == optimised_json["wavelet_index"]
        assert static_filter_analysis["wavelet_index_ho"] == optimised_json["wavelet_index_ho"]
        assert static_filter_analysis["dwt_depth"] == optimised_json["dwt_depth"]
        assert static_filter_analysis["dwt_depth_ho"] == optimised_json["dwt_depth_ho"]
        
        args.picture_bit_width = optimised_json["picture_bit_width"]
        
        quantisation_matrix = deserialise_quantisation_matrix(
            optimised_json["quantisation_matrix"]
        )
        
        synthesis_test_signals = deserialise_test_signals(
            OptimisedTestSignalSpecification,
            optimised_json["optimised_synthesis_test_signals"]
        )
    else:
        quantisation_matrix = parse_quantisation_matrix_argument(
            args.custom_quantisation_matrix,
            static_filter_analysis["wavelet_index"],
            static_filter_analysis["wavelet_index_ho"],
            static_filter_analysis["dwt_depth"],
            static_filter_analysis["dwt_depth_ho"],
        )
    
    # Compute signal bounds for all specified bit widths
    #
    # analysis_bounds_dicts = [{(level, array_name, x, y): (lower_bound, upper_bound), ...}, ...]
    # synthesis_bounds_dicts = same as above
    concrete_analysis_bounds, concrete_synthesis_bounds = evaluate_filter_bounds(
        analysis_signal_bounds,
        synthesis_signal_bounds,
        args.picture_bit_width,
    )
    
    # Find the maximum quantisation index for each bit width
    max_quantisation_index = quantisation_index_bound(
        concrete_analysis_bounds,
        quantisation_matrix,
    )
    
    # Find test signal output values for each bit width
    analysis_outputs, synthesis_outputs = evaluate_test_signal_outputs(
        static_filter_analysis["wavelet_index"],
        static_filter_analysis["wavelet_index_ho"],
        static_filter_analysis["dwt_depth"],
        static_filter_analysis["dwt_depth_ho"],
        args.picture_bit_width,
        quantisation_matrix,
        max_quantisation_index,
        analysis_test_signals,
        synthesis_test_signals,
    )
    
    # Re-add interleaved values (if absent)
    synthesis_outputs = add_omitted_synthesis_values(
        static_filter_analysis["wavelet_index"],
        static_filter_analysis["wavelet_index_ho"],
        static_filter_analysis["dwt_depth"],
        static_filter_analysis["dwt_depth_ho"],
        synthesis_outputs,
    )
    
    # Strip quantisation index from synthesis output info and put bounds in the
    # correct order (since the minimised/maximised test signals may actually
    # produce outputs with the wrong sign in cases of extreme quantisation)
    synthesis_outputs = OrderedDict(
        (key, (min(lower, upper), max(lower, upper)))
        for key, ((lower, _), (upper, _)) in synthesis_outputs.items()
    )
    
    # Aggregate bounds to required level of detail (e.g. aggregate together all
    # filter phases)
    columns = ("level", "array_name", "x", "y")
    if not args.show_all_filter_phases:
        columns = ("level", "array_name")
        concrete_analysis_bounds = dict_aggregate(
            concrete_analysis_bounds,
            strip_phase,
            combine_bounds,
        )
        concrete_synthesis_bounds = dict_aggregate(
            concrete_synthesis_bounds,
            strip_phase,
            combine_bounds,
        )
        analysis_outputs = dict_aggregate(
            analysis_outputs,
            strip_phase,
            combine_bounds,
        )
        synthesis_outputs = dict_aggregate(
            synthesis_outputs,
            strip_phase,
            combine_bounds,
        )
    
    # Combine all bounds & outputs
    all_bounds = OrderedDict(
        (
            (type_name, ) + key,
            (
                bounds[key][0],
                outputs[key][0],
                outputs[key][1],
                bounds[key][1],
            ),
        )
        for type_name, bounds, outputs in [
            ("analysis", concrete_analysis_bounds, analysis_outputs),
            ("synthesis", concrete_synthesis_bounds, synthesis_outputs),
        ]
        for key in bounds.keys()
    )
    
    csv_writer = csv.writer(args.output)
    
    # Header
    csv_writer.writerow(
        ("type", ) +
        columns +
        ("lower_bound", "test_signal_min", "test_signal_max", "upper_bound", "bits")
    )
    
    # Data
    for key, (lower_bound, output_min, output_max, upper_bound) in all_bounds.items():
        output_num_bits = max(
            twos_compliment_bits(output_min),
            twos_compliment_bits(output_max),
        )
        bounds_num_bits = max(
            twos_compliment_bits(lower_bound),
            twos_compliment_bits(upper_bound),
        )
        if output_num_bits == bounds_num_bits:
            num_bits = str(bounds_num_bits)
        else:
            num_bits = "{}-{}".format(
                output_num_bits,
                bounds_num_bits,
            )
        
        csv_writer.writerow(key + (lower_bound, output_min, output_max, upper_bound, num_bits))
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
