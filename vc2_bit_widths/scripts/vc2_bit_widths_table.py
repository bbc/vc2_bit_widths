r"""
.. _vc2-bit-widths-table:

``vc2-bit-widths-table``
========================

Compute the signal ranges and bit widths required for each part of a
VC-2 analysis and synthesis filter and present the results in a CSV
table.

.. note::

    The values printed by this tool are valid only for the wavelet transform,
    depth, picture bit width and quantisation matrix specified. See
    :ref:`caveats` for further limitations and assumptions made by this
    software.

Example usage
-------------

::

    $ vc2-static-filter-analysis \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 2 \
        --output static_analysis.json
    
    $ vc2-bit-widths-table \
        static_analysis.json \
        --picture-bit-width 10 \
        --custom-quantisation-matrix \
            0 LL 1 \
            1 HL 2   1 LH 0   1 HH 4 \
            2 HL 1   2 LH 3   2 HH 3 \
        --output bit_widths.csv
    
    $ column -t -s, bit_widths.csv | head
    type       level  array_name  lower_bound  test_pattern_min  test_pattern_max  upper_bound  bits
    analysis   2      Input       -512         -512              511               511          10
    analysis   2      DC          -1024        -1024             1022              1022         11
    analysis   2      DC'         -2047        -2046             2046              2047         12
    analysis   2      DC''        -2047        -2046             2046              2047         12
    analysis   2      L           -1537        -1535             1534              1535         12
    analysis   2      H           -2047        -2046             2046              2047         12
    analysis   2      L'          -3071        -3069             3069              3071         13
    analysis   2      H'          -4094        -4092             4092              4094         13
    analysis   2      L''         -3071        -3069             3069              3071         13

The command can also be used to display the signal ranges of optimised test
patterns produced by ``vc2-optimise-synthesis-test-patterns``::

    $ vc2-bit-widths-table \
        static_analysis.json \
        optimised_patterns.json \
        --output bit_widths.csv


Arguments
---------

The complete set of arguments can be listed using ``--help``

.. program-output:: vc2-bit-widths-table --help


CSV Format
----------

The generated CSV have one row per analysis and synthesis filter phase. The
columns are defined as follows:

type, level, array_name, x, y
    Specifies the filter phase represented by the row (see :ref:`terminology`).
    The 'x' and 'y' columns are omitted, and the aggregate worst-case shown for
    all phases unless ``--show-all-filter-phases`` is used. 

lower_bound, upper_bound
    The theoretical worst-case lower- and upper-bounds for the signal. These
    values may be over-estimates but are guaranteed not to be under estimates
    of the true worst-case.

test_pattern_min, test_pattern_max
    The minimum and maximum values produced by the test patterns when they're
    passed through a real encoder and decoder. These values may not represent
    true worst-case signal levels, though they may often be close.
    
    .. warning::
        It is possible that both minimum and maximum test pattern values could
        have the same sign. This occurs when quantisation errors are
        sufficiently strong as to make worst-case encodings/decodings of the
        test pattern have the same sign. The true signal range would still
        include zero.

bits
    The final column summarises the number of bits required for a signed two's
    complement representation of the values in the ranges indicated. If the
    test pattern and theory disagree, a range of bits is indicated. The true
    bit width required lies somewhere in the given range.

"""

import sys

import csv

import logging

from collections import OrderedDict

from argparse import ArgumentParser, FileType

from vc2_bit_widths.scripts.loader_utils import (
    load_filter_analysis,
)

from vc2_bit_widths.signal_bounds import (
    twos_compliment_bits,
)

from vc2_bit_widths.helpers import (
    evaluate_test_pattern_outputs,
)


def parse_args(arg_strings=None):
    parser = ArgumentParser(description="""
        Compute the signal ranges and bit widths required for each part of a
        VC-2 analysis and synthesis filter and present the results in a CSV
        table.
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
        "optimised_synthesis_test_patterns",
        type=FileType("r"), nargs="?",
        help="""
            A set of optimised synthesis test patterns produced by
            vc2-optimise-synthesis-test-patterns.
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
    
    if args.optimised_synthesis_test_patterns is not None:
        if (
            args.picture_bit_width is not None or
            args.custom_quantisation_matrix is not None
        ):
            parser.error(
                "--picture-bit-width/-b and --custom-quantisation-matrix/-q "
                "must not be used when optimised synthesis test patterns "
                "are provided"
            )
    else:
        if args.picture_bit_width is None:
            parser.error(
                "A --picture-bit-width/-b argument or a set of "
                "optimised synthesis test patterns are required."
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
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    (
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        picture_bit_width,
        max_quantisation_index,
        concrete_analysis_bounds,
        concrete_synthesis_bounds,
        analysis_test_patterns,
        synthesis_test_patterns,
    ) = load_filter_analysis(
        args.static_filter_analysis,
        args.optimised_synthesis_test_patterns,
        args.custom_quantisation_matrix,
        args.picture_bit_width,
    )
    
    
    # Find test pattern output values for each bit width
    analysis_outputs, synthesis_outputs = evaluate_test_pattern_outputs(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        picture_bit_width,
        quantisation_matrix,
        max_quantisation_index,
        analysis_test_patterns,
        synthesis_test_patterns,
    )
    
    # Strip quantisation index from synthesis output info and put bounds in the
    # correct order (since the minimised/maximised test patterns may actually
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
        ("lower_bound", "test_pattern_min", "test_pattern_max", "upper_bound", "bits")
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
