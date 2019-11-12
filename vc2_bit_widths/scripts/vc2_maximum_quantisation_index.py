"""
Compute the largest quantisation index required for a VC-2 codec
================================================================

"""

import sys

import json

from argparse import ArgumentParser, FileType

from vc2_data_tables import QUANTISATION_MATRICES

from vc2_bit_widths.scripts.argument_parsers import (
    parse_quantisation_matrix_argument,
)

from vc2_bit_widths.helpers import (
    evaluate_filter_bounds,
    quantisation_index_bound,
)

from vc2_bit_widths.json_serialisations import (
    deserialise_signal_bounds,
)


def parse_args(args=None):
    parser = ArgumentParser(description="""
        Compute the maximum quantisation index which is useful for a particular
        VC-2 codec configuration.
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
        "--picture-bit-width", "-b",
        type=int, required=True,
        help="""
            The number of bits in the picture signal.
        """,
    )
    
    parser.add_argument(
        "--custom-quantisation-matrix", "-q",
        nargs="+",
        help="""
            Define the custom quantisation matrix used by a codec. Optional
            except for filters without a default quantisation matrix defined.
            Should be specified as a series 3-argument tuples giving the level,
            orientation and quantisation matrix value for every entry in the
            quantisation matrix.
        """,
    )
    
    parser.add_argument(
        "--verbose", "-v", default=0, action="count",
        help="""
            Show more detailed status information during execution.
        """,
    )
    
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    
    # Load precomputed signal bounds
    static_filter_analysis = json.load(args.static_filter_analysis)
    
    quantisation_matrix = parse_quantisation_matrix_argument(
        args.custom_quantisation_matrix,
        static_filter_analysis["wavelet_index"],
        static_filter_analysis["wavelet_index_ho"],
        static_filter_analysis["dwt_depth"],
        static_filter_analysis["dwt_depth_ho"],
    )
    
    analysis_signal_bounds = deserialise_signal_bounds(
        static_filter_analysis["analysis_signal_bounds"]
    )
    synthesis_signal_bounds = deserialise_signal_bounds(
        static_filter_analysis["synthesis_signal_bounds"]
    )
    
    (
        concrete_analysis_signal_bounds,
        concrete_synthesis_signal_bounds,
    ) = evaluate_filter_bounds(
        static_filter_analysis["wavelet_index"],
        static_filter_analysis["wavelet_index_ho"],
        static_filter_analysis["dwt_depth"],
        static_filter_analysis["dwt_depth_ho"],
        analysis_signal_bounds,
        synthesis_signal_bounds,
        args.picture_bit_width,
    )
    
    print(quantisation_index_bound(
        concrete_analysis_signal_bounds,
        quantisation_matrix,
    ))
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
