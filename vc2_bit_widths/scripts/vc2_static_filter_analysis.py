"""
Static VC-2 Filter Analysis
===========================

"""

import sys

import json

import logging

from argparse import ArgumentParser, FileType

from vc2_data_tables import WaveletFilters

from vc2_bit_widths.pattern_generation import TestPatternSpecification

from vc2_bit_widths.helpers import static_filter_analysis

from vc2_bit_widths.json_serialisations import (
    serialise_signal_bounds,
    serialise_test_patterns,
)

from vc2_bit_widths.scripts.argument_parsers import wavelet_index_or_name


def parse_args(args=None):
    parser = ArgumentParser(description="""
        Statically analyse a VC-2 filter configuration to determine its signal
        range bounds and worst-case test patterns in the general case. This
        tool outputs a JSON dump of the computed information which can later be
        further processed and displayed by other tools.
    """)
    
    parser.add_argument(
        "--wavelet-index", "-w",
        type=wavelet_index_or_name, required=True,
        help="""
            The VC-2 wavelet index for the wavelet transform. One of: {}.
        """.format(", ".join(
            "{} or {}".format(int(index), index.name)
            for index in WaveletFilters
        )),
    )
    parser.add_argument(
        "--wavelet-index-ho", "-W",
        type=wavelet_index_or_name,
        help="""
            The VC-2 wavelet index for the horizontal parts of the wavelet
            transform. If not specified, assumed to be the same as
            --wavelet-index/-w.
        """,
    )
    parser.add_argument(
        "--dwt-depth", "-d",
        type=int, default=0,
        help="""
            The VC-2 transform depth. Defaults to 0 if not specified.
        """
    )
    parser.add_argument(
        "--dwt-depth-ho", "-D",
        type=int, default=0,
        help="""
            The VC-2 horizontal-only transform depth. Defaults to 0 if not
            specified.
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=FileType("w"), default=sys.stdout,
        help="""
            The name of the file to write the computed JSON output to. Defaults
            to stdout.
        """
    )
    
    parser.add_argument(
        "--verbose", "-v", default=0, action="count",
        help="""
            Show more detailed status information during execution.
        """
    )
    
    args = parser.parse_args(args)
    
    if args.wavelet_index_ho is None:
        args.wavelet_index_ho = args.wavelet_index
    
    return args


def main(args=None):
    args = parse_args(args)
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Perform the analysis
    (
        analysis_signal_bounds,
        synthesis_signal_bounds,
        analysis_test_patterns,
        synthesis_test_patterns,
    )= static_filter_analysis(
        wavelet_index=args.wavelet_index,
        wavelet_index_ho=args.wavelet_index_ho,
        dwt_depth=args.dwt_depth,
        dwt_depth_ho=args.dwt_depth_ho,
    )
    
    # Serialise
    out = {
        "wavelet_index": args.wavelet_index,
        "wavelet_index_ho": args.wavelet_index_ho,
        "dwt_depth": args.dwt_depth,
        "dwt_depth_ho": args.dwt_depth_ho,
        "analysis_signal_bounds":
            serialise_signal_bounds(analysis_signal_bounds),
        "synthesis_signal_bounds":
            serialise_signal_bounds(synthesis_signal_bounds),
        "analysis_test_patterns":
            serialise_test_patterns(TestPatternSpecification, analysis_test_patterns),
        "synthesis_test_patterns":
            serialise_test_patterns(TestPatternSpecification, synthesis_test_patterns),
    }
    
    json.dump(out, args.output)
    args.output.write("\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
