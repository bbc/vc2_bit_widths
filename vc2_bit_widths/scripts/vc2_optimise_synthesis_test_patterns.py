"""
Use a stochastic search to produce adversarial test patterns
============================================================

"""

import sys

import json

import logging

import numpy as np

from argparse import ArgumentParser, FileType

from vc2_data_tables import QUANTISATION_MATRICES

from vc2_bit_widths.scripts.argument_parsers import (
    parse_quantisation_matrix_argument,
)

from vc2_bit_widths.pattern_generation import (
    TestPatternSpecification,
    OptimisedTestPatternSpecification,
)

from vc2_bit_widths.json_serialisations import (
    deserialise_signal_bounds,
    deserialise_test_patterns,
    serialise_test_patterns,
    deserialise_test_patterns,
    serialise_quantisation_matrix,
)

from vc2_bit_widths.helpers import (
    evaluate_filter_bounds,
    quantisation_index_bound,
    optimise_synthesis_test_patterns,
)


def parse_args(args=None):
    parser = ArgumentParser(description="""
        Use a stochastic search to produce adversarial test patterns.
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
            Use a custom quantisation matrix for the search. Optional except
            for filters without a default quantisation matrix defined. Should
            be specified as a series 3-argument tuples giving the level,
            orientation and quantisation matrix value for every entry in the
            quantisation matrix.
        """,
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int, default=1,
        help="""
            The seed for the random number generator used to perform searches.
        """,
    )
    
    parser.add_argument(
        "--number-of-searches", "-N",
        type=int, default=10,
        help="""
            The number of independent search runs to use. The larger this
            option, the more likely it is that the search will avoid local
            minima (default: %(default)s).
        """,
    )
    parser.add_argument(
        "--terminate-early", "-t",
        type=int, default=1,
        help="""
            Terminate optimisation early if this many searches fail to find an
            improvement. (Default: %(default)s).
        """,
    )
    
    parser.add_argument(
        "--added-corruption-rate", "-a",
        type=float, default=0.05,
        help="""
            The proportion of pixel values to corrupt in each search
            iteration.  (Default: %(default)s).
        """,
    )
    
    parser.add_argument(
        "--removed-corruption-rate", "-r",
        type=float, default=0.0,
        help="""
            The proportion of pixel values to reset to their original state in
            each search iteration.  (Default: %(default)s).
        """,
    )
    
    parser.add_argument(
        "--base-iterations", "-i",
        type=int, default=200,
        help="""
            The base number of trials to run the search for. The larger this
            value, the longer the search will run without finding any
            improvements before terminating. (Default: %(default)s).
        """,
    )
    
    parser.add_argument(
        "--added-iterations-per-improvement", "-I",
        type=int, default=200,
        help="""
            The number of additional search iterations to perform when an
            improved test pattern is found. (Default: %(default)s).
        """,
    )
    
    parser.add_argument(
        "--output", "-o",
        type=FileType("w"), default=sys.stdout,
        help="""
            The name of the file to write the optimised test pattern JSON file
            to. Defaults to stdout.
        """
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
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
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
    synthesis_test_patterns = deserialise_test_patterns(
        TestPatternSpecification,
        static_filter_analysis["synthesis_test_patterns"]
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
    
    max_quantisation_index = quantisation_index_bound(
        concrete_analysis_signal_bounds,
        quantisation_matrix,
    )
    
    random_state = np.random.RandomState(args.seed)
    
    optimised_synthesis_test_patterns = optimise_synthesis_test_patterns(
        wavelet_index=static_filter_analysis["wavelet_index"],
        wavelet_index_ho=static_filter_analysis["wavelet_index_ho"],
        dwt_depth=static_filter_analysis["dwt_depth"],
        dwt_depth_ho=static_filter_analysis["dwt_depth_ho"],
        quantisation_matrix=quantisation_matrix,
        picture_bit_width=args.picture_bit_width,
        synthesis_test_patterns=synthesis_test_patterns,
        max_quantisation_index=max_quantisation_index,
        random_state=random_state,
        number_of_searches=args.number_of_searches,
        terminate_early=args.terminate_early,
        added_corruption_rate=args.added_corruption_rate,
        removed_corruption_rate=args.removed_corruption_rate,
        base_iterations=args.base_iterations,
        added_iterations_per_improvement=args.added_iterations_per_improvement,
    )
    
    out = {
        "wavelet_index": static_filter_analysis["wavelet_index"],
        "wavelet_index_ho": static_filter_analysis["wavelet_index_ho"],
        "dwt_depth": static_filter_analysis["dwt_depth"],
        "dwt_depth_ho": static_filter_analysis["dwt_depth_ho"],
        "picture_bit_width": args.picture_bit_width,
        "quantisation_matrix": serialise_quantisation_matrix(
            quantisation_matrix,
        ),
        "optimised_synthesis_test_patterns": serialise_test_patterns(
            OptimisedTestPatternSpecification,
            optimised_synthesis_test_patterns,
        )
    }
    
    json.dump(out, args.output)
    args.output.write("\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
