r"""
.. _vc2-optimise-synthesis-test-patterns:

``vc2-optimise-synthesis-test-patterns``
========================================

Use a stochastic search to enchance the synthesis test patterns
produced by vc2-static-filter-analysis so that they produce even more
extreme values.

.. note::

    The enhanced test patterns produced by this tool are valid only for the
    wavelet transform, depth, picture bit width and quantisation matrix
    specified. See :ref:`caveats` for further limitations and assumptions made
    by this software.

Example usage
-------------

::

    $ vc2-static-filter-analysis \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 2 \
        --output static_analysis.json
    
    $ vc2-maximum-quantisation-index \
        static_analysis.json \
        --picture-bit-width 10 \
        --custom-quantisation-matrix \
            0 LL 1 \
            1 HL 2   1 LH 0   1 HH 4 \
            2 HL 1   2 LH 3   2 HH 3 \
        --number-of-searches 10 \
        --added-corruption-rate 0.2 \
        --removed-corruption-rate 0.05 \
        --base-iterations 1000 \
        --added-iterations-per-improvement 500 \
        --output optimised_patterns.json

Arguments
---------

The complete set of arguments can be listed using ``--help``

.. program-output:: vc2-optimise-synthesis-test-patterns --help


.. _vc2-optimise-synthesis-test-patterns-parameter-tuning:

Choosing parameters
-------------------

The runtime and output quality of the optimisation algorithm is highly
dependent on the paramters supplied. The best performing parameters for one
wavelet transform may not be the best for others.

Choosing good parameters requires manual exploration and experimentation and is
unfortunately out of the scope of this manual.

See :py:func:`~vc2_bit_widths.helpers.optimise_synthesis_test_patterns` for
additional details.


JSON file format
----------------

The output is a JSON file with the following structure::

    {
        "wavelet_index": <int>,
        "wavelet_index_ho": <int>,
        "dwt_depth": <int>,
        "dwt_depth_ho": <int>,
        "picture_bit_width": <int>,
        "quantisation_matrix": <quantisation-matrix>,
        "optimised_synthesis_test_patterns": [<test-pattern-specification>, ...],
    }

Quantisation matrix
```````````````````

The ``"quantisation_matrix"`` field encodes the quantisation matrix used in the
same fashion as the VC-2 pseudocode and its format should be self-explanatory.
For example, the quantisation matrix passed to the example above is encoded
as::

    {
        0: {"LL": 1},
        1: {"HL": 2, "LH": 0, "HH": 4},
        2: {"HL": 1, "LH": 3, "HH": 3},
    }

.. note::

    The
    :py:func:`~vc2_bit_widths.json_serialisations.deserialise_quantisation_matrix`
    Python utility function is provided for unpacking this structure.

Test patterns
`````````````

The ``<test-pattern-specification>`` values follow the same format used by
``vc2-static-filter-analysis`` (see
:ref:`vc2-static-filter-analysis-json-test-patterns`) with some additional
fields::

    <test-pattern-specification> = {
        ...,
        "quantisation_index": <int>,
        "decoded_value": <int>,
        "num_search_iterations": <int>,
    }

These fields give the quantisation index found to produce the most extreme
value for the test pattern, the value it managed to produce and the number of
search iterations taken to reach that test pattern. This information is
provided for informational purposes only.

.. note::

    The
    :py:func:`~vc2_bit_widths.json_serialisations.deserialise_test_pattern_specifications`
    Python utility function is provided for unpacking this structure.

Missing values
``````````````

Only intermediate arrays are included which contain novel values. Arrays which
are just renamings, interleavings and subsamplings of other arrays are omitted.

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

from vc2_bit_widths.patterns import (
    TestPatternSpecification,
    OptimisedTestPatternSpecification,
)

from vc2_bit_widths.json_serialisations import (
    deserialise_signal_bounds,
    serialise_test_pattern_specifications,
    deserialise_test_pattern_specifications,
    serialise_quantisation_matrix,
)

from vc2_bit_widths.helpers import (
    evaluate_filter_bounds,
    quantisation_index_bound,
    optimise_synthesis_test_patterns,
)


def parse_args(args=None):
    parser = ArgumentParser(description="""
        Use a stochastic search to enchance the synthesis test patterns
        produced by vc2-static-filter-analysis so that they produce even more
        extreme values.
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
    synthesis_test_patterns = deserialise_test_pattern_specifications(
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
        "optimised_synthesis_test_patterns": serialise_test_pattern_specifications(
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
