r"""
.. _vc2-static-filter-analysis:

``vc2-static-filter-analysis``
==============================

This command statically analyses a VC-2 filter configuration to determine
mathematical expressions for worst-case signal ranges and generate test
patterns.

Example usage
-------------

A simple 2-level LeGall (5, 3) transform::

    $ vc2-static-filter-analysis \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 2 \
        --output static_analysis.json

A more complex asymmetric transform::

    $ vc2-static-filter-analysis \
        --wavelet-index haar_with_shift \
        --wavelet-index-ho le_gall_5_3 \
        --dwt-depth 1 \
        --dwt-depth-ho 2 \
        --output static_analysis.json

Arguments
---------

The complete set of arguments can be listed using ``--help``

.. program-output:: vc2-static-filter-analysis --help



.. _vc2-static-filter-analysis-json-schema:

JSON file format
----------------

The output of ``vc2-static-filter-analysis`` is a JSON file which has the
following structure::

    {
        "wavelet_index": <int>,
        "wavelet_index_ho": <int>,
        "dwt_depth": <int>,
        "dwt_depth_ho": <int>,
        "analysis_signal_bounds": [<signal-bounds>, ...],
        "synthesis_signal_bounds": [<signal-bounds>, ...],
        "analysis_test_patterns": [<test-pattern-specification>, ...],
        "synthesis_test_patterns": [<test-pattern-specification>, ...],
    }


Signal bounds
`````````````

The ``"analysis_signal_bounds"`` and ``"synthesis_signal_bounds"`` lists define
algebraic expressions for the upper- and lower-bounds for each analysis and
synthesis filter phase (see :ref:`terminology`) as follows::

    <signal-bounds> = {
        "level": <int>,
        "array_name": <string>,
        "phase": [<int>, <int>],
        "lower_bound": <algebraic-expression>,
        "upper_bound": <algebraic-expression>,
    }
    
    <algebraic-expression> = [
        {
            "symbol": <string-or-null>,
            "numer": <string>,
            "denom": <string>,
        },
        ...
    ]

Each ``<algebraic-expression>`` defines an algebraic linear expression. As an
example, the following expression:

.. math::

    \frac{2}{3}a + 5b - 2

Would be represented in JSON form as::

    [
        {"symbol": "a", "numer": "2", "denom": "3"},
        {"symbol": "b", "numer": "5", "denom": "1"},
        {"symbol": null, "numer": "-2", "denom": "1"},
    ]

In the expressions defining the analysis filter signal levels, the following
symbols are used:

    * ``signal_min`` -- The minimum picture signal value (e.g. -512 for 10 bit signals).
    * ``signal_max`` -- The minimum picture signal value (e.g. 511 for 10 bit signals).

In the expressions defining the synthesis filter signal levels, symbols with
the form ``coeff_<level>_<orient>_min`` and ``coeff_<level>_<orient>_min`` are
used. For example ``coeff_1_LL_min`` would mean the minimum value a level-1
'LL' subband value could have.

See also :py:class:`~vc2_bit_widths.linexp.LinExp` for a Python API for working
with these expressions.

.. _vc2-static-filter-analysis-json-test-patterns:

Test patterns
`````````````

The ``"analysis_test_patterns`` and  ``"synthesis_test_patterns`` lists define
test patterns for each analysis and synthesis filter phase like so::

    <test-pattern-specification> = {
        "level": <int>,
        "array_name": <string>,
        "phase": [<int>, <int>],
        "target": [<int>, <int>],
        "target_transition_multiple": [<int>, <int>],
        "pattern": <test-pattern>,
        "pattern_transition_multiple": [<int>, <int>],
    }
    
    <test-pattern> = {
        "dx": <int>,
        "dy": <int>,
        "width": <int>,
        "height": <int>,
        "positive": <string>,
        "mask": <string>,
    }

Test patterns are defined in terms of a collection of pixel polarity values
which indicate which pixels should be set to their maximum level and which
should be set to their minimum. All other pixel values may be set arbitrarily.
For the encoding used by the ``<test-pattern>`` object to encode the pixel
polarities themselves, see
:py:class:`~vc2_bit_widths.json_serialisations.serialise_test_pattern`.

Test patterns must be carefully aligned within a test picture when used. See
:py:class:`~vc2_bit_widths.pattern_generation.TestPatternSpecification` for the
meaning of the relevant fields of ``<test-pattern-specification>``.  .



Missing values
``````````````

Only intermediate arrays are included which contain novel values. Arrays which
are just renamings, interleavings and subsamplings of other arrays are omitted.


Runtime and memory consumption
------------------------------

For typical 'real world' filter configurations, this command should complete
within a few seconds and use a trivial amount of memory.

For larger wavelets (e.g. the fidelity filter) and deeper transforms (e.g.
three or more levels), the runtime and memory requirements can grow
significantly. For example, a 4-level LeGall (5, 3) transform will take on the
order of an hour to analyse. As an especially extreme case, a 4-level Fidelity
wavelet will require around 16 GB of RAM and several weeks to complete.

The ``--verbose`` option provides useful progress information as the static
analysis process proceeds. As a guide, the vast majority of the runtime will be
spent on the synthesis filters due to their far greater number. RAM usage grows
rapidly during the processing of the analysis filters and then grow far more
slowly during synthesis filter analysis.

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
        This command statically analyses a VC-2 filter configuration to
        determine mathematical expressions for worst-case signal ranges and
        generate test patterns. Writes the output to a JSON file.
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
