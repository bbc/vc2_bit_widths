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

.. note::

    The
    :py:func:`~vc2_bit_widths.json_serialisations.deserialise_signal_bounds`
    Python utility function is provided for unpacking this structure.

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

.. note::

    The
    :py:func:`~vc2_bit_widths.json_serialisations.deserialise_test_pattern_specifications`
    Python utility function is provided for unpacking this structure.

Test patterns are defined in terms of a collection of pixel polarity values
which indicate which pixels should be set to their maximum level and which
should be set to their minimum. All other pixel values may be set arbitrarily.
For the encoding used by the ``<test-pattern>`` object to encode the pixel
polarities themselves, see
:py:class:`~vc2_bit_widths.json_serialisations.serialise_test_pattern`.

Test patterns must be carefully aligned within a test picture when used. See
:py:class:`~vc2_bit_widths.patterns.TestPatternSpecification` for the
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
rapidly during the processing of the analysis filters and then only grow very
slightly during synthesis filter analysis.

Batched/parallel execution
--------------------------

When analysing extremely large filters, it might be useful to split the
analysis process into multiple batches to be executed simultaneously on several
cores or computers.

To run the analysis process in batched mode, the ``--num-batches`` argument
must be given specifying the total number of batches the execution will be
split into. Each batch must also be given a ``--batch-num`` argument specifying
which batch is to be executed. For example, the following commands might be run
on four machines to parallelise the analysis of a 3-level Fidelity filter.

::

    $ vc2-static-filter-analysis \
        --wavelet-index fidelity \
        --depth 3 \
        --num-batches 4 \
        --batch-num 0 \
        --output static_analysis_batch_0.json
    
    $ vc2-static-filter-analysis \
        --wavelet-index fidelity \
        --depth 3 \
        --num-batches 4 \
        --batch-num 1 \
        --output static_analysis_batch_1.json
    
    $ vc2-static-filter-analysis \
        --wavelet-index fidelity \
        --depth 3 \
        --num-batches 4 \
        --batch-num 2 \
        --output static_analysis_batch_2.json
    
    $ vc2-static-filter-analysis \
        --wavelet-index fidelity \
        --depth 3 \
        --num-batches 4 \
        --batch-num 3 \
        --output static_analysis_batch_3.json

Once all four analyses have finished, the results are then combined together
using the :ref:`vc2-static-filter-analysis-combine` utility::

    $ vc2-static-filter-analysis-combine \
        static_analysis_batch_0.json \
        static_analysis_batch_1.json \
        static_analysis_batch_2.json \
        static_analysis_batch_3.json \
        --output static_analysis.json

.. warning::

    Each batch may still require the same amount of RAM as a complete analysis.

.. warning::

    The batching process requires some duplicated processing in each batch.
    Consequently more total CPU time may be required than non-batched
    execution.

.. warning::

    Though batches are intended to take similar amounts of time to execute,
    this is not guaranteed.

"""

import sys

import json

import logging

from argparse import ArgumentParser, FileType

from vc2_data_tables import WaveletFilters

from vc2_bit_widths.patterns import TestPatternSpecification

from vc2_bit_widths.helpers import static_filter_analysis

from vc2_bit_widths.json_serialisations import (
    serialise_signal_bounds,
    serialise_test_pattern_specifications,
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
        "--num-batches", "-B",
        type=int, default=1,
        help="""
            If the analysis is to be performed in a series of smaller batches,
            the number of batches to split it into.
        """
    )
    parser.add_argument(
        "--batch-num", "-b",
        type=int, default=None,
        help="""
            When --num-batches is used, the specific the batch to run.
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
    
    if args.batch_num is None:
        if args.num_batches != 1:
            parser.error("--batch-num/-b must be given when --num-batches/-B is used")
        else:
            args.batch_num = 0
    if args.batch_num >= args.num_batches:
        parser.error("--batch-num/-b must be less than --num-batches/-B")
    
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
        num_batches=args.num_batches,
        batch_num=args.batch_num,
    )
    
    # Serialise
    out = {
        "wavelet_index": int(args.wavelet_index),
        "wavelet_index_ho": int(args.wavelet_index_ho),
        "dwt_depth": args.dwt_depth,
        "dwt_depth_ho": args.dwt_depth_ho,
        "analysis_signal_bounds":
            serialise_signal_bounds(analysis_signal_bounds),
        "synthesis_signal_bounds":
            serialise_signal_bounds(synthesis_signal_bounds),
        "analysis_test_patterns":
            serialise_test_pattern_specifications(TestPatternSpecification, analysis_test_patterns),
        "synthesis_test_patterns":
            serialise_test_pattern_specifications(TestPatternSpecification, synthesis_test_patterns),
    }
    
    # Mark batched results files
    if args.num_batches > 1:
        out["num_batches"] = args.num_batches
        out["batch_num"] = args.batch_num
    
    json.dump(out, args.output)
    args.output.write("\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
