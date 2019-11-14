r"""
.. _vc2-maximum-quantisation-index:

``vc2-maximum-quantisation-index``
==================================

Compute the maximum quantisation index which is useful for a particular
VC-2 codec configuration.

Using the same approach as the ``vc2-bit-widths-table`` command, this command
works out the most extreme values which a VC-2 quantiser might encounter (given
certain assumptions about the behaviour of the encoder design, see
:ref:`caveats`). Using this information, it is possible to find the smallest
quantisation index sufficient to quantise all transform coefficients to zero.

Though the VC-2 standard does not rule out larger quantisation indices being
used, there is no reason for a sensible encoder implementation to use such a
any larger quantisation index.


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
    
    $ vc2-maximum-quantisation-index \
        static_analysis.json \
        --picture-bit-width 10 \
        --custom-quantisation-matrix \
            0 LL 1 \
            1 HL 2   1 LH 0   1 HH 4 \
            2 HL 1   2 LH 3   2 HH 3
    59


Arguments
---------

The complete set of arguments can be listed using ``--help``

.. program-output:: vc2-maximum-quantisation-index --help

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
