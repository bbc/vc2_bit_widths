r"""
.. _vc2-bundle:

``vc2-bundle``
==============

Create, or query a compressed bundle containing a set of static filter analyses
(from :ref:`vc2-static-filter-analysis`) and optimised synthesis test patterns
(from :ref:`vc2-optimise-synthesis-test-patterns`).

Example usage
-------------

Creating a bundle

::

    $ # Create some files
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
        --output optimised_patterns.json
    
    $ # Bundle the files together
    $ vc2-bundle \
        create \
        bundle.zip \
        static_analysis.json \
        optimised_patterns.json \
        # ...

Listing bundle contents

::

    $ vc2-bundle list bundle.zip
    Static filter analyses
    ======================
    
    0.
        * wavelet_index: le_gall_5_3
        * wavelet_index_ho: le_gall_5_3
        * dwt_depth: 2
        * dwt_depth_ho: 0
    
    Optimised synthesis test patterns
    =================================
    
    0.
        * wavelet_index: le_gall_5_3
        * wavelet_index_ho: le_gall_5_3
        * dwt_depth: 2
        * dwt_depth_ho: 0
        * picture_bit_width: 10
        * custom_quantisation_matrix:
          {
              0: {'LL': 1},
              1: {'HL': 2, 'LH': 0, 'HH' 4},
              2: {'HL': 1, 'LH': 3, 'HH' 3},
          }

Extracting files from a bundle

::

    $ # Extract a static analysis file from the bundle
    $ vc2-bundle \
        extract-static-filter-analysis \
        bundle.zip \
        extracted.json \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 2
    
    $ # Extract a set of optimised synthesis test patterns from the bundle
    $ vc2-bundle \
        extract-optimised-synthesis-test-pattern \
        bundle.zip \
        extracted.json \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 2
        --picture-bit-width 10 \
        --custom-quantisation-matrix \
            0 LL 1 \
            1 HL 2   1 LH 0   1 HH 4 \
            2 HL 1   2 LH 3   2 HH 3 \


Arguments
---------

The complete set of arguments can be listed using ``--help`` for each
subcommand.

.. program-output:: vc2-bundle --help

.. program-output:: vc2-bundle create --help

.. program-output:: vc2-bundle list --help

.. program-output:: vc2-bundle extract-static-filter-analysis --help

.. program-output:: vc2-bundle extract-optimised-synthesis-test-patterns --help

"""

import sys

import json

from argparse import ArgumentParser, FileType

from vc2_data_tables import WaveletFilters

from vc2_bit_widths.scripts.argument_parsers import (
    wavelet_index_or_name,
    parse_quantisation_matrix_argument,
)

from vc2_bit_widths.bundle import (
    bundle_create_from_serialised_dicts,
    bundle_index,
    bundle_get_serialised_static_filter_analysis,
    bundle_get_serialised_optimised_synthesis_test_patterns,
)


def parse_args(args=None):
    parser = ArgumentParser(description="""
        Create, or query a compressed bundle containing a set of static filter
        analyses (from vc2-static-filter-analysis) and optimised synthesis test
        patterns (from vc2-optimise-synthesis-test-patterns).
    """)
    
    subparsers = parser.add_subparsers()
    
    parser.add_argument(
        "--verbose", "-v", default=0, action="count",
        help="""
            Show more detailed status information during execution.
        """,
    )
    
    parser.set_defaults(action=None)
    
    create_parser = subparsers.add_parser(
        "create",
        help="""
            Create a bundle file.
        """,
    )
    create_parser.set_defaults(action="create")
    
    create_parser.add_argument(
        "bundle_file",
        type=FileType("wb"),
        help="""
            Filename for the bundle file to create/overwrite.
        """,
    )
    
    create_parser.add_argument(
        "--static-filter-analysis", "-s",
        type=FileType("rb"),
        nargs="+", metavar="JSON_FILE",
        action="append", default=[],
        help="""
            Filenames of the JSON files containing static filter analyses
            produced by vc2-static-filter-analysis.
        """,
    )
    create_parser.add_argument(
        "--optimised-synthesis-test-patterns", "-o",
        type=FileType("rb"),
        nargs="+", metavar="JSON_FILE",
        action="append", default=[],
        help="""
            Filenames of the JSON files containing optimised synthesis test
            patterns produced by vc2-optimise-synthesis-test-patterns.
        """,
    )
    
    ls_parser = subparsers.add_parser(
        "list",
        help="""
            List the contents of a bundle file.
        """,
    )
    ls_parser.set_defaults(action="list")
    
    ls_parser.add_argument(
        "bundle_file",
        type=FileType("rb"),
        help="""
            Filename for the bundle file to list.
        """,
    )
    
    extract_sfa_parser = subparsers.add_parser(
        "extract-static-filter-analysis",
        help="""
            Extract a static filter analysis from the bundle.
        """,
    )
    extract_sfa_parser.set_defaults(action="extract-static-filter-analysis")
    
    extract_ostp_parser = subparsers.add_parser(
        "extract-optimised-synthesis-test-patterns",
        help="""
            Extract a set of optimised synthesis test patterns from the bundle.
        """,
    )
    extract_ostp_parser.set_defaults(action="extract-optimised-synthesis-test-patterns")
    
    # Arguments shared by both extract subcommands
    for sub_parser in [extract_sfa_parser, extract_ostp_parser]:
        sub_parser.add_argument(
            "bundle_file",
            type=FileType("rb"),
            help="""
                Filename for the bundle file to query.
            """,
        )
        sub_parser.add_argument(
            "--output", "-o",
            type=FileType("w"), default=sys.stdout,
            help="""
                Filename for the extracted JSON file. Defaults to stdout.
            """,
        )
        sub_parser.add_argument(
            "--wavelet-index", "-w",
            type=wavelet_index_or_name, required=True,
            help="""
                The VC-2 wavelet index for the wavelet transform. One of: {}.
            """.format(", ".join(
                "{} or {}".format(int(index), index.name)
                for index in WaveletFilters
            )),
        )
        sub_parser.add_argument(
            "--wavelet-index-ho", "-W",
            type=wavelet_index_or_name,
            help="""
                The VC-2 wavelet index for the horizontal parts of the wavelet
                transform. If not specified, assumed to be the same as
                --wavelet-index/-w.
            """,
        )
        sub_parser.add_argument(
            "--dwt-depth", "-d",
            type=int, default=0,
            help="""
                The VC-2 transform depth. Defaults to 0 if not specified.
            """
        )
        sub_parser.add_argument(
            "--dwt-depth-ho", "-D",
            type=int, default=0,
            help="""
                The VC-2 horizontal-only transform depth. Defaults to 0 if not
                specified.
            """
        )
    
    # Add optimised synthesis test pattern specific arguments
    extract_ostp_parser.add_argument(
        "--picture-bit-width", "-b",
        type=int, required=True,
        help="""
            The number of bits in the picture signal.
        """,
    )
    extract_ostp_parser.add_argument(
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
    
    args = parser.parse_args(args)
    
    if args.action is None:
        parser.error("Expected a command.")
    
    # Flatten create arguments
    if args.action == "create":
        args.static_filter_analysis = [
            entry
            for lst in args.static_filter_analysis
            for entry in lst
        ]
        args.optimised_synthesis_test_patterns = [
            entry
            for lst in args.optimised_synthesis_test_patterns
            for entry in lst
        ]
    
    
    # Auto-fill horizontal only wavelet
    if args.action in (
        "extract-static-filter-analysis",
        "extract-optimised-synthesis-test-patterns",
    ):
        if args.wavelet_index_ho is None:
            args.wavelet_index_ho = args.wavelet_index
    
    if args.action == "extract-optimised-synthesis-test-patterns":
        args.quantisation_matrix = parse_quantisation_matrix_argument(
            args.custom_quantisation_matrix,
            args.wavelet_index,
            args.wavelet_index_ho,
            args.dwt_depth,
            args.dwt_depth_ho,
        )
    
    return args


def main_create(args):
    bundle_create_from_serialised_dicts(
        args.bundle_file,
        serialised_static_filter_analyses=(
            json.load(f)
            for f in args.static_filter_analysis
        ),
        serialised_optimised_synthesis_test_patterns=(
            json.load(f)
            for f in args.optimised_synthesis_test_patterns
        ),
    )
    
    return 0


def main_list(args):
    index = bundle_index(
        args.bundle_file,
    )
    
    printed_something = False
    
    if index["static_filter_analyses"]:
        print("Static filter analyses")
        print("======================")
        print("")
        for i, entry in enumerate(index["static_filter_analyses"]):
            print("{}.".format(i))
            print("    * wavelet_index: {} ({})".format(
                WaveletFilters(entry["wavelet_index"]).name,
                entry["wavelet_index"],
            ))
            print("    * wavelet_index_ho: {} ({})".format(
                WaveletFilters(entry["wavelet_index_ho"]).name,
                entry["wavelet_index_ho"],
            ))
            print("    * dwt_depth: {}".format(entry["dwt_depth"]))
            print("    * dwt_depth_ho: {}".format(entry["dwt_depth_ho"]))
        
        printed_something = True
    
    if index["optimised_synthesis_test_patterns"]:
        if printed_something:
            print("")
            print("")
        print("Optimised synthesis test patterns")
        print("=================================")
        print("")
        for i, entry in enumerate(index["optimised_synthesis_test_patterns"]):
            print("{}.".format(i))
            print("    * wavelet_index: {} ({})".format(
                WaveletFilters(entry["wavelet_index"]).name,
                entry["wavelet_index"],
            ))
            print("    * wavelet_index_ho: {} ({})".format(
                WaveletFilters(entry["wavelet_index_ho"]).name,
                entry["wavelet_index_ho"],
            ))
            print("    * dwt_depth: {}".format(entry["dwt_depth"]))
            print("    * dwt_depth_ho: {}".format(entry["dwt_depth_ho"]))
            print("    * picture_bit_width: {}".format(
                entry["picture_bit_width"],
            ))
            print("    * quantisation_matrix: {")
            for level, orients in sorted(entry["quantisation_matrix"].items()):
                print("          {}: {{{}}},".format(
                    level,
                    ", ".join(
                        "'{}': {}".format(orient, value)
                        for orient, value in sorted(orients.items())
                    ),
                ))
            print("      }")
        
        printed_something = True
    
    if not printed_something:
        print("Bundle is empty.")
    
    return 0


def main_extract_static_filter_analysis(args):
    try:
        analysis = bundle_get_serialised_static_filter_analysis(
            args.bundle_file,
            args.wavelet_index,
            args.wavelet_index_ho,
            args.dwt_depth,
            args.dwt_depth_ho,
        )
        json.dump(analysis, args.output)
        args.output.write("\n")
        return 0
    except KeyError:
        sys.stderr.write("No matching static filter analysis found in bundle.\n")
        return 1


def main_extract_optimised_synthesis_test_patterns(args):
    try:
        analysis = bundle_get_serialised_optimised_synthesis_test_patterns(
            args.bundle_file,
            args.wavelet_index,
            args.wavelet_index_ho,
            args.dwt_depth,
            args.dwt_depth_ho,
            args.quantisation_matrix,
            args.picture_bit_width,
        )
        json.dump(analysis, args.output)
        args.output.write("\n")
        return 0
    except KeyError:
        sys.stderr.write("No matching optimised synthesis test patterns found in bundle.\n")
        return 1


def main(args=None):
    args = parse_args(args)
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if args.action == "create":
        return main_create(args)
    elif args.action == "list":
        return main_list(args)
    elif args.action == "extract-static-filter-analysis":
        return main_extract_static_filter_analysis(args)
    elif args.action == "extract-optimised-synthesis-test-patterns":
        return main_extract_optimised_synthesis_test_patterns(args)
    
    # Unreachable...
    return 99


if __name__ == "__main__":
    import sys
    sys.exit(main())
