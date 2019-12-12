r"""
.. _vc2-static-filter-analysis-combine:

``vc2-static-filter-analysis-combine``
======================================

Combine the results of several batched runs of
:ref:`vc2-static-filter-analysis` into a single set of results.

Example usage
-------------

::

    $ vc2-static-filter-analysis \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 2 \
        --num-batches 2 \
        --batch-num 0 \
        --output static_analysis_batch_0.json
    $ vc2-static-filter-analysis \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 2 \
        --num-batches 2 \
        --batch-num 1 \
        --output static_analysis_batch_1.json
    
    $ vc2-static-filter-analysis-combine \
        static_analysis_batch_0.json \
        static_analysis_batch_1.json \
        --output static_analysis.json


Arguments
---------

The complete set of arguments can be listed using ``--help``

.. program-output:: vc2-static-filter-analysis-combine --help

"""

import sys

import json

from itertools import chain

from argparse import ArgumentParser, FileType


def parse_args(args=None):
    parser = ArgumentParser(description="""
        Combine the outputs of a series of batched calls to
        vc2-static-filter-analysis.
    """)
    
    parser.add_argument(
        "inputs",
        nargs="+",
        help="""
            A complete set of matching batched filter analysis outputs.
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
    
    args = parser.parse_args(args)
    
    return args


def interleave(*iterables):
    """
    Interleave the values of a series of iterators.
    """
    iterators = list(map(iter, iterables))
    
    while iterators:
        for iterator in list(iterators):
            try:
                yield next(iterator)
            except StopIteration:
                iterators.remove(iterator)


def main(args=None):
    args = parse_args(args)
    
    out = None
    num_batches = None
    read_batches = {}
    
    # Read all batch data and check it is self consistent
    for filename in args.inputs:
        data = json.load(open(filename))
        
        if "num_batches" not in data:
            sys.stderr.write("Error: Input {} is not the result of a batched run of vc2-static-filter-analysis.\n".format(
                filename,
            ))
            return 1
        
        if out is None:
            # First file
            out = {
                "wavelet_index": data["wavelet_index"],
                "wavelet_index_ho": data["wavelet_index_ho"],
                "dwt_depth": data["dwt_depth"],
                "dwt_depth_ho": data["dwt_depth_ho"],
            }
            num_batches = data["num_batches"]
            read_batches[data["batch_num"]] = data
        else:
            # Subsequent files: verify that they are other batches from the
            # same filter configuration.
            if (
                out["wavelet_index"] != data["wavelet_index"] or
                out["wavelet_index_ho"] != data["wavelet_index_ho"] or
                out["dwt_depth"] != data["dwt_depth"] or
                out["dwt_depth_ho"] != data["dwt_depth_ho"] or
                num_batches != data["num_batches"]
            ):
                sys.stderr.write("Error: Input {} is from an incompatible run of vc2-static-filter-analysis.\n".format(
                    filename,
                ))
                return 2
            if data["batch_num"] in read_batches:
                sys.stderr.write("Error: Input {} is a duplicate of batch {}.\n".format(
                    filename,
                    data["batch_num"],
                ))
                return 3
            
            read_batches[data["batch_num"]] = data
    
    # Verify that all batches have been provided
    missing_batches = set(range(num_batches)) - set(read_batches)
    if missing_batches:
        sys.stderr.write("Error: Missing batch(es) {}.\n".format(
            missing_batches
        ))
        return 4
    
    # Interleave batch data to produce intended ordering
    iterators = {
        name: interleave(*(
            iter(data[name])
            for batch_num, data in sorted(read_batches.items())
        ))
        for name in [
            "analysis_signal_bounds",
            "synthesis_signal_bounds",
            "analysis_test_patterns",
            "synthesis_test_patterns",
        ]
    }
    
    for name, iterator in iterators.items():
        out[name] = list(iterator)
    
    json.dump(out, args.output)
    args.output.write("\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
