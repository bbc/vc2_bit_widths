"""
Utility functions for parsing more complex command line arguments.
"""

import sys

from collections import defaultdict

from vc2_data_tables import WaveletFilters, QUANTISATION_MATRICES



def wavelet_index_or_name(string):
    try:
        return WaveletFilters(int(string))
    except ValueError:
        for index in WaveletFilters:
            if string == index.name:
                return index
        raise ValueError("No matching wavelet name found.")


def parse_quantisation_matrix_argument(
    arguments,
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho
):
    """
    Convert a custom quantisation matrix specification argument list into a
    standard nested dictionary format, or load the default quantisation matrix
    if possible otherwise.
    
    If the argument can't be parsed, calls :py:func:`sys.exit` with 1 and
    prints a message to stderr.
    
    Parameters
    ==========
    arguments : ["level", "orient", "value", ...] or None
        The argument strings from :py;class:`~argparse.ArgumentParser`.
    wavelet_index : int
    wavelet_index_ho : int
    dwt_depth : int
    dwt_depth_ho : int
        The wavelet filter in use (used only to determine the default
        quantisation matrix.
    
    Returns
    =======
    quantisation_matrix : {level: {orient: value, ...}, ...}
    """
    if arguments is None:
        # No custom matrix provided, use default matrix if possible
        quantisation_matrix = QUANTISATION_MATRICES.get((
            wavelet_index,
            wavelet_index_ho,
            dwt_depth,
            dwt_depth_ho,
        ))
        if quantisation_matrix is None:
            sys.stderr.write(
                "No default quantisation matrix is defined for the current "
                "wavelet, a custom quantisation matrix must be provided using "
                "the --custom-quantisation-matrix argument.\n"
            )
            sys.exit(1)
        else:
            return quantisation_matrix
    else:
        if len(arguments) % 3 != 0:
            sys.stderr.write(
                "A multiple of three arguments are required for "
                "--custom-quantisation-matrix\n"
            )
            sys.exit(1)
        
        quantisation_matrix = defaultdict(dict)
        
        i = iter(arguments)
        for level, orient, value in zip(i, i, i):
            try:
                quantisation_matrix[int(level)][orient] = int(value)
            except ValueError:
                sys.stderr.write(
                    "Level and value arguments in "
                    "--custom-quantisation-matrix must be valid integers.\n"
                )
                sys.exit(1)
        
        return dict(quantisation_matrix)  # Convert to regular dict
