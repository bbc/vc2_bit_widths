"""
:py:mod:`vc2_bit_widths.bundle`: Analysis file bundling
=======================================================

Pre-computed filter analyses (e.g. from multiple runs of
:ref:`vc2-static-filter-analysis`) and optimised test patterns (e.g. from runs
of :ref:`vc2-optimise-synthesis-test-patterns`) may be packed into a 'bundle'
file along with an index of available patterns.

The bundle file is a compressed zip file and offers significant space savings
over storing the analyses uncompressed.

The provision of an index file also enables more efficient lookup of files by
codec parameters than reading every file.

.. autofunction:: bundle_create_from_serialised_dicts

.. autofunction:: bundle_index

.. autofunction:: bundle_get_static_filter_analysis

.. autofunction:: bundle_get_optimised_synthesis_test_patterns

"""

import json

from zipfile import ZipFile, ZIP_DEFLATED

from vc2_bit_widths.patterns import (
    TestPatternSpecification,
    OptimisedTestPatternSpecification,
)

from vc2_bit_widths.json_serialisations import (
    deserialise_quantisation_matrix,
    deserialise_signal_bounds,
    deserialise_test_pattern_specifications,
)

__all__ = [
    "bundle_create_from_serialised_dicts",
    "bundle_index",
    "bundle_get_static_filter_analysis",
    "bundle_get_optimised_synthesis_test_patterns",
]


def bundle_create_from_serialised_dicts(
    file_or_filename,
    serialised_static_filter_analyses=(),
    serialised_optimised_synthesis_test_patterns=(),
):
    """
    Create a bundle file containing the supplied pre-serialised filter analyses
    and test patterns.
    
    Parameters
    ==========
    file_or_filename : file-like or str
        The bundle to write to.
    serialised_static_filter_analyses : iterable
        An iterable of dictionaries containing filter analyses with all entries
        serialised as produced by :ref:`vc2-static-filter-analysis`. Must be
        free of duplicates.
    serialised_optimised_synthesis_test_patterns : iterable
        An iterable of dictionaries containing filter analyses with all entries
        serialised as expected by :ref:`vc2-optimise-synthesis-test-patterns`.
        Must be free of duplicates.
    """
    with ZipFile(file_or_filename, "w", compression=ZIP_DEFLATED) as bundle:
        index = {
            "static_filter_analyses": [],
            "optimised_synthesis_test_patterns": [],
        }
        
        for i, json_data in enumerate(serialised_static_filter_analyses):
            filename = "static_filter_analysis_{}.json".format(i)
            index["static_filter_analyses"].append({
                "wavelet_index": json_data["wavelet_index"],
                "wavelet_index_ho": json_data["wavelet_index_ho"],
                "dwt_depth": json_data["dwt_depth"],
                "dwt_depth_ho": json_data["dwt_depth_ho"],
                "filename": filename,
            })
            
            bundle.writestr(filename, json.dumps(json_data))
        
        for i, json_data in enumerate(serialised_optimised_synthesis_test_patterns):
            filename = "optimised_synthesis_test_patterns_{}.json".format(i)
            index["optimised_synthesis_test_patterns"].append({
                "wavelet_index": json_data["wavelet_index"],
                "wavelet_index_ho": json_data["wavelet_index_ho"],
                "dwt_depth": json_data["dwt_depth"],
                "dwt_depth_ho": json_data["dwt_depth_ho"],
                "quantisation_matrix": json_data["quantisation_matrix"],
                "picture_bit_width": json_data["picture_bit_width"],
                "filename": filename,
            })
            
            bundle.writestr(filename, json.dumps(json_data))
        
        bundle.writestr("index.json", json.dumps(index))


def _deserialise_index(index):
    """
    (Internal function) Deserialises the entries in an index. Makes changes
    in-place, but also returns the mutated index for convenience.
    """
    
    for entry in index["optimised_synthesis_test_patterns"]:
        entry["quantisation_matrix"] = deserialise_quantisation_matrix(
            entry["quantisation_matrix"]
        )
    
    return index


def bundle_index(file_or_filename):
    """
    Get the index from a bundle file.
    
    Parameters
    ==========
    file_or_filename : file-like or str
        The bundle to read.
    
    Returns
    =======
    index : dict
        Returns a dictionary containing two lists under the keys
        "static_filter_analyses" and "optimised_synthesis_test_patterns".
        Both lists contain dictionaries containing the following fields:
        
        * "wavelet_index" (int)
        * "wavelet_index_ho" (int)
        * "dwt_depth" (int)
        * "dwt_depth_ho" (int)
        * "filename": The file in the bundle containing the corresponding
          analysis or test patterns.
        
        The "optimised_synthesis_test_patterns" list dicts also contain the
        following additional entries:
        
        * "quantisation_matrix" (``{level: {orient: value, ...}, ...}``)
        * "picture_bit_width" (int)
    """
    with ZipFile(file_or_filename, "r") as bundle:
        with bundle.open("index.json") as f:
            index = _deserialise_index(json.load(f))
    
    return index


def bundle_get_serialised_static_filter_analysis(
    file_or_filename,
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
):
    """
    Read a still serialised static filter analysis for a particular filter
    configuration from a bundle. Raises :py:exc:`KeyError` if no matching
    analysis is present in the bundle.
    
    Parameters
    ==========
    file_or_filename : file-like or str
        The bundle to read.
    wavelet_index : int
    wavelet_index_ho : int
    dwt_depth : int
    dwt_depth_ho : int
    
    Returns
    =======
    json_data : dict
        The serialised static filter analysis. See also
        :py:func:`bundle_get_static_filter_analysis`.
    """
    with ZipFile(file_or_filename, "r") as bundle:
        with bundle.open("index.json") as f:
            index = _deserialise_index(json.load(f))
        
        for entry in index["static_filter_analyses"]:
            if (
                entry["wavelet_index"] == wavelet_index and
                entry["wavelet_index_ho"] == wavelet_index_ho and
                entry["dwt_depth"] == dwt_depth and
                entry["dwt_depth_ho"] == dwt_depth_ho
            ):
                with bundle.open(entry["filename"]) as f:
                    return json.load(f)
    
    raise KeyError("No matching static filter analysis found.")


def bundle_get_static_filter_analysis(
    file_or_filename,
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
):
    """
    Read a static filter analysis for a particular filter configuration from
    a bundle. Raises :py:exc:`KeyError` if no matching analysis is present in
    the bundle.
    
    Parameters
    ==========
    file_or_filename : file-like or str
        The bundle to read.
    wavelet_index : int
    wavelet_index_ho : int
    dwt_depth : int
    dwt_depth_ho : int
    
    Returns
    =======
    analysis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
    synthesis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
    analysis_test_patterns: {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
    synthesis_test_patterns: {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
        See :py:func:`vc2_bit_widths.helpers.static_filter_analysis`.
    """
    analysis = bundle_get_serialised_static_filter_analysis(
        file_or_filename,
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
    )
    
    return (
        deserialise_signal_bounds(
            analysis["analysis_signal_bounds"]
        ),
        deserialise_signal_bounds(
            analysis["synthesis_signal_bounds"]
        ),
        deserialise_test_pattern_specifications(
            TestPatternSpecification,
            analysis["analysis_test_patterns"]
        ),
        deserialise_test_pattern_specifications(
            TestPatternSpecification,
            analysis["synthesis_test_patterns"]
        ),
    )


def bundle_get_serialised_optimised_synthesis_test_patterns(
    file_or_filename,
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    quantisation_matrix,
    picture_bit_width,
):
    """
    Read a still serialised static filter analysis for a particular filter
    configuration from a bundle. Raises :py:exc:`KeyError` if no matching
    analysis is present in the bundle.
    
    Parameters
    ==========
    file_or_filename : file-like or str
        The bundle to read.
    wavelet_index : int
    wavelet_index_ho : int
    dwt_depth : int
    dwt_depth_ho : int
    quantisation_matrix : {level: {orient: value, ...}, ...}
    picture_bit_width : int
    
    Returns
    =======
    json_data
        The serialised optimised synthesis filter. See also
        :py:func:`bundle_get_static_filter_analysis`.
    """
    with ZipFile(file_or_filename, "r") as bundle:
        with bundle.open("index.json") as f:
            index = _deserialise_index(json.load(f))
        
        for entry in index["optimised_synthesis_test_patterns"]:
            if (
                entry["wavelet_index"] == wavelet_index and
                entry["wavelet_index_ho"] == wavelet_index_ho and
                entry["dwt_depth"] == dwt_depth and
                entry["dwt_depth_ho"] == dwt_depth_ho and
                entry["quantisation_matrix"] == quantisation_matrix and
                entry["picture_bit_width"] == picture_bit_width
            ):
                with bundle.open(entry["filename"]) as f:
                    return json.load(f)
    
    raise KeyError("No matching optimised synthesis test patterns found.")


def bundle_get_optimised_synthesis_test_patterns(
    file_or_filename,
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    quantisation_matrix,
    picture_bit_width,
):
    """
    Read a static filter analysis for a particular filter configuration from
    a bundle. Raises :py:exc:`KeyError` if no matching analysis is present in
    the bundle.
    
    Parameters
    ==========
    file_or_filename : file-like or str
        The bundle to read.
    wavelet_index : int
    wavelet_index_ho : int
    dwt_depth : int
    dwt_depth_ho : int
    quantisation_matrix : {level: {orient: value, ...}, ...}
    picture_bit_width : int
    
    Returns
    =======
    optimised_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.OptimisedTestPatternSpecification`, ...}
        See :py:func:`vc2_bit_widths.helpers.optimise_synthesis_test_patterns`.
    """
    patterns = bundle_get_serialised_optimised_synthesis_test_patterns(
        file_or_filename,
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        picture_bit_width,
    )
    
    return deserialise_test_pattern_specifications(
        OptimisedTestPatternSpecification,
        patterns["optimised_synthesis_test_patterns"]
    )
