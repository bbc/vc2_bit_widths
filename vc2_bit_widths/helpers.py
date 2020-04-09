"""
:py:mod:`vc2_bit_widths.helpers`: Helper Functions
==================================================

The :py:mod:`vc2_bit_widths.helpers` module provides a set of high-level
utility functions implementing the key processes involved in bit-width analysis
and test picture production for VC-2 codecs. These functions are used in much
the same way as the command line interfaces, so refer to :ref:`usage-overview`
for a more general introduction.

Many of the functions in this module can take a significant amount of time to
execute for very large filters. Status information is reported using Python's
built-in :py:mod:`logging` library and may be made visible using::

    >>> import logging
    >>> logging.basicConfig(level=logging.INFO)


Static filter analysis
----------------------

.. autofunction:: static_filter_analysis


Calculating bit-width requirements
----------------------------------

.. autofunction:: evaluate_filter_bounds

.. autofunction:: evaluate_test_pattern_outputs

Bounding quantisation indices
-----------------------------

.. autofunction:: quantisation_index_bound

Optimising synthesis test patterns
----------------------------------

.. autofunction:: optimise_synthesis_test_patterns

Generating test pictures
------------------------

.. autofunction:: generate_test_pictures

.. autoclass:: AnalysisPicture
    :no-members:

.. autoclass:: SynthesisPicture
    :no-members:

.. autoclass:: TestPoint
    :no-members:

"""

from collections import OrderedDict, defaultdict, namedtuple

import logging

import numpy as np

from vc2_data_tables import LIFTING_FILTERS

from vc2_bit_widths.infinite_arrays import (
    SymbolArray,
    SymbolicPeriodicCachingArray,
)

from vc2_bit_widths.vc2_filters import (
    make_symbol_coeff_arrays,
    make_variable_coeff_arrays,
    analysis_transform,
    synthesis_transform,
    add_missing_analysis_values,
    add_missing_synthesis_values,
)

from vc2_bit_widths.signal_bounds import (
    analysis_filter_bounds,
    synthesis_filter_bounds,
    evaluate_analysis_filter_bounds,
    evaluate_synthesis_filter_bounds,
    signed_integer_range,
    unsigned_integer_range,
)

from vc2_bit_widths.patterns import (
    invert_test_pattern_specification,
    TestPatternSpecification,
    OptimisedTestPatternSpecification,
)

from vc2_bit_widths.pattern_generation import (
    make_analysis_maximising_pattern,
    make_synthesis_maximising_pattern,
)

from vc2_bit_widths.pattern_evaluation import (
    evaluate_analysis_test_pattern_output,
    evaluate_synthesis_test_pattern_output,
)

from vc2_bit_widths.pattern_optimisation import (
    optimise_synthesis_maximising_test_pattern,
)

from vc2_bit_widths.quantisation import (
    maximum_dequantised_magnitude,
    maximum_useful_quantisation_index,
)

from vc2_bit_widths.picture_packing import pack_test_patterns


logger = logging.getLogger(__name__)


def static_filter_analysis(
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    num_batches=1,
    batch_num=0,
):
    r"""
    Performs a complete static analysis of a VC-2 filter configuration,
    computing theoretical upper- and lower-bounds for signal values (see
    :ref:`theory-affine-arithmetic`) and heuristic test patterns (see
    :ref:`theory-test-patterns`) for all intermediate and final analysis and
    synthesis filter values.
    
    Parameters
    ==========
    wavelet_index : :py:class:`vc2_data_tables.WaveletFilters` or int
    wavelet_index_ho : :py:class:`vc2_data_tables.WaveletFilters` or int
    dwt_depth : int
    dwt_depth_ho : int
        The filter parameters.
    
    num_batches : int
    batch_num : int
        Though for most filters this function runs either instantaneously or at
        worst in the space of a couple of hours, unusually large filters can
        take an extremely long time to run. For example, a 4-level Fidelity
        transform may take around a month to evaluate.
        
        These arguments may be used to split this job into separate batches
        which may be computed separately (and in parallel) and later combined.
        For example, setting ``num_batches`` to 3 results in only analysing
        every third filter phase. The ``batch_num`` parameter should then be
        set to either 0, 1 or 2 to specify which third.
        
        The skipped phases are simply omitted from the returned dictionaries.
        The dictionaries returned for each batch should be unified to produce
        the complete analysis.
    
    Returns
    =======
    analysis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
    synthesis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
        Expressions defining the upper and lower bounds for all intermediate
        and final analysis and synthesis filter values.
        
        The keys of the returned dictionaries give the level, array name and
        filter phase for which each pair of bounds corresponds (see
        :ref:`terminology`). The naming
        conventions used are those defined by
        :py:func:`vc2_bit_widths.vc2_filters.analysis_transform` and
        :py:func:`vc2_bit_widths.vc2_filters.synthesis_transform`. Arrays which
        are just interleavings, subsamplings or renamings of other arrays are
        omitted.
        
        The lower and upper bounds are given algebraically as
        :py:class:`~vc2_bit_widths.linexp.LinExp`\ s.
        
        For the analysis filter bounds, the expressions are defined in terms of
        the variables ``LinExp("signal_min")`` and ``LinExp("signal_max")``.
        These should be substituted for the minimum and maximum picture signal
        level to find the upper and lower bounds for a particular picture bit
        width.
        
        For the synthesis filter bounds, the expressions are defined in terms
        of variables of the form ``LinExp("coeff_LEVEL_ORIENT_min")`` and
        ``LinExp("coeff_LEVEL_ORIENT_max")`` which give lower and upper bounds
        for the transform coefficients with the named level and orientation.
        
        The :py:func:`~vc2_bit_widths.helpers.evaluate_filter_bounds` function
        may be used to substitute concrete values into these expressions for a
        particular picture bit width.
        
    analysis_test_patterns: {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
    synthesis_test_patterns: {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
        Heuristic test patterns which are designed to maximise a particular
        intermediate or final filter value. For a minimising test pattern,
        invert the polarities of the pixels.
        
        The keys of the returned dictionaries give the level, array name and
        filter phase for which each set of bounds corresponds (see
        :ref:`terminology`). Arrays which are just interleavings, subsamplings
        or renamings of other arrays are omitted.
    """
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    
    # Create the algebraic representation of the analysis transform
    picture_array = SymbolArray(2)
    analysis_coeff_arrays, intermediate_analysis_arrays = analysis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        picture_array,
    )
    
    # Count the total number of arrays for use in logging messages
    num_arrays = sum(
        array.period[0] * array.period[1]
        for array in intermediate_analysis_arrays.values()
        if not array.nop
    )
    array_num = 0
    
    # Compute bounds/test pattern for every intermediate/output analysis value
    analysis_signal_bounds = OrderedDict()
    analysis_test_patterns = OrderedDict()
    for (level, array_name), target_array in intermediate_analysis_arrays.items():
        # Skip arrays which are just views of other arrays
        if target_array.nop:
            continue
        
        for x in range(target_array.period[0]):
            for y in range(target_array.period[1]):
                array_num += 1
                if (array_num-1) % num_batches != batch_num:
                    continue
                
                logger.info(
                    "Analysing analysis filter %d of %d (level %d, %s[%d, %d])",
                    array_num,
                    num_arrays,
                    level,
                    array_name,
                    x,
                    y,
                )
                
                # Compute signal bounds
                analysis_signal_bounds[(level, array_name, x, y)] = analysis_filter_bounds(
                    target_array[x, y]
                )
                
                # Generate test pattern
                analysis_test_patterns[(level, array_name, x, y)] = make_analysis_maximising_pattern(
                    picture_array,
                    target_array,
                    x, y,
                )
    
    # Create the algebraic representation of the synthesis transform
    coeff_arrays = make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho)
    synthesis_output_array, intermediate_synthesis_arrays = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        coeff_arrays,
    )
    
    # Create a view of the analysis coefficient arrays which avoids recomputing
    # already-known analysis filter phases
    cached_analysis_coeff_arrays = {
        level: {
            orient: SymbolicPeriodicCachingArray(array, picture_array)
            for orient, array in orients.items()
        }
        for level, orients in analysis_coeff_arrays.items()
    }
    
    
    # Count the total number of arrays for use in logging messages
    num_arrays = sum(
        array.period[0] * array.period[1]
        for array in intermediate_synthesis_arrays.values()
        if not array.nop
    )
    array_num = 0
    
    # Compute bounds/test pattern for every intermediate/output analysis value
    synthesis_signal_bounds = OrderedDict()
    synthesis_test_patterns = OrderedDict()
    for (level, array_name), target_array in intermediate_synthesis_arrays.items():
        # Skip arrays which are just views of other arrays
        if target_array.nop:
            continue
        
        for x in range(target_array.period[0]):
            for y in range(target_array.period[1]):
                array_num += 1
                if (array_num-1) % num_batches != batch_num:
                    continue
                
                logger.info(
                    "Analysing synthesis filter %d of %d (level %d, %s[%d, %d])",
                    array_num,
                    num_arrays,
                    level,
                    array_name,
                    x,
                    y,
                )
                
                # Compute signal bounds
                synthesis_signal_bounds[(level, array_name, x, y)] = synthesis_filter_bounds(
                    target_array[x, y]
                )
                
                # Compute test pattern
                synthesis_test_patterns[(level, array_name, x, y)] = make_synthesis_maximising_pattern(
                    picture_array,
                    cached_analysis_coeff_arrays,
                    target_array,
                    synthesis_output_array,
                    x, y,
                )
                
                # For extremely large filters, a noteworthy amount of overall
                # RAM can be saved by not caching synthesis filters. These
                # filters generally don't benefit much in terms of runtime from
                # caching so this has essentially no impact on runtime.
                for a in intermediate_synthesis_arrays.values():
                    a.clear_cache()
    
    return (
        analysis_signal_bounds,
        synthesis_signal_bounds,
        analysis_test_patterns,
        synthesis_test_patterns,
    )


def evaluate_filter_bounds(
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    analysis_signal_bounds,
    synthesis_signal_bounds,
    picture_bit_width,
):
    r"""
    Evaluate all analysis and synthesis filter signal bounds expressions for a
    given picture bit width, giving concrete signal ranges.
    
    Parameters
    ==========
    wavelet_index : :py:class:`vc2_data_tables.WaveletFilters` or int
    wavelet_index_ho : :py:class:`vc2_data_tables.WaveletFilters` or int
    dwt_depth : int
    dwt_depth_ho : int
        The filter parameters.
    analysis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
    synthesis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
        The outputs of :py:func:`static_filter_analysis`.
    picture_bit_width : int
        The number of bits in the input pictures.
    
    Returns
    =======
    concrete_analysis_signal_bounds : {(level, array_name, x, y): (lower_bound, upper_bound), ...}
    concrete_synthesis_signal_bounds : {(level, array_name, x, y): (lower_bound, upper_bound), ...}
        The concrete, integer signal bounds for all analysis and signal filters
        given a ``picture_bit_width``-bit input picture.
        
        Includes values for *all* arrays and phases, even if
        array interleavings/subsamplings/renamings are omitted in the input
        arguments.
    """
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    
    # Evaluate analysis bounds
    concrete_analysis_signal_bounds = OrderedDict(
        (key, evaluate_analysis_filter_bounds(
            lower_exp,
            upper_exp,
            picture_bit_width,
        ))
        for key, (lower_exp, upper_exp) in analysis_signal_bounds.items()
    )
    
    # Re-add interleaved/renamed entries
    concrete_analysis_signal_bounds = add_missing_analysis_values(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        concrete_analysis_signal_bounds,
    )
    
    # Inflate analysis output bounds to account for worst-case quantisation of
    # transform coefficients in input to synthesis stages.
    #
    # NB: Intermediate arrays use (1, "L")/(1, "LL") for the DC band while the
    # general VC-2 convention is to use (0, "L")/(0, "LL") and so the level
    # number is tweaked in the dictionary below.
    coeff_bounds = OrderedDict(
        (
            (
                level if orient not in ("L", "LL") else 0,
                orient,
            ),
            (
                maximum_dequantised_magnitude(lower_bound),
                maximum_dequantised_magnitude(upper_bound),
            ),
        )
        for (level, orient, _, _), (lower_bound, upper_bound)
        in concrete_analysis_signal_bounds.items()
    )
    
    # Evaluate synthesis bounds
    concrete_synthesis_signal_bounds = OrderedDict(
        (key, evaluate_synthesis_filter_bounds(
            lower_exp,
            upper_exp,
            coeff_bounds,
        ))
        for key, (lower_exp, upper_exp) in synthesis_signal_bounds.items()
    )
    
    # Re-add interleaved/renamed entries
    concrete_synthesis_signal_bounds = add_missing_synthesis_values(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        concrete_synthesis_signal_bounds,
    )
    
    return concrete_analysis_signal_bounds, concrete_synthesis_signal_bounds


def quantisation_index_bound(concrete_analysis_signal_bounds, quantisation_matrix):
    """
    Find the largest quantisation index which could usefully be used given the
    supplied analysis signal bounds.
    
    Parameters
    ==========
    concrete_analysis_signal_bounds : {(level, orient, x, y): (lower_bound, upper_bound), ...}
        Concrete analysis filter bounds as produced by, e.g.,
        :py:func:`evaluate_filter_bounds`.
    quantisation_matrix : {level: {orient: value, ...}, ...}
        The quantisation matrix in use.
    
    Returns
    =======
    max_quantisation_index : int
        The upper bound for the quantisation indices sensibly used by an
        encoder. This value will be the smallest quantisation index which will
        quantise all possible transform coefficients to zero.
    """
    max_qi = 0
    for level in quantisation_matrix:
        for orient in quantisation_matrix[level]:
            # NB: Intermediate arrays use (1, "L")/(1, "LL") for the DC band
            # while the general VC-2 convention is to use (0, "L")/(0, "LL")
            # and so the level number is tweaked in the dictionary below.
            lower_bound, upper_bound = concrete_analysis_signal_bounds[
                (level or 1, orient, 0, 0)
            ]
            
            value_max_qi = max(
                maximum_useful_quantisation_index(lower_bound),
                maximum_useful_quantisation_index(upper_bound),
            )
            value_max_qi += quantisation_matrix[level][orient]
            
            max_qi = max(max_qi, value_max_qi)

    return max_qi


def optimise_synthesis_test_patterns(
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    quantisation_matrix,
    picture_bit_width,
    synthesis_test_patterns,
    max_quantisation_index,
    random_state,
    number_of_searches,
    terminate_early,
    added_corruption_rate,
    removed_corruption_rate,
    base_iterations,
    added_iterations_per_improvement,
):
    """
    Perform a greedy search based optimisation of a complete set of synthesis
    test patterns.
    
    See :ref:`optimisation` for details of the optimisation process and
    parameters.
    
    Parameters
    ==========
    wavelet_index : :py:class:`vc2_data_tables.WaveletFilters` or int
    wavelet_index_ho : :py:class:`vc2_data_tables.WaveletFilters` or int
    dwt_depth : int
    dwt_depth_ho : int
        The filter parameters.
    quantisation_matrix : {level: {orient: value, ...}, ...}
        The quantisation matrix in use.
    picture_bit_width : int
        The number of bits in the input pictures.
    synthesis_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
        Synthesis test patterns to use as the starting point for optimisation,
        as produced by e.g.  :py:func:`static_filter_analysis`.
    max_quantisation_index : int
        The maximum quantisation index to use, e.g. computed using
        :py:func:`quantisation_index_bound`.
    random_state : :py:class:`numpy.random.RandomState`
        The random number generator to use for the search.
    number_of_searches : int
        Repeat the greedy stochastic search process this many times for each
        test pattern. Since searches will tend to converge on local minima,
        increasing this parameter will tend to produce improved results.
    terminate_early : None or int
        If an integer, stop searching if the first ``terminate_early`` searches
        fail to find an improvement. If None, always performs all searches.
    added_corruption_rate : float
        The proportion of pixels to assign with a random value during each
        search attempt (0.0-1.0).
    removed_corruption_rate : float
        The proportion of pixels to reset to their starting value during each
        search attempt (0.0-1.0).
    base_iterations : int
        The initial number of search iterations to perform in each attempt.
    added_iterations_per_improvement : int
        The number of additional search iterations to perform whenever an
        improved picture is found.
    
    Returns
    =======
    optimised_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.OptimisedTestPatternSpecification`, ...}
        The optimised test patterns.
        
        Note that arrays are omitted for arrays which are just interleavings of
        other arrays.
    """
    # Create PyExps for all synthesis filtering stages, used to decode test
    # encoded patterns
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    _, synthesis_pyexps = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        make_variable_coeff_arrays(dwt_depth, dwt_depth_ho),
    )
    
    # Strip out all arrays which are simply interleavings of others (and
    # therefore don't need optimising several times)
    test_patterns_to_optimise = [
        (level, array_name, x, y, tp)
        for (level, array_name, x, y), tp in synthesis_test_patterns.items()
        if not synthesis_pyexps[(level, array_name)].nop
    ]
    
    input_min, input_max = signed_integer_range(picture_bit_width)
    
    optimised_test_patterns = OrderedDict()
    
    for signal_no, (level, array_name, x, y, tp) in enumerate(test_patterns_to_optimise):
        synthesis_pyexp = synthesis_pyexps[(level, array_name)][tp.target]
        
        added_corruptions_per_iteration = int(np.ceil(
            len(tp.pattern) * added_corruption_rate
        ))
        removed_corruptions_per_iteration = int(np.ceil(
            len(tp.pattern) * removed_corruption_rate
        ))
        
        logger.info(
            "Optimising test pattern %d of %d (level %d, %s[%d, %d])",
            signal_no + 1,
            len(test_patterns_to_optimise),
            level,
            array_name,
            x,
            y,
        )
        
        best_ts = None
        
        for flip_polarity, log_message in [
            (False, "Maximising..."),
            (True, "Minimising..."),
        ]:
            logger.info(log_message)
            
            # Run the search starting from the maximising and minimising signal
            if flip_polarity:
                flipped_tp = invert_test_pattern_specification(tp)
            else:
                flipped_tp = tp
            
            new_ts = optimise_synthesis_maximising_test_pattern(
                h_filter_params=h_filter_params,
                v_filter_params=v_filter_params,
                dwt_depth=dwt_depth,
                dwt_depth_ho=dwt_depth_ho,
                quantisation_matrix=quantisation_matrix,
                synthesis_pyexp=synthesis_pyexp,
                test_pattern_specification=flipped_tp,
                input_min=input_min,
                input_max=input_max,
                max_quantisation_index=max_quantisation_index,
                random_state=random_state,
                number_of_searches=number_of_searches,
                terminate_early=terminate_early,
                added_corruptions_per_iteration=added_corruptions_per_iteration,
                removed_corruptions_per_iteration=removed_corruptions_per_iteration,
                base_iterations=base_iterations,
                added_iterations_per_improvement=added_iterations_per_improvement,
            )
            
            # NB: when given a -ve and +ve value with equal magnitude, the +ve one
            # should be kept because this may require an additional bit to
            # represent in two's compliment arithmetic (e.g. -512 is 10-bits, +512
            # is 11-bits)
            if (
                best_ts is None or
                abs(new_ts.decoded_value) > abs(best_ts.decoded_value) or
                (
                    abs(new_ts.decoded_value) == abs(best_ts.decoded_value) and
                    new_ts.decoded_value > best_ts.decoded_value
                )
            ):
                best_ts = new_ts
        
        logger.info(
            "Largest signal magnitude achieved = %d (qi=%d)",
            best_ts.decoded_value,
            best_ts.quantisation_index,
        )
        
        optimised_test_patterns[(level, array_name, x, y)] = best_ts
    
    return optimised_test_patterns


def evaluate_test_pattern_outputs(
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    picture_bit_width,
    quantisation_matrix,
    max_quantisation_index,
    analysis_test_patterns,
    synthesis_test_patterns,
):
    """
    Given a set of test patterns, compute the signal levels actually produced
    by them when passed through a real encoder/decoder.
    
    Parameters
    ==========
    wavelet_index : :py:class:`vc2_data_tables.WaveletFilters` or int
    wavelet_index_ho : :py:class:`vc2_data_tables.WaveletFilters` or int
    dwt_depth : int
    dwt_depth_ho : int
        The filter parameters.
    picture_bit_width : int
        The number of bits in the input pictures.
    quantisation_matrix : {level: {orient: value, ...}, ...}
        The quantisation matrix.
    max_quantisation_index : int
        The maximum quantisation index to try (e.g. as computed by
        :py:func:`quantisation_index_bound`). Each synthesis test pattern will
        be quantised with every quantisation index up to (and inclusing) this
        limit and the worst-case value for any quantisation index will be
        reported.
    analysis_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
    synthesis_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
        The test patterns to assess, e.g. from
        :py:func:`static_filter_analysis` or
        :py:func:`optimise_synthesis_test_patterns`.
    
    Returns
    =======
    analysis_test_pattern_outputs : {(level, array_name, x, y): (lower_bound, upper_bound), ...}
    synthesis_test_pattern_outputs : {(level, array_name, x, y): ((lower_bound, qi), (upper_bound, qi)), ...}
        The worst-case signal levels achieved for each of the provided test
        signals when using minimising and maximising versions of the test
        pattern respectively.
        
        For the syntehsis test patterns, the quantisation index used to achieve
        the worst-case values is also reported.
        
        Includes values for *all* arrays and phases, even if array
        interleavings/subsamplings/renamings are omitted in the input
        arguments.
    """
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    
    input_min, input_max = signed_integer_range(picture_bit_width)
    
    analysis_test_pattern_outputs = OrderedDict()
    for i, ((level, array_name, x, y), test_pattern) in enumerate(
        analysis_test_patterns.items()
    ):
        logger.info(
            "Evaluating analysis test pattern %d of %d (Level %d, %s[%d, %d])...",
            i + 1,
            len(analysis_test_patterns),
            level,
            array_name,
            x,
            y,
        )
        analysis_test_pattern_outputs[(level, array_name, x, y)] = evaluate_analysis_test_pattern_output(
            h_filter_params=h_filter_params,
            v_filter_params=v_filter_params,
            dwt_depth=dwt_depth,
            dwt_depth_ho=dwt_depth_ho,
            level=level,
            array_name=array_name,
            test_pattern_specification=test_pattern,
            input_min=input_min,
            input_max=input_max,
        )
    
    _, synthesis_pyexps = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        make_variable_coeff_arrays(dwt_depth, dwt_depth_ho),
    )
    
    # Re-add results for interleaved/renamed entries
    analysis_test_pattern_outputs = add_missing_analysis_values(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        analysis_test_pattern_outputs,
    )
    
    synthesis_test_pattern_outputs = OrderedDict()
    for i, ((level, array_name, x, y), test_pattern) in enumerate(
        synthesis_test_patterns.items()
    ):
        logger.info(
            "Evaluating synthesis test pattern %d of %d (Level %d, %s[%d, %d])...",
            i + 1,
            len(synthesis_test_patterns),
            level,
            array_name,
            x,
            y,
        )
        synthesis_test_pattern_outputs[(level, array_name, x, y)] = evaluate_synthesis_test_pattern_output(
            h_filter_params=h_filter_params,
            v_filter_params=v_filter_params,
            dwt_depth=dwt_depth,
            dwt_depth_ho=dwt_depth_ho,
            quantisation_matrix=quantisation_matrix,
            synthesis_pyexp=synthesis_pyexps[(level, array_name)][test_pattern.target],
            test_pattern_specification=test_pattern,
            input_min=input_min,
            input_max=input_max,
            max_quantisation_index=max_quantisation_index,
        )
    
    # Re-add results for interleaved/renamed entries
    synthesis_test_pattern_outputs = add_missing_synthesis_values(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        synthesis_test_pattern_outputs,
    )
    
    return (
        analysis_test_pattern_outputs,
        synthesis_test_pattern_outputs,
    )

TestPoint = namedtuple("TestPoint", "level,array_name,x,y,maximise,tx,ty")
"""
:py:func:`~collections.namedtuple`. Definition of the location of a point in
an encoder/decoder tested by a test picture.

Parameters
==========
level : int
array_name : str
x : int
y : int
    The encoder/decoder intermediate value array and phase tested by this test
    point.
maximise : bool
    If True, at this point the signal level is being maximised, if False it
    will be minimised.
tx : int
ty : int
    The coordinates (in the intermediate value array (level, array_name)) which
    are being maximised or minimised.
"""

AnalysisPicture = namedtuple("AnalysisPicture", "picture,test_points")
"""
:py:func:`~collections.namedtuple`. A test picture which is to be used to
assess an analysis filter.

Parameters
==========
picture : :py:class:`numpy.array`
    The test picture.
test_points : [:py:class:`TestPoint`, ...]
    A list of locations within the analysis filter being tested by this
    picture.
"""

SynthesisPicture = namedtuple("SynthesisPicture", "picture,quantisation_index,test_points")
"""
:py:func:`~collections.namedtuple`. A test picture which is to be used to
assess an synthesis filter.

Parameters
==========
picture : :py:class:`numpy.array`
    The test picture.
quantisation_index : int
    The quantisation index to use for all picture slices when encoding the test
    picture.
test_points : [:py:class:`TestPoint`, ...]
    A list of locations within the synthesis filter being tested by this
    picture.
"""


def make_saturated_picture(polarities, input_min, input_max, zero=0):
    """
    (Internal utility.) Convert an array of -1, 0, +1 polarity values into an
    array of saturated values.
    """
    # NB: dtype=object used to allow unlimited precision Python integers
    # (though I can only hope that this is never actually required -- 128-bit
    # per pixel video, anybody?)
    out = np.full(polarities.shape, zero, dtype=object)
    out[polarities==+1] = input_max
    out[polarities==-1] = input_min
    return out


def generate_test_pictures(
    picture_width,
    picture_height,
    picture_bit_width,
    analysis_test_patterns,
    synthesis_test_patterns,
    synthesis_test_pattern_outputs,
):
    """
    Generate a series of test pictures containing the supplied selection of
    test patterns.
    
    Parameters
    ==========
    picture_width : int
    picture_height : int
        The dimensions of the pictures to generate.
    picture_bit_width : int
        The number of bits in the input pictures.
    analysis_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
    synthesis_test_patterns : {(level, array_name, x, y): :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
        The individual analysis and synthesis test patterns to be combined. A
        maximising and minimising variant of each pattern will be included in
        the output. See :py:func:`static_filter_analysis` and
        :py:func:`optimise_synthesis_test_patterns`.
    synthesis_test_pattern_outputs : {(level, array_name, x, y): ((lower_bound, qi), (upper_bound, qi)), ...}
        The worst-case quantisation indicies for each synthesis test pattern,
        as computed by :py:func:`evaluate_test_pattern_outputs`.
    
    Returns
    =======
    analysis_pictures : [:py:class:`AnalysisPicture`, ...]
    synthesis_pictures : [:py:class:`SynthesisPicture`, ...]
        A series of test pictures containing correctly aligned instances of
        each supplied test pattern.
        
        Pictures are returned with values in the range 0 to
        (2**picture_bit_width)-1, as expected by a VC-2 encoder.
        
        Each analysis picture includes a subset of the test patterns supplied
        (see :py:class:`AnalysisPicture`). The analysis test pictures are
        intended to be passed to an analysis filter as-is.
        
        Each synthesis test picture includes a subset of the test patterns
        supplied, grouped according to the quantisation index to be used.  The
        synthesis test pictures should first be passed through a synthesis
        filter and then the transform coefficients quantised using the
        quantisation index specified (see :py:class:`SynthesisPicture`). The
        quantised transform coefficients should then be passed through the
        synthesis filter.
    """
    input_min, input_max = unsigned_integer_range(picture_bit_width)
    input_mid = (input_max - input_min + 1) // 2
    
    # For better packing, the test patterns will be packed in size order,
    # largest first.
    analysis_test_patterns = OrderedDict(sorted(
        analysis_test_patterns.items(),
        key=lambda kv: len(kv[1].pattern),
        reverse=True,
    ))
    synthesis_test_patterns = OrderedDict(sorted(
        synthesis_test_patterns.items(),
        key=lambda kv: len(kv[1].pattern),
        reverse=True,
    ))
    
    # Pack analysis signals
    logger.info(
        "Packing %d analysis test patterns...",
        len(analysis_test_patterns)*2,
    )
    analysis_test_patterns_bipolar = OrderedDict(
        (
            (level, array_name, x, y, maximise),
            spec if maximise else invert_test_pattern_specification(spec),
        )
        for (level, array_name, x, y), spec in analysis_test_patterns.items()
        for maximise in [True, False]
    )
    pictures, locations = pack_test_patterns(
        picture_width,
        picture_height,
        analysis_test_patterns_bipolar,
    )
    if len(locations) < len(analysis_test_patterns_bipolar):
        logger.warning(
            "%d analysis test patterns were too large to fit in a "
            "%d x %d picture and were omitted.",
            len(analysis_test_patterns_bipolar) - len(locations),
            picture_width, picture_height,
        )
    analysis_pictures = [
        AnalysisPicture(
            make_saturated_picture(picture, input_min, input_max, input_mid),
            [],
        )
        for picture in pictures
    ]
    for (level, array_name, x, y, maximise), (picture_index, tx, ty) in locations.items():
        analysis_pictures[picture_index].test_points.append(TestPoint(
            level, array_name, x, y,
            maximise,
            tx, ty,
        ))
    logger.info(
        "Packed analysis test patterns into %d pictures.",
        len(analysis_pictures),
    )
    
    # Group synthesis test patterns by required quantisation index
    logger.info(
        "Packing %d synthesis test patterns...",
        len(synthesis_test_patterns)*2,
    )
    # {quantisation_index: {(level, array_name, x, y, maximise): spec, ...}, ...}
    synthesis_test_patterns_grouped = defaultdict(OrderedDict)
    for (level, array_name, x, y), spec in synthesis_test_patterns.items():
        (_, minimising_qi), (_, maximising_qi) = synthesis_test_pattern_outputs[
            (level, array_name, x, y)
        ]
        
        synthesis_test_patterns_grouped[maximising_qi][
            (level, array_name, x, y, True)
        ] = spec
        synthesis_test_patterns_grouped[minimising_qi][
            (level, array_name, x, y, False)
        ] = invert_test_pattern_specification(spec)
    
    # Pack the synthesis the test patterns, grouped by QI
    synthesis_pictures = []
    num_too_large = 0
    for qi in sorted(synthesis_test_patterns_grouped):
        pictures, locations = pack_test_patterns(
            picture_width,
            picture_height,
            synthesis_test_patterns_grouped[qi],
        )
        num_too_large += len(synthesis_test_patterns_grouped[qi]) - len(locations)
        
        this_synthesis_pictures = [
            SynthesisPicture(
                make_saturated_picture(picture, input_min, input_max, input_mid),
                qi,
                [],
            )
            for picture in pictures
        ]
        
        for (level, array_name, x, y, maximise), (picture_index, tx, ty) in locations.items():
            this_synthesis_pictures[picture_index].test_points.append(TestPoint(
                level, array_name, x, y,
                maximise,
                tx, ty,
            ))
        
        synthesis_pictures += this_synthesis_pictures
    if len(locations) < len(synthesis_test_patterns_grouped[qi]):
        logger.warning(
            "%d synthesis test patterns were too large to fit in a "
            "%d x %d picture and were omitted.",
            num_too_large,
            picture_width, picture_height,
        )
    logger.info(
        "Packed synthesis test patterns into %d pictures.",
        len(synthesis_pictures),
    )
    
    return (
        analysis_pictures,
        synthesis_pictures,
    )
