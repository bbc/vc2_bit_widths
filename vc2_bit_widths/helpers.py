"""
VC-2 Bit-Width Analysis Helper Functions
========================================

This module contains helper routines which perform all of the key tasks
implemented by this library.

The first step is typically to statically analyse a specified filter:

.. autofunction:: static_filter_analysis

Next, the filter signal range statistics can be made concrete for a particular
picture bit depth using:

.. autofunction:: evaluate_filter_bounds

In addition, the maximum quantisation index required for a picture bit depth
can also be determined:

.. autofunction:: quantisation_index_bound

Finally, the test signals generated for the synthesis filter can be optimised
and specialised for a particular codec configuration and bit depth:

.. autofunction:: optimise_synthesis_test_signals

"""

from collections import OrderedDict

import logging

from math import ceil

from vc2_data_tables import LIFTING_FILTERS

from vc2_bit_widths.infinite_arrays import SymbolArray

from vc2_bit_widths.vc2_filters import (
    make_symbol_coeff_arrays,
    make_variable_coeff_arrays,
    analysis_transform,
    synthesis_transform,
)

from vc2_bit_widths.signal_bounds import (
    analysis_filter_bounds,
    synthesis_filter_bounds,
    evaluate_analysis_filter_bounds,
    evaluate_synthesis_filter_bounds,
    signed_integer_range,
)

from vc2_bit_widths.signal_generation import (
    TestSignalSpecification,
    make_analysis_maximising_signal,
    make_synthesis_maximising_signal,
    optimise_synthesis_maximising_signal,
)

from vc2_bit_widths.quantisation import (
    maximum_dequantised_magnitude,
    maximum_useful_quantisation_index,
)


logger = logging.getLogger(__name__)


def static_filter_analysis(wavelet_index, wavelet_index_ho, dwt_depth, dwt_depth_ho):
    r"""
    Performs a complete static analysis of a VC-2 filter configuration,
    computing upper- and lower-bounds for filter values and worst case and
    heuristic test signals for analysis and synthesis filters respectively.
    
    Internally this function is a thin wrapper around the following principal
    functions:
    
    * :py:func:`vc2_bit_widths.signal_bounds.analysis_filter_bounds`
    * :py:func:`vc2_bit_widths.signal_bounds.synthesis_filter_bounds`
    * :py:func:`vc2_bit_widths.signal_generation.make_analysis_maximising_signal`
    * :py:func:`vc2_bit_widths.signal_generation.make_synthesis_maximising_signal`
    
    Parameters
    ==========
    wavelet_index : :py:class:`vc2_data_tables.WaveletFilters` or int
    wavelet_index_ho : :py:class:`vc2_data_tables.WaveletFilters` or int
    dwt_depth : int
    dwt_depth_ho : int
        The filter parameters.
    
    Returns
    =======
    analysis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
    synthesis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
        Expressions defining the upper and lower bounds for all intermediate
        and final analysis and synthesis filter values.
        
        The keys of the returned dictionaries give the level (int), array name
        (str) and filter phase (x, y) for which each set of bounds corresponds.
        The naming conventions used are those defined by
        :py:func:`vc2_bit_widths.vc2_filters.analysis_transform` and
        :py:func:`vc2_bit_widths.vc2_filters.synthesis_transform`.
        
        The lower and upper bounds are given as
        :py:class:`vc2_bit_widths.linexp.LinExp`\ s.
        
        For the analysis filter bounds, the expressions are defined in terms of
        the variables ``LinExp("signal_min")`` and ``LinExp("signal_max")``.
        These should be substituted for the minimum and maximum picture signal
        level to find the upper and lower bounds for a particular picture bit
        width. (See
        :py:func:`vc2_bit_widths.signal_bounds.evaluate_analysis_filter_bounds`.)
        
        For the synthesis filter bounds, the expressions are defined in terms
        of variables of the form ``LinExp("signal_LEVEL_ORIENT_min")`` and
        ``LinExp("signal_LEVEL_ORIENT_max")``.  These should be substituted for
        the lower and upper bounds computed for the relevant analysis filter
        (See
        :py:func:`vc2_bit_widths.signal_bounds.evaluate_synthesis_filter_bounds`.)
    analysis_test_signals: {(level, array_name, x, y): :py:class:`vc2_bit_widths.signal_generation.TestSignalSpecification`, ...}
    synthesis_test_signals: {(level, array_name, x, y): :py:class:`vc2_bit_widths.signal_generation.TestSignalSpecification`, ...}
        Test signals which attempt to maximise (or minimise) for each
        intermediate and final analysis and synthesis filter output.
        
        The keys of the returned dictionaries give the level (int), array name
        (str) and filter phase (x, y) for which each set of bounds corresponds.
        The naming conventions used are those defined by
        :py:func:`vc2_bit_widths.vc2_filters.analysis_transform` and
        :py:func:`vc2_bit_widths.vc2_filters.synthesis_transform`.
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
    )
    array_num = 0
    
    # Compute bounds/test signal for every intermediate/output analysis value
    analysis_signal_bounds = OrderedDict()
    analysis_test_signals = OrderedDict()
    for (level, array_name), target_array in intermediate_analysis_arrays.items():
        for x in range(target_array.period[0]):
            for y in range(target_array.period[1]):
                array_num += 1
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
                
                # Generate test signal
                analysis_test_signals[(level, array_name, x, y)] = make_analysis_maximising_signal(
                    picture_array,
                    target_array,
                    x, y,
                )
    
    # Create the algebraic representation of the synthesis transform
    coeff_arrays = make_symbol_coeff_arrays(dwt_depth, dwt_depth_ho)
    synthesis_output_array, intermediate_synthesis_values = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        coeff_arrays,
    )
    
    # Count the total number of arrays for use in logging messages
    num_arrays = sum(
        array.period[0] * array.period[1]
        for array in intermediate_synthesis_values.values()
    )
    array_num = 0
    
    # Compute bounds/test signal for every intermediate/output analysis value
    synthesis_signal_bounds = OrderedDict()
    synthesis_test_signals = OrderedDict()
    for (level, array_name), target_array in intermediate_synthesis_values.items():
        for x in range(target_array.period[0]):
            for y in range(target_array.period[1]):
                array_num += 1
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
                
                # Compute test signal
                synthesis_test_signals[(level, array_name, x, y)] = make_synthesis_maximising_signal(
                    picture_array,
                    analysis_coeff_arrays,
                    target_array,
                    synthesis_output_array,
                    x, y,
                )
    
    return (
        analysis_signal_bounds,
        synthesis_signal_bounds,
        analysis_test_signals,
        synthesis_test_signals,
    )


def evaluate_filter_bounds(analysis_signal_bounds, synthesis_signal_bounds, picture_bit_width):
    r"""
    Evaluate all analysis and synthesis filter signal bounds expressions for a
    given picture bit width, giving concrete signal ranges.
    
    Parameters
    ==========
    analysis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
    synthesis_signal_bounds : {(level, array_name, x, y): (lower_bound_exp, upper_bound_exp), ...}
        The outputs of :py:func:`static_filter_analysis`.
    picture_bit_width : int
        The number of bits in the input pictures.
    
    Returns
    =======
    concrete_analysis_signal_bounds : {(level, array_name, x, y): (lower_bound, upper_bound), ...}
    concrete_synthesis_signal_bounds : {(level, array_name, x, y): (lower_bound, upper_bound), ...}
        The concrete signal bounds for all analysis and signal filters given a
        ``picture_bit_width`` input picture and worst-case quantisation.
    """
    # Evaluate analysis bounds
    concrete_analysis_signal_bounds = OrderedDict(
        (key, evaluate_analysis_filter_bounds(
            lower_exp,
            upper_exp,
            picture_bit_width,
        ))
        for key, (lower_exp, upper_exp) in analysis_signal_bounds.items()
    )
    
    # Inflate analysis output bounds to account for worst-case quantisation of
    # transform coefficients in input to synthesis stages.
    #
    # NB: Intermediate arrays use (1, "L")/(1, "LL") for the DC band while the
    # general VC-2 convention is to use (0, "L")/(0, "LL") and so the level
    # number is tweaked in the dictionary below.
    coeff_bounds = {
        (
            level if orient not in ("L", "LL") else 0,
            orient,
        ): (
            maximum_dequantised_magnitude(lower_bound),
            maximum_dequantised_magnitude(upper_bound),
        )
        for (level, orient, _, _), (lower_bound, upper_bound)
        in concrete_analysis_signal_bounds.items()
    }
    
    # Evaluate synthesis bounds
    concrete_synthesis_signal_bounds = OrderedDict(
        (key, evaluate_synthesis_filter_bounds(
            lower_exp,
            upper_exp,
            coeff_bounds,
        ))
        for key, (lower_exp, upper_exp) in synthesis_signal_bounds.items()
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
    quantisation_index : int
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


def optimise_synthesis_test_signals(
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    quantisation_matrix,
    picture_bit_width,
    synthesis_test_signals,
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
    test signals.
    
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
    synthesis_test_signals: {(level, array_name, x, y): :py:class:`vc2_bit_widths.signal_generation.TestSignalSpecification`, ...}
        Test signals which attempt to maximise each intermediate and final
        synthesis filter output, as produced by
        :py:func:`static_filter_analysis`.
    max_quantisation_index : int
        The maximum quantisation index to use, e.g. computed using
        :py:func:`quantisation_index_bound`.
    random_state : :py:class:`numpy.random.RandomState`
        The random number generator to use for the search.
    number_of_searches : int
        Repeat the greedy stochastic search process this many times for each
        test signal. Since searches will tend to converge on local minima,
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
    optimised_test_signals : {(level, array_name, x, y): :py:class:`vc2_bit_widths.signal_generation.OptimisedTestSignalSpecification`, ...}
        The optimised test signals.
        
        Note that arrays are omitted for arrays which are just interleavings of
        other arrays.
    """
    # Create PyExps for all synthesis filtering stages, used to decode test
    # encoded patterns
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    _, synthesis_pyexps = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        make_variable_coeff_arrays(dwt_depth, dwt_depth_ho),
    )
    
    # Strip out all arrays which are simply interleavings of others (and
    # therefore don't need optimising several times)
    test_signals_to_optimise = [
        (level, array_name, x, y, ts)
        for (level, array_name, x, y), ts in synthesis_test_signals.items()
        if not synthesis_pyexps[(level, array_name)].nop
    ]
    
    input_min, input_max = signed_integer_range(picture_bit_width)
    
    optimised_test_signals = OrderedDict()
    
    for signal_no, (level, array_name, x, y, ts) in enumerate(test_signals_to_optimise):
        synthesis_pyexp = synthesis_pyexps[(level, array_name)][ts.target]
        
        added_corruptions_per_iteration = int(ceil(
            len(ts.picture) * added_corruption_rate
        ))
        removed_corruptions_per_iteration = int(ceil(
            len(ts.picture) * removed_corruption_rate
        ))
        
        logger.info(
            "Optimising test pattern %d of %d (level %d, %s[%d, %d])",
            signal_no + 1,
            len(test_signals_to_optimise),
            level,
            array_name,
            x,
            y,
        )
        
        best_ts = None
        
        for flip_polarity, log_message in [
            (+1, "Maximising..."),
            (-1, "Minimising..."),
        ]:
            logger.info(log_message)
            
            # Run the search starting from the maximising and minimising signal
            flipped_ts = TestSignalSpecification(
                target=ts.target,
                picture={
                    (x, y): polarity * flip_polarity
                    for (x, y), polarity in ts.picture.items()
                },
                picture_translation_multiple=ts.picture_translation_multiple,
                target_translation_multiple=ts.target_translation_multiple,
            )
            
            new_ts = optimise_synthesis_maximising_signal(
                h_filter_params=h_filter_params,
                v_filter_params=v_filter_params,
                dwt_depth=dwt_depth,
                dwt_depth_ho=dwt_depth_ho,
                quantisation_matrix=quantisation_matrix,
                synthesis_pyexp=synthesis_pyexp,
                test_signal=flipped_ts,
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
            
            if (
                best_ts is None or
                abs(new_ts.decoded_value) > abs(best_ts.decoded_value)
            ):
                best_ts = new_ts
        
        logger.info(
            "Largest signal magnitude achieved = %d (qi=%d)",
            best_ts.decoded_value,
            best_ts.quantisation_index,
        )
        
        optimised_test_signals[(level, array_name, x, y)] = best_ts
    
    return optimised_test_signals