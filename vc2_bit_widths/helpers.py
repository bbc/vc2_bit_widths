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

The test signals generated for the synthesis filter can be optimised and
specialised for a particular codec configuration and bit depth:

.. autofunction:: optimise_synthesis_test_signals

Missing entries in the output of the above (due to removal of duplicate filter
values, e.g. from interleaved arrays) may be reinstated using:

.. autofunction:: add_omitted_synthesis_values

"""

from collections import OrderedDict

import logging

from math import ceil

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
    OptimisedTestSignalSpecification,
    make_analysis_maximising_signal,
    make_synthesis_maximising_signal,
    optimise_synthesis_maximising_signal,
    evaluate_analysis_test_signal_output,
    evaluate_synthesis_test_signal_output,
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
                    cached_analysis_coeff_arrays,
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
        
        optimised_test_signals[(level, array_name, x, y)] = best_ts
    
    return optimised_test_signals


def evaluate_test_signal_outputs(
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    picture_bit_width,
    quantisation_matrix,
    max_quantisation_index,
    analysis_test_signals,
    synthesis_test_signals,
):
    """
    Given a set of test signals, compute the signal levels actually produced in
    a real encoder/decoder.
    
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
        The quantisation matrix in use.
    max_quantisation_index : int
        The maximum quantisation index to use (e.g. from
        :py:func:`quantisation_index_bound`).
    analysis_test_signals: {(level, array_name, x, y): :py:class:`vc2_bit_widths.signal_generation.TestSignalSpecification`, ...}
    synthesis_test_signals: {(level, array_name, x, y): :py:class:`vc2_bit_widths.signal_generation.TestSignalSpecification`, ...}
        From :py:func:`static_filter_analysis` or
        :py:func:`optimise_synthesis_test_signals`. The test signals to assess.
    
    Returns
    =======
    analysis_test_signal_outputs : {(level, array_name, x, y): (lower_bound, upper_bound), ...}
        The signal levels achieved for each of the provided analysis test
        signals for minimising signal values and maximising values
        respectively.
    synthesis_test_signal_outputs : {(level, array_name, x, y): ((lower_bound, qi), (upper_bound, qi)), ...}
        The signal levels achieved, and quantisation indices used to achieve
        them, for each of the provided synthesis test signals. Given for
        minimising signal values and maximising values respectively.
    """
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    
    input_min, input_max = signed_integer_range(picture_bit_width)
    
    analysis_test_signal_outputs = OrderedDict()
    for i, ((level, array_name, x, y), test_signal) in enumerate(
        analysis_test_signals.items()
    ):
        logger.info(
            "Evaluating analysis test pattern %d of %d (Level %d, %s[%d, %d])...",
            i + 1,
            len(analysis_test_signals),
            level,
            array_name,
            x,
            y,
        )
        analysis_test_signal_outputs[(level, array_name, x, y)] = evaluate_analysis_test_signal_output(
            h_filter_params=h_filter_params,
            v_filter_params=v_filter_params,
            dwt_depth=dwt_depth,
            dwt_depth_ho=dwt_depth_ho,
            level=level,
            array_name=array_name,
            test_signal=test_signal,
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
    
    synthesis_test_signal_outputs = OrderedDict()
    for i, ((level, array_name, x, y), test_signal) in enumerate(
        synthesis_test_signals.items()
    ):
        logger.info(
            "Evaluating synthesis test pattern %d of %d (Level %d, %s[%d, %d])...",
            i + 1,
            len(synthesis_test_signals),
            level,
            array_name,
            x,
            y,
        )
        synthesis_test_signal_outputs[(level, array_name, x, y)] = evaluate_synthesis_test_signal_output(
            h_filter_params=h_filter_params,
            v_filter_params=v_filter_params,
            dwt_depth=dwt_depth,
            dwt_depth_ho=dwt_depth_ho,
            quantisation_matrix=quantisation_matrix,
            synthesis_pyexp=synthesis_pyexps[(level, array_name)][test_signal.target],
            test_signal=test_signal,
            input_min=input_min,
            input_max=input_max,
            max_quantisation_index=max_quantisation_index,
        )
    
    return (
        analysis_test_signal_outputs,
        synthesis_test_signal_outputs,
    )


def add_omitted_synthesis_values(
    wavelet_index,
    wavelet_index_ho,
    dwt_depth,
    dwt_depth_ho,
    synthesis_values,
):
    """
    Re-create entries from a test signal dictionary which have been omitted on
    the basis of just being interleavings of other arrays.
    
    When :py:func;`optimise_synthesis_test_signals` is used, it only outputs
    test signals for filter phases which are not simple interleavings of other
    arrays. When producing human-readable tables of bit width information,
    however, these omitted filters and phases are still required for display.
    
    Parameters
    ==========
    wavelet_index : :py:class:`vc2_data_tables.WaveletFilters` or int
    wavelet_index_ho : :py:class:`vc2_data_tables.WaveletFilters` or int
    dwt_depth : int
    dwt_depth_ho : int
        The filter parameters.
    synthesis_values : {(level, array_name, x, y): values, ...}
        A dictionary of values (one per filter and phase), e.g. from
        :py:func:`evaluate_test_signal_outputs`, with values omitted for
        interleaved values.
    
    Returns
    =======
    full_synthesis_values : {(level, array_name, x, y): values, ...}
        The complete set of values with omitted (interleaved) values present.
    """
    # NB: Used only to enumerate the complete set of arrays/test signals and
    # get array periods
    h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    v_filter_params = LIFTING_FILTERS[wavelet_index]
    _, intermediate_arrays = synthesis_transform(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        make_variable_coeff_arrays(dwt_depth, dwt_depth_ho),
    )
    
    out = OrderedDict()
    
    for (level, array_name), array in intermediate_arrays.items():
        for x in range(array.period[0]):
            for y in range(array.period[1]):
                # Below we work out which source array to use to populate the
                # current array/phase
                src_level = level
                src_array_name = array_name
                src_x = x
                src_y = y
                if (level, array_name, x, y) not in synthesis_values:
                    if array_name.startswith("L'"):
                        if y % 2 == 0:
                            src_array_name = "LL"
                        else:
                            src_array_name = "LH"
                        src_y = y // 2
                    elif array_name.startswith("H'"):
                        if y % 2 == 0:
                            src_array_name = "HL"
                        else:
                            src_array_name = "HH"
                        src_y = y // 2
                    elif array_name.startswith("DC'"):
                        if x % 2 == 0:
                            src_array_name = "L"
                        else:
                            src_array_name = "H"
                        src_x = x // 2
                    elif array_name == "Output":
                        src_array_name = "DC"
                    elif array_name == "LL" or array_name == "L":
                        src_level = level - 1
                        src_array_name = "Output"
                    else:
                        # Should never reach this point so long as only
                        # interleavings and nop-bit-shifts are omitted
                        assert False
                
                px, py = intermediate_arrays[(src_level, src_array_name)].period
                src_x %= px
                src_y %= py
                
                out[(level, array_name, x, y)] = synthesis_values.get((
                    src_level,
                    src_array_name,
                    src_x,
                    src_y,
                ), out.get((
                    src_level,
                    src_array_name,
                    src_x,
                    src_y,
                )))
    
    return out
