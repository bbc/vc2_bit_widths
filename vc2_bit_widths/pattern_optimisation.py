"""
(Synthesis) Test Pattern Optimisation
=====================================

This module contains routines for optimising heuristic synthesis filter test
patterns for a particular codec configuration (e.g. bit width and quantisation
matrix).

"""

import logging

from collections import namedtuple

import numpy as np

from vc2_bit_widths.fast_partial_analyse_quantise_synthesise import (
    FastPartialAnalyseQuantiseSynthesise,
)

from vc2_bit_widths.pattern_generation import (
    convert_test_pattern_to_array_and_slice,
)


logger = logging.getLogger(__name__)


OptimisedTestPatternSpecification = namedtuple(
    "OptimisedTestPatternSpecification",
    (
        "target,pattern,pattern_translation_multiple,target_translation_multiple,"
        "quantisation_index,decoded_value,num_search_iterations"
    ),
)
"""
A test pattern specification which has been optimised to produce more extreme
signal values for a particular codec configuration.

Parameters
==========
target : (tx, ty)
pattern : {(x, y): polarity, ...}
pattern_translation_multiple : (mx, my)
target_translation_multiple : (tmx, tmy)
    Same as :py:class:`TestPatternSpecification`
quantisation_index : int
    The quantisation index which, when used for all coded picture slices,
    produces the largest values when this pattern is decoded.
decoded_value : int
    For informational purposes. The value which will be produced in the target
    decoder array for this input pattern.
num_search_iterations : int
    For informational purposes. The number of search iterations performed to
    find this value.
"""


def find_quantisation_index_with_greatest_output_magnitude(
    pattern,
    codec,
):
    """
    Find the quantisation index which causes the decoded value to have the
    largest magnitude.
    
    Parameters
    ==========
    pattern : :py:class:`numpy.array`
        The pixels to be encoded and decoded. This array will be modified
        during the call.
    codec : :py:class:`~vc2_bit_widths.fast_partial_analyse_quantise_synthesise.FastPartialAnalyseQuantiseSynthesise`
        An encoder/quantiser/decoder object to evaluate the pattern with.
    
    Returns
    =======
    greatest_decoded_value : int
        The decoded value with the greatest magnitude of any quantisation index
        tested.
    quantisation_index : int
        The quantisation index which produced the decoded value with the
        greatest magnitude.
    """
    decoded_values = codec.analyse_quantise_synthesise(pattern)
    i = np.argmax(np.abs(decoded_values))
    return (decoded_values[i], codec.quantisation_indices[i])


def choose_random_indices_of(array, num_indices, random_state):
    """
    Generate an index array for :py:class:`array` containing ``num_indices``
    independent random indices (possibly including repeats).
    
    Parameters
    ==========
    array : :py:class:`numpy.array`
    num_indices : int
    random_state : :py:class:`numpy.random.RandomState`
    
    Returns
    =======
    ([i, ...], [j, ...], ...)
        An index array (or rather, a tuple of index arrays) which index random
        entries in ``array``.
    """
    return tuple(
        random_state.randint(0, s, num_indices)
        for s in array.shape
    )


def greedy_stochastic_search(
    starting_pattern,
    search_slice,
    input_min,
    input_max,
    codec,
    random_state,
    added_corruptions_per_iteration,
    removed_corruptions_per_iteration,
    base_iterations,
    added_iterations_per_improvement,
):
    """
    Use a greedy stochastic search to find perturbations to the supplied test
    pattern which produce larger decoded values after quantisation.
    
    At a high-level the greedy search proceeds as follows:
    
    1. Make some random changes to the input pattern.
    2. See if these changes made the decoded output larger. If they did, keep
       the changes, if they didn't, discard them.
    3. Go to step 1.
    
    The algorithm uses a simple self-termination scheme to stop the search once
    a local maximum has been found. A counter is initially set to
    ``base_iterations`` and decremented after every iteration. If the iteration
    produced an larger output value, the counter is incremented by
    ``added_iterations_per_improvement``. Once the counter reaches zero, the
    search terminates and the best pattern found is returned.
    
    Parameters
    ==========
    starting_pattern : :py:class:`numpy.array`
        The pattern to use as the starting-point for the search.
    search_slice
        A :py:mod:`numpy` compatible array slice specification which defines
        the region of ``starting_pattern`` which should be mutated during the
        search.
    input_min : number
    input_max : number
        The value ranges for random values which may be inserted into the
        pattern.
    codec : :py:class:`~vc2_bit_widths.fast_partial_analyse_quantise_synthesise.FastPartialAnalyseQuantiseSynthesise`
        An encoder/quantiser/decoder object which will be used to test each
        pattern.
    random_state : :py:class:`numpy.random.RandomState`
        The random number generator to use for the search.
    added_corruptions_per_iteration : int
        The number of pixels to assign with a random value during each search
        attempt.
    removed_corruptions_per_iteration : int
        The number of pixels entries to reset to their original value during
        each search attempt.
    base_iterations : int
        The initial number of search iterations to perform.
    added_iterations_per_improvement : int
        The number of additional search iterations to perform whenever an
        improved pattern is found.
    
    Returns
    =======
    pattern : :py:class:`numpy.array`
        The perturbed pattern which produced the decoded value with the
        greatest magnitude after quantisation.
    greatest_decoded_value : float
        The value produced by decoding the input at the worst-case quantisation
        index.
    quantisation_index : int
        The quantisation index which yielded the largest decoded value.
    decoded_values : [int, ...]
        The largest decoded value found during each search iteration.
        Indirectly indicates the number of search iterations performed. May be
        used as an aid to tuning search parameters.
    """
    # Use the input with no corruption as the baseline for the search
    best_pattern = starting_pattern.copy()
    cur_pattern = best_pattern.copy()
    best_decoded_value, best_qi = find_quantisation_index_with_greatest_output_magnitude(
        cur_pattern,
        codec,
    )
    
    # Array to use as the working array for the partial encode/decode (since
    # this corrupts the array passed in)
    working_array = np.empty_like(cur_pattern)
    
    decoded_values = []
    iterations_remaining = base_iterations
    while iterations_remaining > 0:
        iterations_remaining -= 1
        
        cur_pattern[:] = best_pattern
        
        # Slice out the pattern 
        starting_slice = starting_pattern[search_slice]
        cur_slice = cur_pattern[search_slice]
        
        # Restore a random set of indices in the vector to their original value
        if removed_corruptions_per_iteration:
            reset_indices = choose_random_indices_of(
                cur_slice,
                removed_corruptions_per_iteration,
                random_state,
            )
            cur_slice[reset_indices] = starting_slice[reset_indices]
        
        # Corrupt a random set of indices in the vector
        corrupt_indices = choose_random_indices_of(
            cur_slice,
            added_corruptions_per_iteration,
            random_state,
        )
        cur_slice[corrupt_indices] = random_state.choice(
            (input_min, input_max),
            added_corruptions_per_iteration,
        )
        
        # Find the largest decoded value for the newly modified input vector
        working_array[:] = cur_pattern
        cur_decoded_value, cur_qi = find_quantisation_index_with_greatest_output_magnitude(
            working_array,
            codec,
        )
        
        decoded_values.append(cur_decoded_value)
        
        # Keep the input vector iff it is an improvement on the previous best
        #
        # NB: when given a -ve and +ve value with equal magnitude, the +ve one
        # should be kept because this may require an additional bit to
        # represent in two's compliment arithmetic (e.g. -512 is 10-bits, +512
        # is 11-bits)
        if (
            abs(cur_decoded_value) > abs(best_decoded_value) or
            (
                abs(cur_decoded_value) == abs(best_decoded_value) and
                cur_decoded_value > best_decoded_value
            )
        ):
            best_pattern[:] = cur_pattern
            best_decoded_value = cur_decoded_value
            best_qi = cur_qi
            
            iterations_remaining += added_iterations_per_improvement
    
    return best_pattern, best_decoded_value, best_qi, decoded_values


def optimise_synthesis_maximising_test_pattern(
    h_filter_params,
    v_filter_params,
    dwt_depth,
    dwt_depth_ho,
    quantisation_matrix,
    synthesis_pyexp,
    test_pattern,
    input_min,
    input_max,
    max_quantisation_index,
    random_state,
    number_of_searches,
    terminate_early,
    added_corruptions_per_iteration,
    removed_corruptions_per_iteration,
    base_iterations,
    added_iterations_per_improvement,
):
    """
    Optimise a test pattern generated by
    :py:func:`make_synthesis_maximising_signal` using repeated greedy
    stochastic search, attempting to produce larger signal values for specific
    codec configurations.
    
    The basic :py:func:`make_synthesis_maximising_signal` function simply uses
    a heuristic to produce a patterns likely to produce extreme values in the
    general case. By contrast this function attempts to more directly optimise
    the test pattern for particular input signal ranges and quantiser
    configurations.
    
    Parameters
    ==========
    h_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
    v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
        Horizontal and vertical filter *synthesis* (not analysis!) filter
        parameters (e.g. from :py:data:`vc2_data_tables.LIFTING_FILTERS`)
        defining the wavelet transform used.
    dwt_depth : int
    dwt_depth_ho : int
        The transform depth used by the filters.
    quantisation_matrix : {level: {orient: int, ...}, ...}
        The VC-2 quantisation matrix to use.
    synthesis_pyexp : :py:class:`~vc2_bit_widths.PyExp`
        A :py:class:`~vc2_bit_widths.PyExp` expression which defines the
        synthesis process for the decoder value the test pattern is
        maximising/minimising. Such an expression is usually obtained from the
        use of :py:func;`~vc2_bit_widths.vc2_filters.synthesis_transform` and
        :py:func;`~vc2_bit_widths.vc2_filters.make_variable_coeff_arrays`.
    test_pattern : :py:class:`TestPatternSpecification`
        The test pattern to optimise. This test pattern must be translated such
        that no analysis or synthesis step depends on VC-2's edge extension
        behaviour. This will be the case for test patterns produced by
        :py:class:`make_synthesis_maximising_signal`.
    input_min : int
    input_max : int
        The minimum and maximum value which may be used in the test pattern.
    max_quantisation_index : int
        The maximum quantisation index to use. This should be set high enough
        that at the highest quantisation level all transform coefficients are
        quantised to zero.
    random_state : :py:class:`numpy.random.RandomState`
        The random number generator to use during the search.
    number_of_searches : int
        Repeat the greedy stochastic search process this many times. Since
        searches will tend to converge on local minima, increasing this
        parameter will tend to produce improved results.
    terminate_early : None or int
        If an integer, stop searching if the first ``terminate_early`` searches
        fail to find an improvement. If None, always performs all searches.
    added_corruptions_per_iteration : int
    removed_corruptions_per_iteration : int
    base_iterations : int
    added_iterations_per_improvement : int
        Greedy stochastic search parameters. See the equivalently named
        parameters to :py:func:`greedy_stochastic_search`.
    
    Returns
    =======
    optimised_test_pattern : :py:class:`OptimisedTestPatternSpecification`
        The optimised test pattern found during the search.
    """
    pattern_array, search_slice = convert_test_pattern_to_array_and_slice(
        test_pattern.pattern,
        input_min,
        input_max,
        dwt_depth,
        dwt_depth_ho,
    )
    
    # Prepare the encoder/quantiser/decoder
    quantisation_indices = list(range(max_quantisation_index + 1))
    codec = FastPartialAnalyseQuantiseSynthesise(
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        quantisation_indices,
        synthesis_pyexp,
    )
    
    # Compute initial score prior to search for display in the debug log
    if logger.getEffectiveLevel() <= logging.INFO:
        _, base_decoded_value, base_qi, _ = greedy_stochastic_search(
            pattern_array,
            search_slice,
            input_min,
            input_max,
            codec,
            None,
            0,
            0,
            0,
            0,
        )
        logger.info("Unmodified value = %d (qi=%d)", base_decoded_value, base_qi)
    
    # Run the greedy search several times, keeping the best result.
    best_pattern = None
    best_decoded_value = None
    best_qi = None
    total_iterations = 0
    improvement_found = False
    assert number_of_searches >= 1
    for search_no in range(number_of_searches):
        new_pattern, new_decoded_value, new_qi, decoded_values = greedy_stochastic_search(
            pattern_array,
            search_slice,
            input_min,
            input_max,
            codec,
            random_state,
            added_corruptions_per_iteration,
            removed_corruptions_per_iteration,
            base_iterations,
            added_iterations_per_improvement,
        )
        
        total_iterations += len(decoded_values)
        
        logger.info(
            "Search %d of %d found value = %d (qi=%d) after %d iterations.",
            search_no + 1,
            number_of_searches,
            new_decoded_value,
            new_qi,
            len(decoded_values),
        )
        
        # NB: when given a -ve and +ve value with equal magnitude, the +ve one
        # should be kept because this may require an additional bit to
        # represent in two's compliment arithmetic (e.g. -512 is 10-bits, +512
        # is 11-bits)
        if (
            best_decoded_value is None or
            abs(new_decoded_value) > abs(best_decoded_value) or
            (
                abs(new_decoded_value) == abs(best_decoded_value) and
                new_decoded_value > best_decoded_value
            )
        ):
            best_pattern = new_pattern
            best_decoded_value = new_decoded_value
            best_qi = new_qi
        
        # Terminate early if no improvements to the pattern are made (detected
        # by the returning of a new pattern by the search)
        if not improvement_found and not np.array_equal(pattern_array, new_pattern):
            improvement_found = True
        if (
            terminate_early is not None and
            search_no + 1 >= terminate_early and
            not improvement_found
        ):
            logger.info(
                "Terminating early after %d search(es) returned no improvement.",
                search_no + 1,
            )
            break
    
    logger.info(
        "Final value = %d (qi=%d) after %d iterations.",
        best_decoded_value,
        best_qi,
        total_iterations,
    )
    
    # Convert test pattern description back from vector form
    optimised_pattern = {
        (x, y): 1 if best_pattern[y, x] == input_max else -1
        for (x, y) in test_pattern.pattern
        if best_pattern[y, x] in (input_min, input_max)
    }
    
    # Convert out of Numpy int type (which cannot be serialised into JSON
    # later)
    best_decoded_value = int(best_decoded_value)
    
    return OptimisedTestPatternSpecification(
        target=test_pattern.target,
        pattern=optimised_pattern,
        pattern_translation_multiple=test_pattern.pattern_translation_multiple,
        target_translation_multiple=test_pattern.target_translation_multiple,
        quantisation_index=best_qi,
        decoded_value=best_decoded_value,
        num_search_iterations=total_iterations,
    )