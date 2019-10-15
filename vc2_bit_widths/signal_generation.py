"""
Test Signal Generation
======================

The routines in this module are designed to produce test pictures and signals
for VC-2 encoders and decoders which produce near-maximum magnitude values in
the target.

Due to non-linearities in VC-2's filters (i.e. rounding and quantisation), the
test signals generated are not true worst-case signals but rather a 'best
effort' to get close to the worst case. Encoder test signals will tend to be
very close to worst-case signals while decoder signals are likely to be modest
under-estimates. Nevertheless, these signals are likely to have value ranges
well above real picture signals.

"""

from functools import reduce

from fractions import Fraction

import numpy as np

from vc2_bit_widths.linexp import (
    LinExp,
    AAError,
)

from vc2_bit_widths.fast_partial_analyse_quantise_synthesise import (
    FastPartialAnalyseQuantiseSynthesise,
)


def get_maximising_inputs(expression):
    """
    Find the symbol value assignment which maximises the provided expression.
    
    Parameters
    ==========
    expression : :py:class:`~vc2_bit_widths.linexp.LinExp`
        The expression whose value is to be maximised.
    
    Returns
    =======
    maximising_symbol_assignment : {sym: +1 or -1, ...}
        A dictionary giving the polarity of a maximising assignment to each
        non-:py:class:`~vc2_bit_widths.affine_arithmetic.Error` term in the
        expression.
    """
    return {
        sym: +1 if coeff > 0 else -1
        for sym, coeff in LinExp(expression)
        if sym is not None and not isinstance(sym, AAError)
    }


def make_analysis_maximising_signal(input_array, target_array, tx, ty):
    """
    Create a test picture which maximises a value within an intermediate/final
    output of an analysis filter.
    
    .. note::
        
        In lossless coding modes, test signals which maximise a given value in
        the encoder also maximise the corresponding value in the decoder.
        Consequently this function may also be used to (indirectly) produce
        lossless decoder test signals.
    
    .. warning::
        
        The returned test signal is designed to maximise a real-valued
        implementation of the target filter. Though it is likely that this
        signal also maximises integer-based implementations (such as those used
        by VC-2) it is not guaranteed.
    
    Parameters
    ==========
    input_array : :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`
        The input array to the analysis filter.
    target_array : :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`
        An intermediate or final output array produced by the analysis filter
        within which a value should be maximised.
    tx, ty : int
        The index of the value within the target_array which is to be
        maximised.
    
    Returns
    =======
    test_signal : {(x, y): +1 or -1, ...}
        The input pixel coordinates and polarities which define the test
        pattern.
        
        Pixels which should be set to their maximum value are specified as +1,
        pixels to be set to their minimum value are specified as -1. All other
        pixels may be set arbitrarily.
        
        All pixel coordinates will be non-negative.
        
        For a test pattern which *minimises* the target value, invert this test
        pattern.
    
    tx, ty : int
        The coordinates of the value in the target_array which will be
        maximised by the returned test signal and may differ from the supplied
        tx and ty coordinates. The returned coordinates will be the
        top-left-most coordinates which do not require a pixel at a negative
        coordinate in the test signal.
    
    mx, my : int
        ('Multiple-X'/'Multiple-Y') The multiple by which the test signal may
        be translated while still eliciting an equivalent filter response.
    
    tmx, tmy : int
        ('Target-Multiple-X'/'Target-Multiple-Y') For a translation of the test
        signal by mx and my, the target value's coordinates will be translated
        by this amount.
    """
    test_signal = {
        (x, y): polarity
        for (prefix, x, y), polarity in get_maximising_inputs(
            target_array[tx, ty]
        ).items()
    }
    
    xs, ys = zip(*test_signal)
    min_x = min(xs)
    min_y = min(ys)
    
    # Find the multiple by which test signal coordinates must be translated to
    # achieve equivalent filter behaviour
    tmx, tmy = target_array.period
    mx, my = target_array.relative_step_size_to(input_array)
    assert mx.denominator == my.denominator == 1
    mx = int(mx) * tmx
    my = int(my) * tmy
    
    translate_steps_x = min_x // mx
    translate_steps_y = min_y // my
    
    test_signal = {
        (x - (translate_steps_x * mx), y - (translate_steps_y * my)): polarity
        for (x, y), polarity in test_signal.items()
    }
    
    tx -= translate_steps_x * tmx
    ty -= translate_steps_y * tmy
    
    return test_signal, tx, ty, mx, my, tmx, tmy


def make_synthesis_maximising_signal(
    analysis_input_array,
    analysis_transform_coeff_arrays,
    synthesis_target_array,
    synthesis_output_array,
    tx, ty,
):
    """
    Create a test picture which, after lossy encoding, is likely to maximise an
    intermediate/final value of the synthesis filter.
    
    .. warning::
        
        Because (heavy) lossy VC-2 encoding is a non-linear process, finding
        encoder inputs which maximise the decoder output is not feasible in
        general. This function uses a simple heuristic to attempt to achieve
        this goal but cannot provide any guarantees about the extent to which
        it succeeds.
    
    Parameters
    ==========
    analysis_input_array : :py:class:`~vc2_bit_widths.infinite_arrays.SymbolArray`
        The input array to a compatible analysis filter for the synthesis
        filter whose values will be maximised.
    analysis_transform_coeff_arrays : {level: {orient: :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`, ...}}
        The final output arrays from a compatible analysis filter for the
        synthesis filter whose values will be maximised. These should be
        provided as a nested dictionary of the form produced by
        :py:func:`~vc2_bit_widths.vc2_filters.analysis_transform`.
    synthesis_target_array : :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`
        The intermediate or final output array produced by the synthesis filter
        within which a value should be maximised.
    synthesis_output_array : :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`
        The output array for the synthesis filter.
    tx, ty : int
        The index of the value within the synthesis_target_array which is to be
        maximised.
    
    Returns
    =======
    test_signal : {(x, y): +1 or -1, ...}
        The input pixel coordinates and polarities which define the test
        pattern. This pattern should be encoded by a VC-2 encoder and the
        encoded signal presented to the decoder under test. The pattern is
        likely to produce more extreme values in the target decoder value when
        the encoder is configured to use very low bit-rates.
        
        Pixels which should be set to their maximum value are specified as +1,
        pixels to be set to their minimum value are specified as -1. All other
        pixels may be set arbitrarily.
        
        All pixel coordinates will be non-negative.
        
        For a test pattern which *minimises* the target value, invert this test
        pattern.
    
    tx, ty : int
        The coordinates of the value in the target_array which will be
        maximised by the returned test signal and may differ from the supplied
        tx and ty coordinates. The returned coordinates will be the
        top-left-most coordinates which do not require a pixel at a negative
        coordinate in the test signal.
    
    mx, my : int
        ('Multiple-X'/'Multiple-Y') The multiple by which the test signal may
        be translated while still eliciting an equivalent filter response.
    
    tmx, tmy : int
        ('Target-Multiple-X'/'Target-Multiple-Y') For a translation of the test
        signal by mx and my, the target value's coordinates will be translated
        by this amount.
    """
    # Enumerate the transform coefficients which maximise the target value
    # {(level, orient, x, y): coeff, ...}
    target_transform_coeffs = {
        (sym[0][1], sym[0][2], sym[1], sym[2]): coeff
        for sym, coeff in synthesis_target_array[tx, ty]
        if sym is not None and not isinstance(sym, AAError)
    }
    
    # Find the input pixels which (in the absence of quantisation) contribute
    # to the target value. In theory it will be sufficient to maximise/minimise
    # these to achieve a maximum value in the target.
    directly_contributing_input_expr = LinExp(0)
    for (level, orient, x, y), transform_coeff in target_transform_coeffs.items():
        directly_contributing_input_expr += (
            analysis_transform_coeff_arrays[level][orient][x, y] * transform_coeff
        )
    directly_contributing_input_pixels = {
        (x, y): coeff
        for (prefix, x, y), coeff in get_maximising_inputs(
            directly_contributing_input_expr
        ).items()
    }
    
    # As well as setting the pixels which directly contribute to maximising the
    # target (see above), we will also set other nearby pixels which contribute
    # to the same transform coefficients. In the absence of quantisation this
    # will have no effect on the target value, however quantisation can cause
    # the extra energy in the transform coefficients to 'leak' into the target
    # value.
    #
    # A simple greedy approach is used to maximising all of the coefficient
    # magnitudes simultaneously: priority is given to transform coefficients
    # with the greatest weight.
    test_signal = {}
    for (level, orient, cx, cy), transform_coeff in sorted(
        target_transform_coeffs.items(),
        # NB: Key includes (level, orient, x, y) to break ties and ensure
        # deterministic output
        key=lambda loc_coeff: (abs(loc_coeff[1]), loc_coeff[0]),
    ):
        for (prefix, px, py), pixel_coeff in get_maximising_inputs(
            analysis_transform_coeff_arrays[level][orient][cx, cy]
        ).items():
            test_signal[px, py] = pixel_coeff * (1 if transform_coeff > 0 else -1)
    
    # The greatest priority, however, must be given to the pixels which
    # directly control the target value!
    test_signal.update(directly_contributing_input_pixels)
    
    # The test signal may contain negative pixel coordinates. To be useful, it
    # must be translated to a position implementing the same filter but which
    # is free from negative pixel coordinates.
    #
    # Find the multiples by which test signal and target array coordinates must
    # be translated to still maximise equivalent transform coefficients.
    tmx, tmy = synthesis_target_array.period
    mx, my = synthesis_output_array.relative_step_size_to(synthesis_target_array)
    mx = (1 / mx) * tmx
    my = (1 / my) * tmy
    assert mx.denominator == my.denominator == 1
    mx = int(mx)
    my = int(my)
    
    # Translate the test signal accordingly
    xs, ys = zip(*test_signal)
    min_x = min(xs)
    min_y = min(ys)
    
    translate_steps_x = min_x // mx
    translate_steps_y = min_y // my
    
    test_signal = {
        (x - (translate_steps_x * mx), y - (translate_steps_y * my)): polarity
        for (x, y), polarity in test_signal.items()
    }
    
    tx -= translate_steps_x * tmx
    ty -= translate_steps_y * tmy
    
    return test_signal, tx, ty, mx, my, tmx, tmy


def find_quantisation_index_with_greatest_output_magnitude(
    picture,
    codec,
):
    """
    Find the quantisation index which causes the decoded value to have the
    largest magnitude.
    
    Parameters
    ==========
    picture : :py:class:`numpy.array`
        The pixels to be encoded and decoded. This array will be modified
        during the call.
    codec : :py:class:`~vc2_bit_widths.fast_partial_analyse_quantise_synthesise.FastPartialAnalyseQuantiseSynthesise`
        An encoder/quantiser/decoder object to evaluate the picture with.
    
    Returns
    =======
    greatest_decoded_value : int
        The decoded value with the greatest magnitude of any quantisation index
        tested.
    quantisation_index : int
        The quantisation index which produced the decoded value with the
        greatest magnitude.
    """
    decoded_values = codec.analyse_quantise_synthesise(picture)
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
    starting_picture,
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
    picture which produce larger decoded values after quantisation.
    
    At a high-level the greedy search proceeds as follows:
    
    1. Make some random changes to the input picture.
    2. See if these changes made the decoded output larger. If they did, keep
       the changes, if they didn't, discard them.
    3. Go to step 1.
    
    The algorithm uses a simple self-termination scheme to stop the search once
    a local maximum has been found. A counter is initially set to
    ``base_iterations`` and decremented after every iteration. If the iteration
    produced an larger output value, the counter is incremented by
    ``added_iterations_per_improvement``. Once the counter reaches zero, the
    search terminates and the best picture found is returned.
    
    Parameters
    ==========
    starting_picture : :py:class:`numpy.array`
        The picture to use as the starting-point for the search.
    search_slice
        A :py:mod:`numpy` compatible array slice specification which defines
        the region of ``starting_picture`` which should be mutated during the
        search.
    input_min : number
    input_max : number
        The value ranges for random values which may be inserted into the
        picture.
    codec : :py:class:`~vc2_bit_widths.fast_partial_analyse_quantise_synthesise.FastPartialAnalyseQuantiseSynthesise`
        An encoder/quantiser/decoder object which will be used to test each
        picture.
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
        improved picture is found.
    
    Returns
    =======
    picture : :py:class:`numpy.array`
        The perturbed picture which produced the decoded value with the
        greatest magnitude after quantisation.
    greatest_decoded_value : float
        The value produced by decoding the input at the worst-case quantisation
        index.
    quantisation_index : int
        The quantisation index which yielded the largest decoded value.
    total_iterations : int
        The number of search iterations used.
    """
    # Use the input with no corruption as the baseline for the search
    best_picture = starting_picture.copy()
    cur_picture = best_picture.copy()
    best_decoded_value, best_qi = find_quantisation_index_with_greatest_output_magnitude(
        cur_picture,
        codec,
    )
    
    # Array to use as the working array for the partial encode/decode (since
    # this corrupts the array passed in)
    working_array = np.empty_like(cur_picture)
    
    total_iterations = 0
    iterations_remaining = base_iterations
    while iterations_remaining > 0:
        total_iterations += 1
        iterations_remaining -= 1
        
        cur_picture[:] = best_picture
        
        # Slice out the picture 
        starting_slice = starting_picture[search_slice]
        cur_slice = cur_picture[search_slice]
        
        # Restore a random set of indices in the vector to their original value
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
        working_array[:] = cur_picture
        cur_decoded_value, cur_qi = find_quantisation_index_with_greatest_output_magnitude(
            working_array,
            codec,
        )
        
        # Keep the input vector iff it is an improvement on the previous best
        if abs(cur_decoded_value) > abs(best_decoded_value):
            best_picture[:] = cur_picture
            best_decoded_value = cur_decoded_value
            best_qi = cur_qi
            
            iterations_remaining += added_iterations_per_improvement
    
    return best_picture, best_decoded_value, best_qi, total_iterations


def improve_synthesis_maximising_signal(
    h_filter_params,
    v_filter_params,
    dwt_depth,
    dwt_depth_ho,
    quantisation_matrix,
    synthesis_pyexp,
    test_signal,
    input_min,
    input_max,
    max_quantisation_index,
    random_state,
    number_of_searches,
    added_corruptions_per_iteration,
    removed_corruptions_per_iteration,
    base_iterations,
    added_iterations_per_improvement,
):
    """
    'Improve' a test signal generated by
    :py:func:`make_synthesis_maximising_signal` using repeated greedy
    stochastic search.
    
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
        synthesis process for the decoder value the test signal is
        maximising/minimising. Such an expression is usually obtained from the
        use of :py:func;`~vc2_bit_widths.vc2_filters.synthesis_transform` and
        :py:func;`~vc2_bit_widths.vc2_filters.make_variable_coeff_arrays`.
    test_signal : {(x, y): polarity, ...}
        The test signal to 'improve'. This test signal must be translated such
        that no analysis or synthesis step depends on VC-2's edge extension
        behaviour. This will be the case for test signals produced by
        :py:class:`make_synthesis_maximising_signal`.
    input_min : int
    input_max : int
        The minimum and maximum value which may be used in the test signal.
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
    added_corruptions_per_iteration : int
    removed_corruptions_per_iteration : int
    base_iterations : int
    added_iterations_per_improvement : int
        Greedy stochastic search parameters. See the equivalently named
        parameters to :py:func:`greedy_stochastic_search`.
    
    Returns
    =======
    improved_test_signal : {(x, y): polarity, ...}
        The improved test signal found during the search.
    quantisation_index : int
        The quantisation index which produces the most extreme value in the
        target for this test signal.
    decoded_value : int
        The value that this test signal will produce in the target decoder
        value when the specified quantisation index is used for all slices.
    total_iterations : int
        The total number of search iterations used.
    """
    # Convert test signal into picture array with true signal levels
    xs, ys = zip(*test_signal)
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    search_slice = (slice(y0, y1+1), slice(x0, x1+1))
    
    # Round picture width/height up to be compatible with the current lifting
    # filter
    x_multiple = 2**(dwt_depth + dwt_depth_ho)
    y_multiple = 2**(dwt_depth)
    width = (((x1 + 1) + x_multiple - 1) // x_multiple) * x_multiple
    height = (((y1 + 1) + y_multiple - 1) // y_multiple) * y_multiple
    
    test_picture = np.array([
        [
            0
            if test_signal.get((x, y), 0) == 0 else
            input_min
            if test_signal.get((x, y), 0) < 0 else
            input_max
            for x in range(width)
        ]
        for y in range(height)
    ], dtype=int)
    
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
    
    # Run the greedy search several times, keeping the best result.
    best_picture = None
    best_decoded_value = None
    best_qi = None
    total_iterations = 0
    assert number_of_searches >= 1
    for _ in range(number_of_searches):
        new_picture, new_decoded_value, new_qi, new_iterations = greedy_stochastic_search(
            test_picture,
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
        
        total_iterations += new_iterations
        
        if (
            best_decoded_value is None or
            abs(new_decoded_value) > abs(best_decoded_value)
        ):
            best_picture = new_picture
            best_decoded_value = new_decoded_value
            best_qi = new_qi
    
    # Convert test picture description back from vector form
    improved_test_signal = {
        (x, y): 1 if best_picture[y, x] == input_max else -1
        for (x, y) in test_signal
        if best_picture[y, x] in (input_min, input_max)
    }
    
    return improved_test_signal, best_decoded_value, best_qi, total_iterations
