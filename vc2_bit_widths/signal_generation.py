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
    strip_affine_errors,
)

from vc2_bit_widths.infinite_arrays import lcm

from vc2_bit_widths.quantisation import quant_factor


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


def partial_analysis_to_matrix(input_symbols, transform_coeff_symbols, transform_coeff_expressions):
    """
    Produce a matrix representation of a partial analysis transform.
    
    The returned :py:mod:`numpy` matrix may be used to relatively efficiently
    compute the values of a small set of transform coefficients. Where only a
    subset of the transform coefficients are required, this can be more
    efficient than using a Python-based lifting filter to perform the same
    computation.
    
    Parameters
    ==========
    input_symbols : [sym, ...]
        An enumeration of input (pixel) symbols used in the transform
        coefficient expressions. This list should contain exactly the set of
        symbols used by the transform coefficients to be computed by the
        analysis transform.
    transform_coeff_symbols : [sym, ...]
        An enumeration of the symbols representing each of the transform
        coefficients to be computed by the analysis transform.
    transform_coeff_expressions : {sym: :py;class:`~vc2_bit_widths.linexp.LinExp`, ...}
        For each symbol in transform_coeff_symbols, an expression which
        describes the function used to compute that value in terms of the
        symbols in input_symbols.
    
    Returns
    =======
    matrix : :py:class:`numpy.array`
        A matrix with ``len(input_symbols)`` columns and
        ``len(transform_coeff_symbols)`` rows. When multiplied with a vector of
        pixel values (in the order given by input_symbols), produces a vector
        with the values of the transform coefficients (in the order given by
        transform_coeff_symbols).
    """
    analysis_matrix = np.zeros(
        (
            len(transform_coeff_symbols),
            len(input_symbols),
        ),
        dtype=float
    )
    
    for transform_coeff_sym, expr in transform_coeff_expressions.items():
        row = transform_coeff_symbols.index(transform_coeff_sym)
        for input_sym, coeff in expr:
            col = input_symbols.index(input_sym)
            analysis_matrix[row, col] = coeff
    
    return analysis_matrix


def partial_synthesis_to_matrix(transform_coeff_symbols, synthesis_expression):
    """
    Produce a matrix representation of a synthesis transform producing a single
    value.
    
    The returned :py:mod:`numpy` matrix may be used to synthesise a single
    value from a set of transform coefficients. When only a single output pixel
    is of interest, this will matrix operation will be faster than a Python
    based lifting-based computation.
    
    Parameters
    ==========
    transform_coeff_symbols : [sym, ...]
        An enumeration of the symbols representing each of the transform
        coefficients used by the synthesis transform.
    synthesis_expression : :py;class:`~vc2_bit_widths.linexp.LinExp`
        An expression which describes the function used to compute the
        synthesised value in terms of the symbols in transform_coeff_symbols.
    
    Returns
    =======
    matrix : :py:class:`numpy.array`
        A row-matrix with ``len(transform_coeff_symbols)`` columns. When
        multiplied with a vector of transform coefficients (in the order given by
        input_symbols), produces a scalar with the synthesised result.
    """
    return np.array([[
        synthesis_expression[sym]
        for sym in transform_coeff_symbols
    ]], dtype=float)


def make_quantisation_factor_sweep(
    quantisation_indices,
    quantisation_matrix,
    transform_coeff_symbols,
):
    """
    Return a matrix of quantisation factors where each column may be used to
    quantise a vector of transform coefficients by a different overall
    quantisation factor.
    
    Parameters
    ==========
    quantisation_indices : [int, ...]
        An iterable of quantisation indices to use.
    quantisation_matrix : {level: {orient: int, ...}, ...}
        The VC-2 quantisation matrix to use.
    transform_coeff_symbols : [((prefix, level, orient), x, y), , ...]
        The symbol for each transform coefficient value to be quantised, with
        symbols of the type produced by
        :py:func:`~vc2_bit_widths.vc2_filters.make_symbol_coeff_arrays`.
    
    Returns
    =======
    matrix : :py:class:`numpy.array`
        A matrix with ``len(quantisation_indices)`` columns and
        ``len(transform_coeff_symbols)`` rows. For each entry in
        quantisation_indices, a column of quantisation factors is provided to
        be applied to a transform coefficient vector (in the order given by
        transform_coeff_symbols).
    """
    # A row-vector giving the quantisation factor to be used in each output
    # column
    quantisation_factors = np.array([[
        quant_factor(qi) / 4.0
        for qi in quantisation_indices
    ]])
    
    # A column-vector giving quantisation factor adjustments for each transform
    # coefficient according to the VC-2 quantisation matrix.
    quantisation_factor_adjustments = np.array([[
        4.0 / quant_factor(quantisation_matrix[level][coeff])
        for ((prefix, level, coeff), x, y) in transform_coeff_symbols
    ]]).T
    
    # A matrix of quantisation factors. Each entry corresponds with the
    # quantisation factor used for each output matrix value.
    quantisation_factor_matrix = np.clip(
        np.matmul(quantisation_factor_adjustments, quantisation_factors),
        1.0,
        None,
    )
    
    return quantisation_factor_matrix


def apply_quantisation_sweep(quantisation_factor_matrix, transform_coeff_vector):
    """
    Given a vector containing a set of transform coefficients, quantise then
    dequantise every value by the quantisation factors given in the supplied
    matrix.
    
    Parameters
    ==========
    quantisation_factor_matrix : :py:class:`numpy.array`
        An matrix of quantisation factors to use during quantisation, as
        returned by :py:func:`make_quantisation_factor_sweep`. Each column
        gives a set of quantisation factors to apply to each transform
        coefficient.
    transform_coeff_vector : :py:class:`numpy.array`
        The transform coefficients as a vector, as produced by
        :py:func:`partial_analysis_to_matrix`.
    
    Returns
    =======
    matrix : :py:class:`numpy.array`
        A matrix with the same shape as quantisation_factor_matrix. Each column
        of this matrix represents the supplied transform coefficient values as
        they would appear after quantisation and dequantisation with the
        corresponding quantisation index in quantisation_indices.
    """
    # Quantise every transform coefficient by every quantisation index
    transform_coeff_matrix = np.tile(
        np.reshape(transform_coeff_vector, (len(transform_coeff_vector), 1)),
        (1, quantisation_factor_matrix.shape[1]),
    )
    transform_coeff_matrix = np.fix(
        transform_coeff_matrix / quantisation_factor_matrix
    )
    
    # Dequantise again
    transform_coeff_matrix = np.fix(
        (transform_coeff_matrix + (np.sign(transform_coeff_matrix)*0.5)) *
        quantisation_factor_matrix
    )
    
    return transform_coeff_matrix


def find_quantisation_index_with_greatest_output_magnitude(
    input_vector,
    analysis_matrix,
    quantisation_indices,
    quantisation_factor_matrix,
    synthesis_matrix,
):
    """
    Find the quantisation index which causes the decoded value to have the
    largest magnitude.
    
    Parameters
    ==========
    input_vector : :py:class:`numpy.array`
        The vector of input pixels to be encoded using the analysis_matrix.
    analysis_matrix : :py:class:`numpy.array`
        The matrix which implements a partial analysis transform on
        input_vector. See :py:func:`partial_analysis_to_matrix`.
    quantisation_indices : [int, ...]
        The quantisation indices used to produce the quantisation_factor_matrix
        (see :py:func:`make_quantisation_factor_sweep`).
    quantisation_factor_matrix : :py:class:`numpy.array`
        A matrix of quantisation factors to apply to the transform coefficients
        encoded by the analysis_matrix. This should have one column per
        quantisation index and one row per transform coefficient. See
        :py:func:`make_quantisation_factor_sweep`.
    synthesis_matrix : :py:class:`numpy.array`
        The matrix which implements a partial synthesis transform on the
        encoded transform coefficients producing a single result. See
        :py:func:`partial_synthesis_to_matrix`.
    
    Returns
    =======
    greatest_decoded_value : float
        The decoded value with the greatest magnitude of any quantisation index
        tested.
    quantisation_index : int
        The quantisation index which produced the decoded value with the
        greatest magnitude.
    """
    transform_coeff_vector = np.matmul(analysis_matrix, input_vector)
    
    # Each column is a transform_coeff_vector quantised by a different
    # quantisation index.
    transform_coeff_matrix = apply_quantisation_sweep(
        quantisation_factor_matrix,
        transform_coeff_vector,
    )
    
    decoded_values = np.matmul(synthesis_matrix, transform_coeff_matrix)
    
    i = np.argmax(np.abs(decoded_values))
    return (decoded_values[0, i], quantisation_indices[i])


def greedy_stochastic_search(
    starting_input_vector,
    input_min,
    input_max,
    analysis_matrix,
    quantisation_indices,
    quantisation_factor_matrix,
    synthesis_matrix,
    random_state,
    added_corruptions_per_iteration,
    removed_corruptions_per_iteration,
    base_iterations,
    added_iterations_per_improvement,
):
    """
    Use a greedy stochastic search to find perturbations to the supplied test
    signal which produce larger decoded values after quantisation.
    
    At a high-level the greedy search proceeds as follows:
    
    1. Make some random changes to the input vector.
    2. See if these changes made the decoded output larger. If they did, keep
       the changes, if they didn't, discard them.
    3. Go to step 1.
    
    The algorithm uses a simple self-termination scheme to stop the search once
    a local maximum has been found. A counter is initially set to
    ``base_iterations`` and decremented after every iteration. If the iteration
    produced an larger output value, the counter is incremented by
    ``added_iterations_per_improvement``. Once the counter reaches zero, the
    search terminates and the best vector found is returned.
    
    Parameters
    ==========
    starting_input_vector : :py:class:`numpy.array`
        The vector of input pixel values to use as the starting-point for the
        search.
    input_min : number
    input_max : number
        The range for random values which may be inserted into the vector.
    analysis_matrix : :py:class:`numpy.array`
    quantisation_indices : [int, ...]
    quantisation_factor_matrix : :py:class:`numpy.array`
    synthesis_matrix : :py:class:`numpy.array`
        Parameters for
        :py:func:`find_quantisation_index_with_greatest_output_magnitude`.
    random_state : :py:class:`numpy.random.RandomState`
        The random number generator to use during the search.
    added_corruptions_per_iteration : int
        The number of vector entries to assign with a random value during each
        search attempt.
    removed_corruptions_per_iteration : int
        The number of vector entries to reset to their original value during
        each search attempt.
    base_iterations : int
        The initial number of search iterations to perform.
    added_iterations_per_improvement : int
        The number of additional search iterations to perform whenever an
        improved vector is found.
    
    Returns
    =======
    input_vector : :py:class:`numpy.array`
        The input vector which produced the decoded value with the greatest
        magnitude after quantisation.
    greatest_decoded_value : float
        The value produced by decoding the input at the worst-case quantisation
        index.
    quantisation_index : int
        The quantisation index which yielded the largest decoded value.
    """
    # Use the input with no corruption as the baseline for the search
    best_input_vector = starting_input_vector.astype(float)
    best_decoded_value, best_qi = find_quantisation_index_with_greatest_output_magnitude(
        best_input_vector,
        analysis_matrix,
        quantisation_indices,
        quantisation_factor_matrix,
        synthesis_matrix,
    )
    
    # Pre-allocate a working copy of the input vector
    cur_input_vector = np.empty_like(best_input_vector)
    
    iterations_remaining = base_iterations
    while iterations_remaining > 0:
        iterations_remaining -= 1
        
        cur_input_vector[:] = best_input_vector
        
        # Restore a random set of indices in the vector to their original value
        reset_indices = random_state.randint(
            0, len(cur_input_vector),
            removed_corruptions_per_iteration,
        )
        cur_input_vector[reset_indices] = starting_input_vector[reset_indices]
        
        # Corrupt a random set of indices in the vector
        corrupt_indices = random_state.randint(
            0, len(cur_input_vector),
            added_corruptions_per_iteration,
        )
        cur_input_vector[corrupt_indices] = random_state.choice(
            (input_min, input_max),
            added_corruptions_per_iteration,
        )
        
        # Find the largest decoded value for the newly modified input vector
        cur_decoded_value, cur_qi = find_quantisation_index_with_greatest_output_magnitude(
            cur_input_vector,
            analysis_matrix,
            quantisation_indices,
            quantisation_factor_matrix,
            synthesis_matrix,
        )
        
        # Keep the input vector iff it is an improvement on the previous best
        if abs(cur_decoded_value) > abs(best_decoded_value):
            best_input_vector[:] = cur_input_vector
            best_decoded_value = cur_decoded_value
            best_qi = cur_qi
            
            iterations_remaining += added_iterations_per_improvement
    
    return best_input_vector, best_decoded_value, best_qi


def improve_synthesis_maximising_signal(
    test_signal,
    analysis_transform_coeff_arrays,
    synthesis_target_expression,
    input_min,
    input_max,
    max_quantisation_index,
    quantisation_matrix,
    random_state,
    number_of_searches,
    added_corruptions_per_iteration,
    removed_corruptions_per_iteration,
    base_iterations,
    added_iterations_per_improvement,
):
    """
    Improve a test signal generated by
    :py:func:`make_synthesis_maximising_signal` using repeated greedy
    stochastic search.
    
    The basic :py:func:`make_synthesis_maximising_signal` function simply uses
    a heuristic to produce a patterns likely to produce extreme values in the
    general case. By contrast this function attempts to more directly optimise
    the test pattern for particular input signal ranges and quantiser
    configurations.
    
    Parameters
    ==========
    test_signal : {(x, y): polarity, ...}
        The test signal to improve.
    analysis_transform_coeff_arrays : {level: {orient: :py:class:`~vc2_bit_widths.infinite_arrays.InfiniteArray`, ...}}
        The final output arrays from a compatible analysis filter for the
        synthesis filter whose values will be maximised. These should be
        provided as a nested dictionary of the form produced by
        :py:func:`~vc2_bit_widths.vc2_filters.analysis_transform`.
    synthesis_target_expression : :py:class:`~vc2_bit_widths.linexp.LinExp`
        An expression describing the decoding process for the target value.
    input_min : int
    input_max : int
        The minimum and maximum value which may be used in the test signal.
    max_quantisation_index : int
        The maximum quantisation index to use. This should be set high enough
        that at the highest quantisation level all transform coefficients are
        quantised to zero.
    quantisation_matrix : {level: {orient: int, ...}, ...}
        The VC-2 quantisation matrix to use.
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
    improved_test_signal : {(x, y), value, ...}
        The improved test signal found during the search. Values (not
        polarities) are given in this dictionary and will be in the range
        input_min to input_max.
    quantisation_index : int
        The quantisation index which produces the most extreme value in the
        target for this test signal.
    approximate_decoded_value : float
        An approximation of the value that this test signal will produce in the
        target decoder value when the specified quantisation index is used for
        all slices.
        
        .. note::
        
            For performance reasons, a real VC-2 encoder and decoder are not
            used. Instead, an optimised floating-point approximation is used.
            As such the expected value is only an approximation of the output
            of a real VC-2 decoder.
    """
    # This function essentially consists of three steps:
    #
    # 1. Convert the test signal and encoder/decoder/quantiser descriptions
    #    into matrix form suitable for greedy_stochastic_search()
    #
    # 2. Run greedy_stochastic_search a few times.
    #
    # 3. Convert improved test signal description from greedy_stochastic_search
    #    into a more useful form.
    
    # Remove error/constant terms in synthesis expression: these will not be
    # included in the evaluation
    synthesis_target_expression = strip_affine_errors(synthesis_target_expression)
    synthesis_target_expression -= synthesis_target_expression[None]
    
    # Collect all of the input/transform coeff symbols and transform
    # coefficient expressions used for the target
    # value
    input_symbols = set()
    transform_coeff_symbols = set()
    transform_coeff_expressions = {}
    for coeff_sym in synthesis_target_expression.symbols():
        transform_coeff_symbols.add(coeff_sym)
        
        (_, level, orient), x, y = coeff_sym
        coeff_expr = analysis_transform_coeff_arrays[level][orient][x, y]
        
        # Remove error/constant terms in the analysis expressions too
        coeff_expr = strip_affine_errors(coeff_expr)
        coeff_expr -= coeff_expr[None]
        
        transform_coeff_expressions[coeff_sym] = coeff_expr
        
        for input_sym in coeff_expr.symbols():
            input_symbols.add(input_sym)
    
    # Sort the symbols to ensure repeatable behaviour
    input_symbols = sorted(input_symbols)
    transform_coeff_symbols = sorted(transform_coeff_symbols)
    
    # Compile the analysis and synthesis filters for the target value into
    # matrices for evaluation during greedy random search
    analysis_matrix = partial_analysis_to_matrix(
        input_symbols,
        transform_coeff_symbols,
        transform_coeff_expressions,
    )
    synthesis_matrix = partial_synthesis_to_matrix(
        transform_coeff_symbols,
        synthesis_target_expression,
    )
    
    # Compile the quantisation indices into a matrix of quantisation factors to
    # perform bulk quantisation of the transform coefficients
    quantisation_indices = list(range(max_quantisation_index + 1))
    quantisation_factor_matrix = make_quantisation_factor_sweep(
        quantisation_indices,
        quantisation_matrix,
        transform_coeff_symbols,
    )
    
    # Convert the test signal into vector form (to allow
    # encoding/quantisation/decoding using the above matrices)
    base_input_vector = np.array([
        0
        if test_signal.get((x, y), 0) == 0 else
        input_max
        if test_signal[(x, y)] > 0 else
        input_min
        for prefix, x, y in input_symbols
    ])
    
    # Run the greedy search, keeping the best result.
    best_input_vector = None
    best_decoded_value = None
    best_qi = None
    assert number_of_searches >= 1
    for _ in range(number_of_searches):
        new_input_vector, new_decoded_value, new_qi = greedy_stochastic_search(
            base_input_vector,
            input_min,
            input_max,
            analysis_matrix,
            quantisation_indices,
            quantisation_factor_matrix,
            synthesis_matrix,
            random_state,
            added_corruptions_per_iteration,
            removed_corruptions_per_iteration,
            base_iterations,
            added_iterations_per_improvement,
        )
        
        if (
            best_decoded_value is None or
            abs(new_decoded_value) > abs(best_decoded_value)
        ):
            best_input_vector = new_input_vector
            best_decoded_value = new_decoded_value
            best_qi = new_qi
    
    # Convert test signal description back from vector form
    improved_test_signal = dict(zip(
        ((x, y) for (prefix, x, y) in input_symbols),
        map(int, best_input_vector),
    ))
    
    return improved_test_signal, best_decoded_value, best_qi
