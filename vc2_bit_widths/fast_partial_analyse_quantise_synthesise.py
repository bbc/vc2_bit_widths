"""
:py:mod:`vc2_bit_widths.fast_partial_analyse_quantise_synthesise`: Fast single value encode, quantise and decode
================================================================================================================

This module implements a relatively efficient (for Python) implementation of an
encode, quantise and decode cycle where only a single decoder output (or
intermediate) value is computed.

Example usage
-------------

The example below demonstrates how the
:py:func:`FastPartialAnalyseQuantiseSynthesise` function may be used to compute
the output of a VC-2 decoder intermediate value for a given picture::

    >>> import numpy as np
    >>> from vc2_data_tables import (
    ...     LIFTING_FILTERS,
    ...     WaveletFilters,
    ...     QUANTISATION_MATRICES,
    ... )
    >>> from vc2_bit_widths.vc2_filters import (
    ...     synthesis_transform,
    ...     make_variable_coeff_arrays,
    ... )
    >>> from vc2_bit_widths.fast_partial_analyse_quantise_synthesise import (
    ...     FastPartialAnalyseQuantiseSynthesise,
    ... )
    
    >>> # Codec parameters
    >>> wavelet_index = WaveletFilters.le_gall_5_3
    >>> wavelet_index_ho = WaveletFilters.le_gall_5_3
    >>> dwt_depth = 2
    >>> dwt_depth_ho = 0
    
    >>> # Quantisation settings
    >>> quantisation_indices = list(range(64))
    >>> quantisation_matrix = QUANTISATION_MATRICES[(
    ...     wavelet_index,
    ...     wavelet_index_ho,
    ...     dwt_depth,
    ...     dwt_depth_ho,
    ... )]
    
    >>> # A test picture
    >>> width = 1024  # NB: Must be appropriate multiple for
    >>> height = 512  # filter depth chosen!
    >>> picture = np.random.randint(-512, 511, (height, width))
    
    >>> h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    >>> v_filter_params = LIFTING_FILTERS[wavelet_index]
    
    >>> # Construct synthesis expression for target synthesis value
    >>> level = 1
    >>> array_name = "L"
    >>> tx, ty = 200, 100
    >>> _, intermediate_synthesis_arrays = synthesis_transform(
    ...     h_filter_params,
    ...     v_filter_params,
    ...     dwt_depth,
    ...     dwt_depth_ho,
    ...     make_variable_coeff_arrays(dwt_depth, dwt_depth_ho)
    ... )
    >>> synthesis_exp = intermediate_synthesis_arrays[(level, array_name)][tx, ty]
    
    >>> # Setup codec
    >>> codec = FastPartialAnalyseQuantiseSynthesise(
    ...     h_filter_params,
    ...     v_filter_params,
    ...     dwt_depth,
    ...     dwt_depth_ho,
    ...     quantisation_matrix,
    ...     quantisation_indices,
    ...     synthesis_exp,
    ... )
    
    >>> # Perform the analysis/quantise/synthesis transforms
    >>> codec.analyse_quantise_synthesise(picture.copy())  # NB: corrupts array!


API
---

.. autoclass:: FastPartialAnalyseQuantiseSynthesise
    :members:

"""

import numpy as np

from collections import defaultdict

from vc2_bit_widths.pyexp import Argument

from vc2_bit_widths.quantisation import quant_factor, quant_offset

from vc2_bit_widths.fast_partial_analysis_transform import (
    fast_partial_analysis_transform,
)


__all__ = [
    "FastPartialAnalyseQuantiseSynthesise",
]


def get_transform_coeffs_used_by_synthesis_exp(exp):
    """
    Enumerate the transform coefficients used by a
    :py:class:`~vc2_bit_widths.pyexp.PyExp` describing a synthesis
    transformation.
    
    Parameters
    ==========
    exp : :py:class:`~vc2_bit_widths.pyexp.PyExp`
        This expression should be defined in terms of a single
        :py:class:`~vc2_bit_widths.pyexp.Argument` which is subscripted in the
        form ``arg[level][orient][x, y]``.
    
    Returns
    =======
    transform_coeffs_used : [(level, orient, x, y), ...]
        An enumeration (in no particular order) of the transform coefficients
        used in the expression.
    """
    f = exp.make_function()
    
    # Dirty trick: use default dict to track which transform coefficients are
    # actually accessed by the function
    mock_coeffs = defaultdict(  # level
        lambda: defaultdict(  # orient
            lambda: defaultdict(  # (x, y)
                lambda: 0
            )
        )
    )
    f(mock_coeffs)
    
    return [
        (level, orient, x, y)
        for level, orients in mock_coeffs.items()
        for orient, coords in orients.items()
        for x, y in coords
    ]


def exp_coeff_nested_dicts_to_list(exp):
    """
    Transform a synthesis transform describing
    :py:class:`~vc2_bit_widths.pyexp.PyExp` such that instead of taking a
    nested dictionary of coefficient value arrays it takes a single list of
    coefficient values.
    
    This transformation can speed up the execution of the generated function by
    around 3x while also eliminating the need to construct nested dictionaries
    of values as arguments (also a substantial performance saving).
    
    Parameters
    ==========
    exp : :py:class:`~vc2_bit_widths.pyexp.PyExp`
        An expression defined in terms of a single
        :py:class:`~vc2_bit_widths.pyexp.Argument` which is always used
        subscripted in the form ``arg[level][orient][x, y]``.
    
    Returns
    =======
    exp : :py:class:`~vc2_bit_widths.pyexp.PyExp`
        A modified expression which takes a single :py:class:`list` as
        argument. The values in this list should correspond to the transform
        coefficients enumerated in ``transform_coeffs_used``
    transform_coeffs_used : [(level, orient, x, y), ...]
        The transform coefficients expected by the returned expression, in the
        order they're expected.
    """
    transform_coeffs_used = get_transform_coeffs_used_by_synthesis_exp(exp)
    
    # Find the Argument in the expression
    all_argument_names = exp.get_all_argument_names()
    assert len(all_argument_names) == 1
    argument = Argument(all_argument_names[0])
    
    new_exp = exp.subs({
        argument[level][orient][x, y]: argument[i]
        for i, (level, orient, x, y) in enumerate(transform_coeffs_used)
    })
    
    return (new_exp, transform_coeffs_used)


def to_interleaved_transform_coord(dwt_depth, dwt_depth_ho, level, orient, x, y):
    """
    Given a transform coefficient subband coordinate, return the equivalent
    coordinate in an interleaved transform signal.
    
    Parameters
    ==========
    dwt_depth : int
    dwt_depth_ho : int
        The transform depth the coordinate corresponds to.
    level : int
    orient : str
    x : int
    y : int
        The subband coordinate.
    
    Returns
    =======
    x, y : int
    """
    if level == 0:
        level += 1
    
    h_level = level
    v_level = max(level, dwt_depth_ho + 1)
    
    h_depth = (dwt_depth + dwt_depth_ho + 1) - h_level
    v_depth = (dwt_depth + dwt_depth_ho + 1) - v_level
    
    h_scale = 2**h_depth
    v_scale = 2**v_depth
    
    h_offset = int(orient in ("H", "HL", "HH")) * 2**(h_depth - 1)
    v_offset = int(orient in ("LH", "HH")) * 2**(max(0, v_depth - 1))
    
    return (x*h_scale + h_offset, y*v_scale + v_offset)


def transform_coeffs_to_index_array(dwt_depth, dwt_depth_ho, transform_coeffs):
    """
    Convert a list of transform coefficient (subband) coordinates into an index
    array for an interleaved array of transform coefficients.
    
    Parameters
    ==========
    dwt_depth : int
    dwt_depth_ho : int
        The transform depth used.
    transform_coeffs : [(level, orient, x, y), ...]
        The list of transform coefficients.
    
    Returns
    =======
    index_array : (ys, xs)
        A 2D index array (tuple of two 1D arrays) giving equivalent interleaved
        coordinates for the supplied transform coefficients.
    """
    # Covert the coordinates of the coefficients used into interleaved
    # coordinate arrays
    xs, ys = zip(*(
        to_interleaved_transform_coord(
            dwt_depth,
            dwt_depth_ho,
            level,
            orient,
            x,
            y,
        )
        for (level, orient, x, y) in transform_coeffs
    ))
    
    # Create numpy slice definition suitable for indexing a 2D array of
    # interleaved transform coefficients.
    return (
        np.array(ys, dtype=np.intp),
        np.array(xs, dtype=np.intp),
    )


def make_quantisation_factor_sweep(
    quantisation_indices,
    quantisation_matrix,
    transform_coeffs,
):
    """
    Return a matrix of quantisation factors (from the quant_factor pseudocode
    function) and a matrix of quantisation offsets (from the quant_offset
    pseudocode function) where each row may be used to quantise a vector of
    transform coefficients by a particular quantisation factor.
    
    Parameters
    ==========
    quantisation_indices : [int, ...]
        An iterable of quantisation indices to use. The first row will use the
        first index, and so on...
    quantisation_matrix : {level: {orient: int, ...}, ...}
        The VC-2 quantisation matrix to use.
    transform_coeffs : [(level, orient, x, y), , ...]
        The transform coefficients which will be quantised. (The 'x' and 'y'
        values are ignored since only the level and orientation determine the
        quantisation index to use).
    
    Returns
    =======
    quantisation_factor_matrix : :py:class:`numpy.array`
        A matrix with ``len(quantisation_indices)`` rows and
        ``len(transform_coeffs)`` columns. For each entry in
        quantisation_indices, a row of quantisation factors is provided to
        be applied to a transform coefficient vector (in the order given by
        transform_coeffs).
    quantisation_offset_matrix : :py:class:`numpy.array`
        Like quantisation_factor_matrix but includes quantisation offset values
        instead.
    """
    quantisation_factor_matrix = np.array([
        [
            quant_factor(max(0, qi - quantisation_matrix[level][orient]))
            for (level, orient, x, y) in transform_coeffs
        ]
        for qi in quantisation_indices
    ])
    
    quantisation_offset_matrix = np.array([
        [
            quant_offset(max(0, qi - quantisation_matrix[level][orient]))
            for (level, orient, x, y) in transform_coeffs
        ]
        for qi in quantisation_indices
    ])
    
    return (quantisation_factor_matrix, quantisation_offset_matrix)


def apply_quantisation_sweep(
    quantisation_factor_matrix,
    quantisation_offset_matrix,
    transform_coeff_vector,
):
    """
    Given a vector of transform coefficients, quantise then
    dequantise every value by the quantisation factors given in the supplied
    matrix.
    
    Parameters
    ==========
    quantisation_factor_matrix : :py:class:`numpy.array`
        An matrix of quantisation factors to use during quantisation, as
        returned by :py:func:`make_quantisation_factor_sweep`. Each row gives a
        set of quantisation factors to the transform coefficient vector.
    quantisation_offset_matrix : :py:class:`numpy.array`
        An matrix of quantisation offsets to use during quantisation, as
        returned by :py:func:`make_quantisation_factor_sweep`. Each row gives a
        set of quantisation offsets to the transform coefficient vector.
    transform_coeff_vector : :py:class:`numpy.array`
        A vector of transform coefficients, in an order compatible with the
        quantisation factor matrix..
    
    Returns
    =======
    matrix : :py:class:`numpy.array`
        A matrix with the same shape as quantisation_factor_matrix. Each row of
        this matrix contains the transform coefficient values after
        quantisation and dequantisation with a different quantisation index.
    """
    transform_coeff_vector = transform_coeff_vector.reshape((1, -1))
    
    # Quantisation/dequantisation is run entirely on positive values to save
    # special cases for negative rounding behaviour. The sign is reintroduced
    # later.
    signs = np.sign(transform_coeff_vector)
    values = np.abs(transform_coeff_vector)
    
    # Repeat the transform coeff vector once for every quantisation index
    num_quantisation_indices = quantisation_factor_matrix.shape[0]
    value_matrix = np.tile(
        values,
        (num_quantisation_indices, 1),
    )
    
    # Quantise every transform coefficient by every quantisation index
    value_matrix *= 4
    value_matrix //= quantisation_factor_matrix
    
    
    # Dequantise, ignoring special treatment for zeros which will be reset
    # later.
    zeros = value_matrix == 0
    value_matrix *= quantisation_factor_matrix
    value_matrix += quantisation_offset_matrix
    value_matrix += 2
    value_matrix >>= 2
    
    # Apply special cases...
    value_matrix[zeros] = 0
    value_matrix *= signs
    
    return value_matrix


class FastPartialAnalyseQuantiseSynthesise(object):
    r"""
    An object which encodes a picture, quantises the transform coefficients
    at a range of different quantisation indices, and then decodes a single
    decoder output or intermediate value at each quantisation level.
    
    For valid results to be produced by this class, both the analysis and
    synthesis processes must be completely edge-effect free. If this condition
    is not met, behaviour is undefined and may crash or produce invalid
    results.
    
    Parameters
    ==========
    h_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
    v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
        Horizontal and vertical filter *synthesis* (not analysis!) filter
        parameters (e.g. from :py:data:`vc2_data_tables.LIFTING_FILTERS`)
        defining the wavelet transform types to use.
    dwt_depth : int
    dwt_depth_ho : int
        Transform depths for 2D and horizontal-only transforms.
    quantisation_matrix : {level: {orient: int, ...}, ...}
        The VC-2 quantisation matrix to use (e.g. from
        :py:data:`vc2_data_tables.QUANTISATION_MATRICES`).
    quantisation_indices : [qi, ...]
        A list of quantisation indices to use during the quantisation step.
    synthesis_exp : :py:class:`~vc2_bit_widths.pyexp.PyExp`
        A :py:class:`~vc2_bit_widths.pyexp.PyExp` object which describes
        the partial synthesis (decoding) process to be performed.
        
        This expression will typically be an output of a
        :py:func:`vc2_bit_widths.vc2_filters.synthesis_transform` which has
        been fed coefficient :py:class:`~vc2_bit_widths.pyexp.Argument`\ s
        produced by
        :py:func:`vc2_bit_widths.vc2_filters.make_variable_coeff_arrays`.
        
        This expression must only rely on transform coefficients known to
        be edge-effect free in the encoder's output.
    """
    
    def __init__(
        self,
        h_filter_params,
        v_filter_params,
        dwt_depth,
        dwt_depth_ho,
        quantisation_matrix,
        quantisation_indices,
        synthesis_exp,
    ):
        self._h_filter_params = h_filter_params
        self._v_filter_params = v_filter_params
        self._dwt_depth = dwt_depth
        self._dwt_depth_ho = dwt_depth_ho
        
        self._quantisation_indices = quantisation_indices
        
        exp, transform_coeffs_used = exp_coeff_nested_dicts_to_list(synthesis_exp)
        
        self._decode = exp.make_function("_decode")
        
        # Get an index array to extract just the transform coeffs used by the
        # expression from an interleaved transform data array.
        self._transform_coeff_index_array = transform_coeffs_to_index_array(
            dwt_depth,
            dwt_depth_ho,
            transform_coeffs_used,
        )
        
        # Create a quantisation factor matrix to use to apply all of the
        # desired quantisation levels to the transform values.
        (
            self._quantisation_factor_matrix,
            self._quantisation_offset_matrix,
        ) = make_quantisation_factor_sweep(
            self._quantisation_indices,
            quantisation_matrix,
            transform_coeffs_used,
        )
    
    @property
    def quantisation_indices(self):
        """The quantisation indices used during decoding."""
        return self._quantisation_indices
    
    def analyse_quantise_synthesise(self, picture):
        """
        Encode, quantise (at multiple levels) and then decode the supplied
        picture. The decoded result at each quantisation level will be
        returned.
        
        Parameters
        ==========
        picture : :py:class:`numpy.array`
            The picture to be encoded. Will be corrupted as a side effect of
            calling this method.
            
            Width must be a multiple of ``2**(dwt_depth+dwt_depth_ho)`` pixels
            and height a multiple of ``2**dwt_depth`` pixels.
            
            Dimensions must also be sufficient that all transform coefficients
            used by the ``synthesis_exp`` will be edge-effect free.
        
        Returns
        =======
        decoded_values : [value, ...]
            The decoded value at for each quantisation index in
            ``quantisation_indices``.
        """
        # Encode
        fast_partial_analysis_transform(
            self._h_filter_params,
            self._v_filter_params,
            self._dwt_depth,
            self._dwt_depth_ho,
            picture,
        )
        
        # Extract only the coeffs required for decoding
        transform_coeffs_vector = picture[self._transform_coeff_index_array]
        
        # Quantise (by every quantisation level at once)
        all_quantised_transform_coeff_vectors = apply_quantisation_sweep(
            self._quantisation_factor_matrix,
            self._quantisation_offset_matrix,
            transform_coeffs_vector,
        )
        
        # Decode at each quantisation level
        return np.array([
            self._decode(coeff_vector.tolist())
            for coeff_vector in all_quantised_transform_coeff_vectors
        ])
