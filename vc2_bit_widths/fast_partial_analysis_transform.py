"""
:py:mod:`vc2_bit_widths.fast_partial_analysis_transform`: Wavelet analysis transform
====================================================================================

This module contains a :py:mod:`numpy`-based implementation of a VC-2-like
integer lifting wavelet analysis (encoding) transform. This implementation is
approximately 100x faster than executing the equivalent VC-2 pseudocode under
Python. It also optionally may be used to extract intermediate results from the
codec.

This module is not indended as a general-purpose encoder implementation but
rather for rapid evaluation of test patterns. Since test patterns are
edge-effect free, this implementation does not implement VC-2's edge effect
handling behaviour -- hence the 'partial' part of the module name. Output
values which would have been effected by edge effects will contain nonsense
values.

.. note::

    This Python+numpy filter is still extremely slow compared to any reasonable
    native software implementation. However, being pure Python, it is more
    portable and therefore useful in this application.

Example usage
-------------

The example below demonstrates how the
:py:func:`fast_partial_analysis_transform` function may be used to perform an
analysis transform on an example picture::

    >>> import numpy as np
    >>> from vc2_data_tables import LIFTING_FILTERS, WaveletFilters
    >>> from vc2_bit_widths.fast_partial_analysis_transform import (
    ...     fast_partial_analysis_transform,
    ... )
    
    >>> # Codec parameters
    >>> wavelet_index = WaveletFilters.le_gall_5_3
    >>> wavelet_index_ho = WaveletFilters.le_gall_5_3
    >>> dwt_depth = 2
    >>> dwt_depth_ho = 0
    
    >>> # A test picture
    >>> width = 1024  # NB: Must be appropriate multiple for
    >>> height = 512  # filter depth chosen!
    >>> picture = np.random.randint(-512, 511, (height, width))
    
    >>> h_filter_params = LIFTING_FILTERS[wavelet_index_ho]
    >>> v_filter_params = LIFTING_FILTERS[wavelet_index]
    
    >>> # Perform the analysis transform (in place)
    >>> transform_coeffs = fast_partial_analysis_transform(
    ...     h_filter_params,
    ...     v_filter_params,
    ...     dwt_depth,
    ...     dwt_depth_ho,
    ...     picture,
    ... )


API
---

.. autofunction:: fast_partial_analysis_transform

"""

import numpy as np

from collections import namedtuple

from vc2_bit_widths.vc2_filters import convert_between_synthesis_and_analysis

from vc2_data_tables import WaveletFilters, LIFTING_FILTERS, LiftingFilterTypes


__all__ = [
    "fast_partial_analysis_transform",
]


# The VC-2 wavelet transform is defined in terms of a series of lifting stages.
# In the description below we'll describe how a single lifting stage is
# implemented.
#
# This implementation of lifting is based on vectorised array operations which
# act on whole arrays at once. In this overview of the process we'll only show
# 1D signals for brevity. The extension to 2D is to apply the same processes to
# every row or column of a 2D signal.
#
# To explain the implementation we'll use the second lifting stage of a
# Deslauriers Dubuc (13,7) synthesis transform as a running example.
# Conceptually, this lifting stage is described by the following pseudocode on
# a signal of length 'n':
#
#     # Accumulating and weighted signal values
#     accumulator[0:n/2] = 0
#     accumulator[0:n/2] += signal[-2:n-3:2] * -1   # Tap 0
#     accumulator[0:n/2] += signal[0:n-1:2] * 9     # Tap 1
#     accumulator[0:n/2] += signal[2:n+1:2] * 9     # Tap 2
#     accumulator[0:n/2] += signal[4:n+3:2] * -1    # Tap 3
#
#     # Scaling accumulator and updating signal
#     accumulator[0:n/2] += 8
#     accumulator[0:n/2] >>= 4
#     signal[1:n:2] += accumulator[0:n/2]
#
# A practical difficulty arises from the fact that the filter relies on values
# beyond the end of the input signal, a feature of all but the Haar wavelet
# filter's lifting stages.
#
# The diagram below illustrates how signal values are accumulated into the
# accumulator array.
#
#                 -3  -2  -1   0   1   2   3   4   5   6   7   8              n-9 n-8 n-7 n-6 n-5 n-4 n-3 n-2 n-1  n  n+1 n+2
#              - + - + - + - +---+---+---+---+---+---+---+---+---+--       --+---+---+---+---+---+---+---+---+---+ - + - + - + -
#  Signal:   ... '   '   '   |   |   |   |   |   |   |   |   |   | ...   ... |   |   |   |   |   |   |   |   |   |   '   '   ' ...
#              - + - + - + - +---+---+---+---+---+---+---+---+---+--       --+---+---+---+---+---+---+---+---+---+ - + - + - + -
#                      '-------'---+---'-------'                           '-------'---+---'-------'
#                              '-------'---+---'-------'                           '-------'---+---'-------'
#                                  |   '-------'---+---'-------'                       |   '-------'---+---'-------'
#                                  |       |   '-------'---+---'-------'               |       |   '-------'---+---'-------'
#                                  |       |       |       |                           |       |       |       |
#                            +-------+-------+-------+-------+------       ------+-------+-------+-------+-------+
#  Accumulator:              |XXXXXXX|       |       |       |     ...   ...     |       |       |XXXXXXX|XXXXXXX|
#                            +-------+-------+-------+-------+------       ------+-------+-------+-------+-------+
#                                0       1       2       3       4         n/2-5   n/2-4   n/2-3   n/2-2   n/2-1
#
# The accumulator values which depend on signal values beyond the ends of the
# signal are marked with Xs in the diagram. In this case, the first and final
# two accumulator values depend on signal values beyond the ends of the signal
# array.
#
# In a real VC-2 implementation, values beyond the ends of the signal array are
# filled in using edge extension. In this module, we do not implement these
# edge effects and just skip computing these accumulator values altogether.
#
# To compute the number of accumulator values to be skipped for a given filter,
# the following formulae may be used:
#
# Let:
#     update_offset = 0   if even samples are updated and odd samples accumulated
#                     1   if odd samples are updated and even samples accumulated
#
#         e.g. update_offset = 1 for the example above
#
#     delay = the lifting filter delay as listed in Tables (15.1)-(15.6) in the
#             VC-2 specification
#
#         e.g. delay = -1 for the example above
#
#     length = the lifting filter length as listed in Tables (15.1)-(15.6) in
#              the VC-2 specification
#
#         e.g. length = 4 for the example above
#
#     first_tap = update_offset + 2*delay - 1
#     last_tap = update_offset + 2*(delay + length - 1) - 1
#
#         Which give the indices of the first and last signal array values used
#         to compute the first accumulator value.
#
#         e.g. first_tap = -2, last_tap = 4 for the example above
#
# Then:
#     acc_left_skip = -(first_tap//2)
#     acc_right_skip = last_tap//2
#
#         Which give the number of accumulator entries at the left and right
#         end of the accumulator array which should be skipped due to
#         depending on values beyond the ends of the signal.
#
#         e.g. acc_left_skip = 1, acc_right_skip = 2 for the example above
#
# When computing this subset of accumulator values, each filter tap samples a
# particular set of signal values as illustrated below:
#
#  Used by tap 0:   @       @       @       @       @    ...    @       @
#  Used by tap 1:           @       @       @       @    ...    @       @       @
#  Used by tap 2:                   @       @       @    ...    @       @       @       @
#  Used by tap 3:                           @       @    ...    @       @       @       @       @
#
#                   0   1   2   3   4   5   6   7   8              n-9 n-8 n-7 n-6 n-5 n-4 n-3 n-2 n-1 
#                 +---+---+---+---+---+---+---+---+---+--       --+---+---+---+---+---+---+---+---+---+
#  Signal:        |   |   |   |   |   |   |   |   |   | ...   ... |   |   |   |   |   |   |   |   |   |
#                 +---+---+---+---+---+---+---+---+---+--       --+---+---+---+---+---+---+---+---+---+
#                   '-------'---+---'-------'                   '-------'---+---'-------'
#                           '-------'---+---'-------'                   '-------'---+---'-------'
#                               |   '-------'---+---'-------'               |       |
#                               |       |       |                           |       |
#                 +-------+-------+-------+-------+------       ------+-------+-------+-------+-------+
#  Accumulator:   |XXXXXXX|       |       |       |     ...   ...     |       |       |XXXXXXX|XXXXXXX|
#                 +-------+-------+-------+-------+------       ------+-------+-------+-------+-------+
#                     0       1       2       3       4         n/2-5   n/2-4   n/2-3   n/2-2   n/2-1
#
# Here the values indicated by an '@' are used to compute the accumulator
# values which were not skipped. The first and last signal indices (inclusive)
# of the accumulated values for the i-th tap may be computed as follows:
#
#     tap_i_first_index = first_tap + 2*acc_left_skip + 2*i
#     tap_i_last_index = n - (2*(acc_left_skip + acc_right_skip) + update_offset + 1 - 2*i)
#
#     e.g. for the example above:
#          tap_0_first_index = 0,  tap_0_last_index = n-8
#          tap_1_first_index = 2,  tap_1_last_index = n-6
#          tap_2_first_index = 4,  tap_2_last_index = n-4
#          tap_3_first_index = 6,  tap_3_last_index = n-2
#
# Given the acc_left_skip, acc_right_skip, tap_i_first_index and
# tap_i_last_index values calculated above, we can now modify our pseudocode
# program to avoid accessing any signal values beyond the ends of the signal
# (leaving edge-effect containing samples unchanged):
#
#     # Accumulating and weighted signal values
#     accumulator[1:n/2-2] = 0
#     accumulator[1:n/2-2] += signal[0:n-7:2] * -1   # Tap 0
#     accumulator[1:n/2-2] += signal[2:n-5:2] * 9    # Tap 1
#     accumulator[1:n/2-2] += signal[4:n-3:2] * 9    # Tap 2
#     accumulator[1:n/2-2] += signal[6:n-1:2] * -1   # Tap 3
#
#     # Scaling accumulator and updating signal
#     accumulator[1:n/2-2] += 8
#     accumulator[1:n/2-2] >>= 4
#     signal[3:n-4:2] += accumulator[1:n/2-2]


LiftingStageSlices = namedtuple("LiftingStageSlices", "signal_update_slice,accumulator_slice,tap_signal_slices")
"""
A set of :py:class:`slice` objects which define the array slices to be used
when performing a lifting stage.

Parameters
==========
signal_update_slice : :py:class:`slice`
    An slice into the signal array which selects only those values which will
    be modified by this lifting stage.
accumulator_slice : :py:class:`slice`
    An slice into the accumulator array selecting only those values in the
    accumulator which will be used during this lifting stage.
tap_signal_slices : [:py:class:`slice`, ...]
    For each filter tap, a slice into the signal array giving the values to be
    weighted and accumulated.
"""


def compute_lifting_stage_slices(stage):
    r"""
    Compute array :py:class:`slice`\ s for performing the specified lifting
    operation without using out-of-range signal values.
    
    Parameters
    ==========
    stage : :py:class:`vc2_data_tables.LiftingStage`
    
    Returns
    =======
    slices : :py:class:`LiftingStageSlices`
    """
    # For walk-through of the formulae used below, see the large introductory
    # comment block above this function.
    
    if (
        stage.lift_type == LiftingFilterTypes.odd_add_even or
        stage.lift_type == LiftingFilterTypes.odd_subtract_even
    ):
        update_offset = 1
    else:
        update_offset = 0
    
    first_tap = update_offset + 2*stage.D - 1
    last_tap = update_offset + 2*(stage.D + stage.L - 1) - 1
    
    acc_left_skip = -(first_tap//2)
    acc_right_skip = last_tap//2
    
    # Note: the 'or None' idiom used below for slice end indices is due to a
    # quirk of Python's array negative indexing. If a slice ending with -1 goes
    # up to, but not including, the last value, logically the slice which
    # includes the last value would end with 0, but since this isn't negative
    # Python interprets it as meaning start of array! Instead we must use None
    # to indicate the end of the array.
    accumulator_slice = slice(acc_left_skip, -acc_right_skip or None)
    signal_update_slice = slice(
        update_offset + 2*acc_left_skip,
        -(2*acc_right_skip + 1 - update_offset) or None,
        2,
    )
    
    tap_signal_slices = []
    for i in range(stage.L):
        tap_first_index = first_tap + 2*acc_left_skip + 2*i
        tap_last_index = -(2*(acc_left_skip + acc_right_skip) + update_offset + 1 - 2*i)
        tap_signal_slices.append(slice(
            tap_first_index,
            (tap_last_index + 1) or None,
            2,
        ))
    
    return LiftingStageSlices(
        signal_update_slice,
        accumulator_slice,
        tap_signal_slices,
    )


def h_stage(signal, accumulator1, accumulator2, lifting_stage, lifting_stage_slices):
    """
    Apply a single one dimensional lifting filter pass to values in the
    horizontal dimension.
    
    Parameters
    ==========
    signal : :py:class:`numpy.array`
        The signal array to be transformed (modified in place)
    accumulator1 : :py:class:`numpy.array`
    accumulator2 : :py:class:`numpy.array`
        Working arrays for the filter. Must be the same as signal but with half
        the width.
    lifting_stage : :py:class:`vc2_data_tables.LiftingStage`
        The lifting stage filter parameters.
    lifting_stage_slices : :py:class:`LiftingStageSlices`
        The array slices computed for this lifting stage by
        :py:func:`compute_lifting_stage_slices`.
    """
    length = signal.shape[1]
    
    accumulator1[:] = 0
        
    # Accumulate a weighted sum of all filter taps
    acc_slice = lifting_stage_slices.accumulator_slice
    for weight, sig_slice in zip(lifting_stage.taps, lifting_stage_slices.tap_signal_slices):
        if weight == 1:
            # Special case: no need to scale first
            accumulator1[:, acc_slice] += signal[:, sig_slice]
        else:
            accumulator2[:, acc_slice] = signal[:, sig_slice]
            accumulator2[:, acc_slice] *= weight
            accumulator1[:, acc_slice] += accumulator2[:, acc_slice]
        
    # Shift accumulated values
    if lifting_stage.S:
        accumulator1[:, acc_slice] += 1 << (lifting_stage.S - 1)
        accumulator1[:, acc_slice] >>= lifting_stage.S
    
    # Update signal
    sig_slice = lifting_stage_slices.signal_update_slice
    if (
        lifting_stage.lift_type == LiftingFilterTypes.odd_add_even or
        lifting_stage.lift_type == LiftingFilterTypes.even_add_odd
    ):
        signal[:, sig_slice] += accumulator1[:, acc_slice]
    else:
        signal[:, sig_slice] -= accumulator1[:, acc_slice]


def v_stage(signal, accumulator1, accumulator2, lifting_stage, lifting_stage_slices):
    """
    Apply a single one dimensional lifting filter pass to values in the
    vertical dimension.
    
    Parameters
    ==========
    signal : :py:class:`numpy.array`
        The signal array to be transformed (modified in place)
    accumulator1 : :py:class:`numpy.array`
    accumulator2 : :py:class:`numpy.array`
        Working arrays for the filter. Must be the same as signal but with half
        the height.
    lifting_stage : :py:class:`vc2_data_tables.LiftingStage`
        The lifting stage filter parameters.
    lifting_stage_slices : :py:class:`LiftingStageSlices`
        The array slices computed for this lifting stage by
        :py:func:`compute_lifting_stage_slices`.
    """
    h_stage(
        signal.T,
        accumulator1.T,
        accumulator2.T,
        lifting_stage,
        lifting_stage_slices,
    )


def fast_partial_analysis_transform(
    h_filter_params,
    v_filter_params,
    dwt_depth,
    dwt_depth_ho,
    signal,
    target=None,
):
    """
    Perform a multi-level 2D analysis transform, ignoring edge effects.
    
    Parameters
    ==========
    h_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
    v_filter_params : :py:class:`vc2_data_tables.LiftingFilterParameters`
        Horizontal and vertical filter *synthesis* (not analysis!) filter
        parameters (e.g. from :py:data:`vc2_data_tables.LIFTING_FILTERS`).
    dwt_depth, dwt_depth_ho: int
        Transform depths for 2D and horizontal-only transforms.
    signal : :py:class:`numpy.array`
        The picture to be transformed. Will be transformed in-place (in
        interleaved form) but de-interleaved views will also be returned.
        
        Width must be a multiple of ``2**(dwt_depth+dwt_depth_ho)`` pixels and
        height a multiple of ``2**dwt_depth`` pixels.
    target : (level, array_name, tx, ty) or None
        If None, the complete analysis transform will be performed. If a tuple
        is given, the transform will run until the specified value has been
        computed and this value alone will be returned.
    
    Returns
    =======
    transform_coeffs : {level: {orient: :py:class:`numpy.array`, ...}, ...} or intermediate_value
        Subsampled views of the ``signal`` array (which will have been modified
        in-place).
        
        Alternatively, if the ``target`` argument is not None, the single
        intermediate value requested will be returned.
    """
    h_filter_params = convert_between_synthesis_and_analysis(h_filter_params)
    v_filter_params = convert_between_synthesis_and_analysis(v_filter_params)
    
    h_stage_slices = list(map(compute_lifting_stage_slices, h_filter_params.stages))
    v_stage_slices = list(map(compute_lifting_stage_slices, v_filter_params.stages))
    
    # Accumulators will be reshaped into the correct 2D shape on demand
    acc1 = np.empty(signal.size//2, dtype=signal.dtype)
    acc2 = np.empty(signal.size//2, dtype=signal.dtype)
    
    out = {}
    
    for level in reversed(range(1, dwt_depth_ho + dwt_depth + 1)):
        out[level] = {}
        
        if (target is not None and target[0] == level and target[1] == "Input"):
            return signal[target[3], target[2]]
        
        # Bit shift
        if h_filter_params.filter_bit_shift:
            signal <<= h_filter_params.filter_bit_shift
        
        if (target is not None and target[0] == level and target[1] == "DC"):
            return signal[target[3], target[2]]
        
        # H-lift
        hacc1 = acc1.reshape(signal.shape[0], signal.shape[1]//2)
        hacc2 = acc2.reshape(signal.shape[0], signal.shape[1]//2)
        for num, (stage, stage_slices) in enumerate(zip(h_filter_params.stages, h_stage_slices)):
            h_stage(signal, hacc1, hacc2, stage, stage_slices)
            if target is not None and target[0] == level and target[1] == ("DC'" + "'"*num):
                return signal[target[3], target[2]]
        
        if level > dwt_depth_ho:
            if target is not None and target[0] == level:
                if target[1] == "L":
                    return signal[:, 0::2][target[3], target[2]]
                if target[1] == "H":
                    return signal[:, 1::2][target[3], target[2]]
            
            # V-lift (in 2D stages only)
            vacc1 = acc1.reshape(signal.shape[0]//2, signal.shape[1])
            vacc2 = acc2.reshape(signal.shape[0]//2, signal.shape[1])
            for num, (stage, stage_slices) in enumerate(zip(v_filter_params.stages, v_stage_slices)):
                v_stage(signal, vacc1, vacc2, stage, stage_slices)
                
                if target is not None:
                    if target[0] == level and target[1] == ("L'" + "'"*num):
                        return signal[:, 0::2][target[3], target[2]]
                    if target[0] == level and target[1] == ("H'" + "'"*num):
                        return signal[:, 1::2][target[3], target[2]]
            
            # 2D Subsample
            out[level]["HL"] = signal[0::2, 1::2]
            out[level]["LH"] = signal[1::2, 0::2]
            out[level]["HH"] = signal[1::2, 1::2]
            signal = signal[0::2, 0::2]  # LL
            
            if target is not None and target[0] == level:
                if target[1] == "HL":
                    return out[level]["HL"][target[3], target[2]]
                if target[1] == "LH":
                    return out[level]["LH"][target[3], target[2]]
                if target[1] == "HH":
                    return out[level]["HH"][target[3], target[2]]
                if target[1] == "LL":
                    return signal[target[3], target[2]]
        else:
            # Horizontal-only Subsample
            out[level]["H"] = signal[:, 1::2]  # L
            signal = signal[:, 0::2]  # L
            
            if target is not None and target[0] == level:
                if target[1] == "H":
                    return out[level]["H"][target[3], target[2]]
                if target[1] == "L":
                    return signal[target[3], target[2]]
        
        # Resize accumulators to match subsampling
        acc1 = acc1[0:signal.size//2]
        acc2 = acc2[0:signal.size//2]
    
    out[0] = {}
    if dwt_depth_ho:
        out[0]["L"] = signal
    else:
        out[0]["LL"] = signal
    
    if target is None:
        return out
    else:
        raise ValueError("Target {} not part of transform!".format(target))
