r"""
:py:mod:`vc2_bit_widths.infinite_arrays`: Infinite arrays
=========================================================

:py:class:`InfiniteArray` and its subclasses provide a way to define and
analyse VC-2 filters.


Tutorial
--------

In the worked example below we'll see how a simple LeGall (5, 3) synthesis
filter can be described using :py:class:`InfiniteArray`\ s. We'll use this
description to enumerate all of the resulting filter phases and generate
algebraic expressions for them.

Building a VC-2 filter
``````````````````````

A horizontal-only LeGall (5, 3) synthesis transform takes as input two arrays
of transform coefficients (a low-pass band, ``L``, and high-pass band, ``H``)
and produces a single output array. Lets start by defining the two input arrays::

    >>> from vc2_bit_widths.infinite_arrays import SymbolArray
    
    >>> L = SymbolArray(2, "L")
    >>> H = SymbolArray(2, "H")

The :py:class:`SymbolArray` class defines an infinite array of
:py:class:`~vc2_bit_widths.linexp.LinExp`\ s containing a single symbol of the
form ``LinExp((prefix, x, y))``. Like all :py:class:`InfiniteArray` subclasses,
we can access entries in our arrays using the usual Python subscript syntax::

    >>> L[3, 4]
    LinExp(('L', 3, 4))
    >>> H[7, -5]
    LinExp(('H', 7, -5))

Notice that the coordinate system extends to positive and negative infinity.

Next we'll create an :py:class:`InterleavedArray` which contains a horizontal
interleaving of the two inputs::

    >>> from vc2_bit_widths.infinite_arrays import InterleavedArray
    
    >>> interleaved = InterleavedArray(L, H, 0)

The new interleaved array can be accessed just as before::

    >>> interleaved[0, 0]
    LinExp(('L', 0, 0))
    >>> interleaved[1, 0]
    LinExp(('H', 0, 0))
    >>> interleaved[2, 1]
    LinExp(('L', 1, 1))
    >>> interleaved[3, 1]
    LinExp(('H', 1, 1))

Next we'll apply the two lifting stages which implement the LeGall (5, 3)
transform using a :py:class:`LiftedArray`::

    >>> from vc2_bit_widths.infinite_arrays import LiftedArray
    
    >>> from vc2_data_tables import LIFTING_FILTERS, WaveletFilters
    >>> wavelet = LIFTING_FILTERS[WaveletFilters.le_gall_5_3]
    >>> stage_0, stage_1 = wavelet.stages
    
    >>> lifted_once = LiftedArray(interleaved, stage_0, 0)
    >>> lifted_twice = LiftedArray(lifted_once, stage_1, 0)

Finally we'll use a :py:class:`RightShiftedArray` to apply the final bit shift
operation to the result::

    >>> from vc2_bit_widths.infinite_arrays import RightShiftedArray
    
    >>> output = RightShiftedArray(lifted_twice, wavelet.filter_bit_shift)

This final array now, in effect, contains a complete algebraic description of
how each entry is computed in terms of our two input arrays, ``L`` and ``H``::

    >>> output[0, 0]
    LinExp({('L', 0, 0): Fraction(1, 2), ('H', -1, 0): Fraction(-1, 8), ('H', 0, 0): Fraction(-1, 8), AAError(id=1): Fraction(-1, 4), AAError(id=2): Fraction(1, 2)})

Notice that the expression above refers to transform coefficients with negative
coordinates (e.g. ``('H', -1, 0)``). This is because the infinite dimensions of
:py:class:`InfiniteArray`\ s allows :py:class:`LiftedArray` to ignore VC-2's
filter edge effect behaviour. This makes it easier to analyse filter behaviours
without having to carefully avoid edge effected behaviour.

Filter phases
`````````````

The :py:class:`InfiniteArray` subclasses automatically keep track of the period
of each array (see :ref:`terminology`) in the :py:attr:`InfiniteArray.period`
property::

    >>> L.period
    (1, 1)
    >>> H.period
    (1, 1)
    
    >>> output.period
    (2, 1)

This makes it easy to automatically obtain an example of each filter phase in
our output::

    >>> all_output_filter_phases = [
    ...     output[x, y]
    ...     for x in range(output.period[0])
    ...     for y in range(output.period[1])
    ... ]


Detecting no-operations
```````````````````````

When performing a computationally expensive analysis on a set of filters
described by :py:class:`InfiniteArray`\ s it can be helpful to skip arrays
which just consist of interleavings and other non-transformative views of other
arrays. For this purpose, the :py:attr:`InfiniteArray.nop` property indicates
if a given array implements a no-operation (nop) and therefore may be skipped.

From the example above::

    >>> L.nop
    False
    >>> H.nop
    False
    
    >>> interleaved.nop
    True
    
    >>> lifted_once.nop
    False
    >>> lifted_twice.nop
    False
    
    >>> output.nop
    False

Here we can see that the interleaving stage is identified as having no material
effect on the values it contains.


Relative array stepping
```````````````````````

:py:class:`InfiniteArray`\ s provide a
:py:meth:`InfiniteArray.relative_step_size_to` method for calculating the
scaling relationships between arrays. This can be useful for finding array
values which don't use inputs with negative coordinates (i.e. would not be
affected by edge-effect behaviour in a real VC-2 filter).

For example, consider the top-left pixel of the filter output from before::

    >>> output[0, 0]
    LinExp({('L', 0, 0): Fraction(1, 2), ('H', -1, 0): Fraction(-1, 8), ('H', 0, 0): Fraction(-1, 8), AAError(id=1): Fraction(-1, 4), AAError(id=2): Fraction(1, 2)})

This uses the input ``('H', -1, 0)`` which has negative coordinates. Thanks to
the :py:attr:`~InfiniteArray.period` property we know that any array value with
an even-numbered X coordinate implements the same filter phase as [0, 0].

We could start trying every even-numbered X coordinate until we find an entry
without a negative ``H`` coordinate, but this would be fairly inefficient.
Instead lets use :py:meth:`~InfiniteArray.relative_step_size_to` to work out
how many steps we must move in ``output`` to move all our ``H`` coordinates
right by one step.

    >>> output.relative_step_size_to(H)
    (Fraction(1, 2), Fraction(1, 1))

This tells us that for every step we move in the X-dimension of ``output``, we
move half a step in that direction in ``H``.  Therefore, we must pick a
coordinate at least two steps to the right in ``output`` to avoid the -1
coordinate in ``H``. Moving to [2, 0] also gives an even
numbered X coordinate so we know that we're going to see the same filter phase
and so we get::

    >>> output[2, 0]
    LinExp({('L', 1, 0): Fraction(1, 2), ('H', 0, 0): Fraction(-1, 8), ('H', 1, 0): Fraction(-1, 8), AAError(id=3): Fraction(-1, 4), AAError(id=6): Fraction(1, 2)})

Which implements the same filter phase as ``output[0, 0]`` but without any
negative coordinates.

Caching
```````

All of the subclasses of :py:class:`InfiniteArray` which perform a non-trivial
operation internally cache array values when they're accessed.

The most obvious benefit of this behaviour is to impart a substantial
performance improvement for larger filter designs.

A secondary benefit applies to :py:class:`InfiniteArray`\ s containing
:py:class:`~vc2_bit_widths.linexp.LinExp`\ s. By caching the results of
operations which introduce affine arithmetic error symbols
(:py:class:`~vc2_bit_widths.linexp.AAError`), these errors can correctly
combine or cancel when that result is reused. As a result, while caching is not
essential for correctness, it can materially improve the tightness of the
bounds produced.

In some instances, this basic caching behaviour may not go far enough.  For
example, the contents of ``output[0, 0]`` and ``output[2, 0]`` are extremely
similar, consisting of essentially the same coefficients, just translated to
the right::

    >>> output[0, 0]
    LinExp({('L', 0, 0): Fraction(1, 2), ('H', -1, 0): Fraction(-1, 8), ('H', 0, 0): Fraction(-1, 8), AAError(id=1): Fraction(-1, 4), AAError(id=2): Fraction(1, 2)})
    >>> output[2, 0]
    LinExp({('L', 1, 0): Fraction(1, 2), ('H', 0, 0): Fraction(-1, 8), ('H', 1, 0): Fraction(-1, 8), AAError(id=3): Fraction(-1, 4), AAError(id=6): Fraction(1, 2)})

In principle, the cached result of ``output[0, 0]`` could be re-used (and the
coefficients suitably translated) to save the computational cost of evaluating
``output[2, 0]`` from scratch. For extremely large transforms, this
optimisation can result in substantial savings in runtime and RAM. For use in
such scenarios the :py:class:`SymbolicPeriodicCachingArray` array type is provided::

    >>> from vc2_bit_widths.infinite_arrays import SymbolicPeriodicCachingArray
    
    >>> output_cached = SymbolicPeriodicCachingArray(output, L, H)

The constructor takes the array to be cached along with all
:py:class:`SymbolArray`\ s used its definition. The new array may be accessed
as usual but with more aggressive caching taking place internally::

    >>> output_cached[0, 0]
    LinExp({('L', 0, 0): Fraction(1, 2), ('H', -1, 0): Fraction(-1, 8), ('H', 0, 0): Fraction(-1, 8), AAError(id=1): Fraction(-1, 4), AAError(id=2): Fraction(1, 2)})
    >>> output_cached[2, 0]
    LinExp({('L', 1, 0): Fraction(1, 2), ('H', 0, 0): Fraction(-1, 8), ('H', 1, 0): Fraction(-1, 8), AAError(id=1): Fraction(-1, 4), AAError(id=2): Fraction(1, 2)})

.. warning::

    Note that in the cached version of the array, the
    :py:class:`~vc2_bit_widths.linexp.AAError` terms are not unique between
    ``output_cached[0, 0]`` and ``output_cached[2, 0]``, though they should be.
    This is a result of the caching mechanism having no way to know how error
    terms should change between entries in the array. As a result,
    :py:class:`SymbolicPeriodicCachingArray` must not be used when the affine
    error terms in its output are expected to be significant.

.. note::

    :py:class:`SymbolicPeriodicCachingArray` only works with
    :py:class:`InfiniteArray`\ s defined in terms of :py:class:`SymbolArray`\
    s.


API
---

:py:class:`InfiniteArray` (base class)
``````````````````````````````````````

.. autoclass:: InfiniteArray
    :members:


Base value type arrays
``````````````````````

.. autoclass:: SymbolArray
    :no-members:

.. autoclass:: VariableArray
    :no-members:


Computation arrays
``````````````````

.. autoclass:: LiftedArray
    :no-members:

.. autoclass:: LeftShiftedArray
    :no-members:

.. autoclass:: RightShiftedArray
    :no-members:


Sampling arrays
```````````````

.. autoclass:: SubsampledArray
    :no-members:

.. autoclass:: InterleavedArray
    :no-members:


Caching arrays
``````````````

.. autoclass:: SymbolicPeriodicCachingArray
    :no-members:


"""

from vc2_bit_widths.fast_fractions import Fraction, gcd

from vc2_data_tables import LiftingFilterTypes

from vc2_bit_widths.linexp import LinExp, AAError

from vc2_bit_widths.pyexp import PyExp


__all__ = [
    "InfiniteArray",
    "SymbolArray",
    "VariableArray",
    "LiftedArray",
    "RightShiftedArray",
    "LeftShiftedArray",
    "SubsampledArray",
    "InterleavedArray",
    "SymbolicPeriodicCachingArray",
]


def lcm(a, b):
    """
    Compute the Lowest Common Multiple (LCM) of two integers.
    """
    return abs(a*b) // gcd(a, b)


class InfiniteArray(object):
    """
    An abstract base class describing an immutable infinite N-dimensional array
    of symbolic values.
    
    Subclasses should implement :py:meth:`get` to return the value at a given
    position in an array.
    
    Instances of this type may be indexed like an N-dimensional array.
    
    The 'cache' argument controls whether array values are cached or not. It is
    recommended that this argument be set to 'True' for expensive to compute
    functions or functions which introduce new error terms (to ensure error
    terms are re-used)
    """
    
    def __init__(self, ndim, cache):
        self._ndim = ndim
        self._cache = {} if cache else None
    
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys, )
        
        if len(keys) != self._ndim:
            raise TypeError("{dims}D ValueArray requires {dims}D coordinates".format(
                dims=self._ndim,
            ))
        
        if self._cache is None:
            return self.get(keys)
        elif keys in self._cache:
            return self._cache[keys]
        else:
            value = self.get(keys)
            self._cache[keys] = value
            return value
    
    def get(self, keys):
        """
        Called by :py:meth:`__getitem__` with a tuple of array indices.
        
        The array of keys is guaranteed to be a tuple with :py:data:`ndim`
        entries.
        
        Values returned by this method will be memoized (cached) by
        :py:meth:`__getitem__`. Consequently, this method will only be called
        the first time a particular array index is requested. Subsequent
        accesses will return the cached value.
        """
        raise NotImplementedError()
    
    def clear_cache(self):
        """
        Clear the cache (if enabled).
        """
        if self._cache is not None:
            self._cache.clear()
    
    @property
    def ndim(self):
        """The number of dimensions in the array."""
        return self._ndim
    
    @property
    def period(self):
        """
        Return the period of this array (see :ref:`terminology`).
        
        Returns
        =======
        period : (int, ...)
            The period of the array in each dimension.
        """
        raise NotImplementedError()
    
    @property
    def nop(self):
        """
        True if this :py:class:`InfiniteArray` is a no-operation (nop), i.e. it
        just views values in other arrays. False if some computation is
        performed.
        """
        raise NotImplementedError()
    
    def relative_step_size_to(self, other):
        r"""
        For a step along a dimension in this array, compute the equivalent step
        size in the provided array.
        
        Parameters
        ==========
        other : :py:class:`InfiniteArray`
            An array to compare the step size with. Must have been used (maybe
            indirectly) to define this array.
        
        Returns
        =======
        relative_step_size : (:py:class:`fractions.Fraction`, ...) or None
            The relative step sizes for each dimension, or None if the provided
            array was not used in the computation of this array.
        """
        raise NotImplementedError()


class SymbolArray(InfiniteArray):
    r"""
    An infinite array of :py:class:`~vc2_bit_widths.linexp.LinExp` symbols.
    
    Symbols will be identified by tuples like ``(prefix, n)`` for a one
    dimensional array, ``(prefix, n, n)`` for a two-dimensional array,
    ``(prefix, n, n, n)`` for a three-dimensional array and so-on.
    
    Example usage::

        >>> a = SymbolArray(3, "foo")
        >>> a[1, 2, 3]
        LinExp(('foo', 1, 2, 3))
        >>> a[100, -5, 0]
        LinExp(('foo', 100, -5, 0))
    
    Parameters
    ==========
    ndim : int
        The number of dimensions in the array.
    prefix : object
        A prefix to be used as the first element of every symbol tuple.
    """
    
    def __init__(self, ndim, prefix="v"):
        self._prefix = prefix
        super(SymbolArray, self).__init__(ndim, cache=False)
    
    def get(self, keys):
        return LinExp((self._prefix, ) + keys)
    
    @property
    def prefix(self):
        """The prefix used for all symbol names in this array."""
        return self._prefix
    
    @property
    def period(self):
        return (1, ) * self.ndim
    
    @property
    def nop(self):
        return False
    
    def relative_step_size_to(self, other):
        if other is self:
            return (Fraction(1), ) * self.ndim
        else:
            return None


class VariableArray(InfiniteArray):
    r"""
    An infinite array of subscripted :py:class:`~vc2_bit_widths.pyexp.PyExp`
    expressions.
    
    Example usage::

        >>> from vc2_bit_widths.pyexp import Argument
        
        >>> arg = Argument("arg")
        >>> a = VariableArray(2, arg)
        >>> a[1, 2]
        Subscript(Argument('arg'), Constant((1, 2)))
        >>> a[-10, 3]
        Subscript(Argument('arg'), Constant((-10, 3)))
    
    Parameters
    ==========
    ndim : int
        The number of dimensions in the array.
    exp : :py:class:`~vc2_bit_widths.pyexp.PyExp`
        The expression to be subscripted in each array element.
    """
    
    def __init__(self, ndim, exp):
        self._exp = exp
        super(VariableArray, self).__init__(ndim, cache=False)
    
    def get(self, keys):
        return self._exp[keys]
    
    @property
    def exp(self):
        """The :py:class:`PyExp` subscripted in this array."""
        return self._exp
    
    @property
    def period(self):
        return (1, ) * self.ndim
    
    @property
    def nop(self):
        return False
    
    def relative_step_size_to(self, other):
        if other is self:
            return (Fraction(1), ) * self.ndim
        else:
            return None


class LiftedArray(InfiniteArray):
    """
    Apply a one-dimensional lifting filter step to an array, as described
    in the VC-2 specification (15.4.4).
    
    Parameters
    ==========
    input_array : :py:class:`InfiniteArray`
        The input array whose entries will be filtered by the specified
        lifting stage.
    stage : :py:class:`vc2_data_tables.LiftingStage`
        A description of the lifting stage.
    interleave_dimension: int
        The dimension along which the filter will act.
    """
    
    def __init__(self, input_array, stage, filter_dimension):
        if filter_dimension >= input_array.ndim:
            raise TypeError("filter dimension out of range")
        
        self._input_array = input_array
        self._stage = stage
        self._filter_dimension = filter_dimension
        
        super(LiftedArray, self).__init__(self._input_array.ndim, cache=True)
    
    # For each lifting stage type, does this update the odd or even numbered
    # samples?
    LIFT_UPDATES_EVEN = {
        LiftingFilterTypes.even_add_odd: True,
        LiftingFilterTypes.even_subtract_odd: True,
        LiftingFilterTypes.odd_add_even: False,
        LiftingFilterTypes.odd_subtract_even: False,
    }
    
    # For each lifting stage type, does this add or subtract the sum of the
    # weighted filter taps to the sample being updated?
    LIFT_ADDS = {
        LiftingFilterTypes.even_add_odd: True,
        LiftingFilterTypes.even_subtract_odd: False,
        LiftingFilterTypes.odd_add_even: True,
        LiftingFilterTypes.odd_subtract_even: False,
    }
    
    def get(self, keys):
        key = keys[self._filter_dimension]
        
        is_even = (key & 1) == 0
        
        if is_even != LiftedArray.LIFT_UPDATES_EVEN[self._stage.lift_type]:
            # This lifting stage doesn't update this sample index; pass through
            # existing value.
            return self._input_array[keys]
        else:
            # This sample is updated by the lifting stage, work out the filter
            # coefficients.
            tap_indices = [
                key + (i*2) - 1
                for i in range(
                    self._stage.D,
                    self._stage.L + self._stage.D,
                )
            ]
            
            taps = [
                self._input_array[tuple(
                    key if i != self._filter_dimension else tap_index
                    for i, key in enumerate(keys)
                )]
                for tap_index in tap_indices
            ]
            
            total = sum(
                tap * weight
                for tap, weight in zip(taps, self._stage.taps)
            )
            
            if self._stage.S > 0:
                total += 1 << (self._stage.S - 1)
            
            total = total >> self._stage.S
            
            if LiftedArray.LIFT_ADDS[self._stage.lift_type]:
                return self._input_array[keys] + total
            else:
                return self._input_array[keys] - total
    
    @property
    def period(self):
        # A lifting filter applies the same filtering operation to all odd and
        # all even entries in an array.
        #
        # For an input with period 1 or 2, this results in an output with
        # period 2 (since all odd samples will be the result of one filtering
        # operation and all even samples another).
        #
        #           Input (period=1)              Input (period=2)
        #      +---+---+---+---+---+---+     +---+---+---+---+---+---+
        #      | a | a | a | a | a | a |     | a | b | a | b | a | b |
        #      +---+---+---+---+---+---+     +---+---+---+---+---+---+
        #                  |                             |
        #                  |                             |
        #                  V                             V
        #      +---+---+---+---+---+---+     +---+---+---+---+---+---+
        #      | Ea| Oa| Ea| Oa| Ea| Oa|     | Ea| Ob| Ea| Ob| Ea| Ob|
        #      +---+---+---+---+---+---+     +---+---+---+---+---+---+
        #          Output (period=2)             Output (period=2)
        #
        #                              +---+
        #                       Key:   | Ea|
        #                              +---+
        #                               /  \
        #                              /    \
        #             'E' = Even filter      'a' = Aligned with input 'a'
        #             'O' = Odd filter       'b' = Aligned with input 'b'
        #                                    ...
        # For inputs with an *even* period greater than 2, the resulting output
        # will have the same period as the input:
        #
        #             Input (period=4)
        #     +---+---+---+---+---+---+---+---+
        #     | a | b | c | d | a | b | c | d |
        #     +---+---+---+---+---+---+---+---+
        #                     |
        #                     |
        #                     V
        #     +---+---+---+---+---+---+---+---+
        #     | Ea| Ob| Ec| Od| Ea| Ob| Ec| Od|
        #     +---+---+---+---+---+---+---+---+
        #             Output (period=4)
        #
        # Finally, for inputs with an *odd* period greater than 2, the
        # resulting output will have a period of *double* the input as the
        # filters drift in and out of phase with the repeating input:
        #
        #                     Input (period=3)
        #     +---+---+---+---+---+---+---+---+---+---+---+---+
        #     | a | b | c | a | b | c | a | b | c | a | b | c |
        #     +---+---+---+---+---+---+---+---+---+---+---+---+
        #                             |
        #                             |
        #                             V
        #     +---+---+---+---+---+---+---+---+---+---+---+---+
        #     | Ea| Ob| Ec| Oa| Eb| Oc| Ea| Ob| Ec| Oa| Eb| Oc|
        #     +---+---+---+---+---+---+---+---+---+---+---+---+
        #                     Output (period=6)
        
        return tuple(
            lcm(2, p) if dim == self._filter_dimension else p
            for dim, p in enumerate(self._input_array.period)
        )
    
    @property
    def nop(self):
        return False
    
    def relative_step_size_to(self, other):
        if other is self:
            return (Fraction(1), ) * self.ndim
        else:
            return self._input_array.relative_step_size_to(other)


class RightShiftedArray(InfiniteArray):
    r"""
    Apply a right bit shift to every value in an input array.
    
    The right-shift operation is based on the description in the VC-2
    specification (15.4.3). Specifically, :math:`2^{\text{shiftbits}-1}` is
    added to the input values prior to the right-shift operation (which is used
    to implement rounding behaviour in integer arithmetic).
    
    Parameters
    ==========
    input_array : :py:class:`InfiniteArray`
        The array to have its values right-sifted
    shift_bits: int
        Number of bits to shift by
    """
    
    def __init__(self, input_array, shift_bits=1):
        self._input_array = input_array
        self._shift_bits = shift_bits
        
        super(RightShiftedArray, self).__init__(self._input_array.ndim, cache=True)
    
    def get(self, keys):
        if self._shift_bits == 0:
            return self._input_array[keys]
        else:
            value = self._input_array[keys]
            value += 1 << (self._shift_bits - 1)
            value = value >> self._shift_bits
            
            return value
    
    @property
    def period(self):
        return self._input_array.period
    
    @property
    def nop(self):
        return self._shift_bits == 0
    
    def relative_step_size_to(self, other):
        if other is self:
            return (Fraction(1), ) * self.ndim
        else:
            return self._input_array.relative_step_size_to(other)


class LeftShiftedArray(InfiniteArray):
    """
    Apply a left bit shift to every value in an input array.
    
    Parameters
    ==========
    input_array : :py:class:`InfiniteArray`
        The array to have its values left-shifted.
    shift_bits: int
        Number of bits to shift by.
    """
    
    def __init__(self, input_array, shift_bits=1):
        self._input_array = input_array
        self._shift_bits = shift_bits
        
        super(LeftShiftedArray, self).__init__(self._input_array.ndim, cache=True)
    
    def get(self, keys):
        return self._input_array[keys] << self._shift_bits
    
    @property
    def period(self):
        return self._input_array.period
    
    @property
    def nop(self):
        return self._shift_bits == 0
    
    def relative_step_size_to(self, other):
        if other is self:
            return (Fraction(1), ) * self.ndim
        else:
            return self._input_array.relative_step_size_to(other)


class SubsampledArray(InfiniteArray):
    """
    A subsampled view of another :py:class:`InfiniteArray`.
    
    Parameters
    ==========
    input_array : :py:class:`InfiniteArray`
        The array to be subsampled.
    steps, offsets: (int, ...)
        Tuples giving the step size and start offset for each dimension.
        
        When this array is indexed, the index into the input array is
        computed as::
        
            input_array_index[n] = (index[n] * step[n]) + offset[n]
    """
    
    def __init__(self, input_array, steps, offsets):
        if len(steps) != len(offsets):
            raise TypeError("steps and offsets do not match in length")
        if input_array.ndim != len(steps):
            raise TypeError("length of steps and offsets do not match input dimensions")
        
        self._input_array = input_array
        self._steps = steps
        self._offsets = offsets
        
        super(SubsampledArray, self).__init__(self._input_array.ndim, cache=False)
    
    def get(self, keys):
        return self._input_array[tuple(
            (key*step) + offset
            for key, step, offset in zip(keys, self._steps, self._offsets)
        )]
    
    @property
    def period(self):
        # In cases where the input period is divisible by the step size,
        # the output period will be the former divided by the latter:
        #
        #           Input (period=2)
        #      +---+---+---+---+---+---+
        #      | a | b | a | b | a | b |
        #      +---+---+---+---+---+---+
        #        .       . |     .
        #        .       . |     .       step=2, offset=0
        #        .       . V     .
        #      +-------+-------+-------+
        #      | a     | a     | a     |
        #      +-------+-------+-------+
        #          Output (period=1)
        #
        # However, if the input period is not evenly divisible by the step
        # interval, the resulting output period will be greater due to the
        # changing phase relationship of the subsampling and input period.
        #
        #                      Input (period=3)
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #      | a | b | c | a | b | c | a | b | c | a | b | c |
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #        .       .       .     | .       .       .
        #        .       .       .     | .       .       .       step=2, offset=0
        #        .       .       .     V .       .       .
        #      +-------+-------+-------+-------+-------+-------+
        #      | a     | c     | b     | a     | c     | b     |
        #      +-------+-------+-------+-------+-------+-------+
        #                     Output (period=3)
        #
        # Expressed mathematically, the actual period is found as the LCM of
        # the input period and step size, divided by the step size.
        return tuple(
            lcm(p, step)//step
            for p, step in zip(self._input_array.period, self._steps)
        )
    
    @property
    def nop(self):
        return True
    
    def relative_step_size_to(self, other):
        if other is self:
            return (Fraction(1), ) * self.ndim
        
        relative_step_size = self._input_array.relative_step_size_to(other)
        if relative_step_size is None:
            return None
        
        return tuple(
            step_size * prev_relative_step_size
            for step_size, prev_relative_step_size in zip(
                self._steps,
                relative_step_size,
            )
        )


class InterleavedArray(InfiniteArray):
    r"""
    An array view which interleaves two :py:class:`InfiniteArray`\ s
    together into a single array.
    
    Parameters
    ==========
    array_even, array_odd : :py:class:`InfiniteArray`
        The two arrays to be interleaved. 'array_even' will be used for
        even-indexed values in the specified dimension and 'array_odd' for
        odd.
    interleave_dimension: int
        The dimension along which the two arrays will be interleaved.
    """
    
    def __init__(self, array_even, array_odd, interleave_dimension):
        if array_even.ndim != array_odd.ndim:
            raise TypeError("arrays do not have same number of dimensions")
        if interleave_dimension >= array_even.ndim:
            raise TypeError("interleaving dimension out of range")
        
        self._array_even = array_even
        self._array_odd = array_odd
        self._interleave_dimension = interleave_dimension
        
        super(InterleavedArray, self).__init__(self._array_even.ndim, cache=False)
    
    def get(self, keys):
        is_even = (keys[self._interleave_dimension] & 1) == 0
        
        downscaled_keys = tuple(
            key if i != self._interleave_dimension else key//2
            for i, key in enumerate(keys)
        )
        
        if is_even:
            return self._array_even[downscaled_keys]
        else:
            return self._array_odd[downscaled_keys]
    
    @property
    def period(self):
        # When a pair of arrays are interleaved the resulting period in the
        # interleaved dimension will be double the LCM of their respective
        # periods. This is because, the respective phases of the two inputs may
        # drift with respect to eachother. For example:
        #
        #      +-------+-------+-------+-------+
        #      | a     | a     | a     | a     | Input (period=1)
        #      +-------+-------+-------+-------+
        #      +-.-----+-.-----+-.-----+-.-----+
        #      | .   A | .   B | .   A | .   B | Input (period=2)
        #      +-.-----+-.-----+-.-----+-.-----+
        #        .   .   .   . | .   .   .   .
        #        .   .   .   . | .   .   .   .
        #        .   .   .   . V .   .   .   .
        #      +---+---+---+---+---+---+---+---+
        #      | a | A | a | B | a | A | a | B | Output (period=4)
        #      +---+---+---+---+---+---+---+---+
        #
        # Example 2:
        #
        #      +-------+-------+-------+-------+-------+-------+
        #      | a     | b     | a     | b     | a     | b     | Input (period=2)
        #      +-------+-------+-------+-------+-------+-------+
        #      +-.-----+-.-----+-.-----+-.-----+-.-----+-.-----+
        #      | .   A | .   B | .   C | .   A | .   B | .   C | Input (period=3)
        #      +-.-----+-.-----+-.-----+-.-----+-.-----+-.-----+
        #        .   .   .   .   .   . | .   .   .   .   .   .
        #        .   .   .   .   .   . | .   .   .   .   .   .
        #        .   .   .   .   .   . V .   .   .   .   .   .
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #      | a | A | b | B | a | C | b | A | a | B | b | C | Output (period=12)
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #
        # In the dimension not being interleaved, the period will simply be the
        # LCM of the two inputs in that dimension as the phases of the
        # interleaved values drift with respect to eachother. In the
        # illustraton below, interleaving occurs on dimension 1:
        #
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #      | a | b | a | b | a | b | a | b | a | b | a | b | Input (period=(2, xxx))
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #      | A | B | C | A | B | C | A | B | C | A | B | C | Input (period=(3, xxx))
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #                              |
        #                              |
        #                              V
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        #      | a | b | a | b | a | b | a | b | a | b | a | b |
        #      +---+---+---+---+---+---+---+---+---+---+---+---+ Output (period=(6, xxx))
        #      | A | B | C | A | B | C | A | B | C | A | B | C |
        #      +---+---+---+---+---+---+---+---+---+---+---+---+
        return tuple(
            lcm(pa, pb) * (2 if dim == self._interleave_dimension else 1)
            for dim, (pa, pb) in enumerate(zip(
                self._array_even.period,
                self._array_odd.period,
            ))
        )
    
    @property
    def nop(self):
        return True
    
    def relative_step_size_to(self, other):
        if other is self:
            return (Fraction(1), ) * self.ndim
        
        even_step_size = self._array_even.relative_step_size_to(other)
        odd_step_size = self._array_odd.relative_step_size_to(other)
        
        if even_step_size is None and odd_step_size is None:
            return None
        
        if even_step_size is not None and not odd_step_size is None:
            raise ValueError(
                "Cannot find relative step size of {} in {} "
                "as it is interleaved with itself.".format(other, self)
            )
        
        if even_step_size is not None:
            relative_step_size = even_step_size
        else:
            relative_step_size = odd_step_size
        
        return tuple(
            s/2 if d == self._interleave_dimension else s
            for d, s in enumerate(relative_step_size)
        )


class SymbolicPeriodicCachingArray(InfiniteArray):
    r"""
    A caching view of a :py:class:`InfiniteArray` of
    :py:class:`~vc2_bit_widths.linexp.LinExp` values.
    
    This view will request at most one value from each filter phase of
    the input array and compute all other values by altering the indices in
    the previously computed values.
    
    For example, normally, when accessing values within a
    :py:class:`InfiniteArray`, values are computed from scratch on-demand::
    
        >>> s = SymbolArray(2, "s")
        >>> stage = <some filter stage>
        >>> la = LiftedArray(s, stage, 0)
        >>> la[0, 0]  # la[0, 0] worked out from scratch
        LinExp(...)
        >>> la[0, 1]  # la[0, 1] worked out from scratch
        LinExp(...)
        >>> la[0, 2]  # la[0, 2] worked out from scratch
        LinExp(...)
        >>> la[0, 3]  # la[0, 3] worked out from scratch
        LinExp(...)
    
    By contrast, when values are accessed in a
    :py:class:`SymbolicPeriodicCachingArray`, only the first access to each
    filter phase is computed from scratch. All other values are found by
    changing the symbol indices in the cached array value::
        
        >>> cached_la = SymbolicPeriodicCachingArray(la, s)
        >>> cached_la[0, 0]  # la[0, 0] worked out from scratch
        LinExp(...)
        >>> cached_la[0, 1]  # la[0, 1] worked out from scratch
        LinExp(...)
        >>> cached_la[0, 2]  # Cached value of la[0, 0] reused and mutated
        LinExp(...)
        >>> cached_la[0, 3]  # Cached value of la[0, 1] reused and mutated
        LinExp(...)
    
    Parameters
    ==========
    array : :py:class:`InfiniteArray`
        The array whose values are to be cached. These array values must
        consist only of symbols from :py:class:`SymbolArray`\ s passed as
        arguments, :py:class:`~vc2_bit_widths.linexp.AAError` terms and
        constants.
        
        .. warning::
            
            Error terms may be repeated as returned from this array where
            they would usually be unique. If this is a problem, you should
            not use this caching view.
            
            For example::
            
                >>> s = SymbolArray(2, "s")
                >>> sa = RightShiftedArray(s, 1)
                
                >>> # Unique AAError terms (without caching)
                >>> sa[0, 0]
                LinExp({('s', 0, 0): Fraction(1, 2), AAError(id=1): Fraction(1, 2)})
                >>> sa[0, 1]
                LinExp({('s', 0, 1): Fraction(1, 2), AAError(id=2): Fraction(1, 2)})
                
                >>> # Non-unique AAError terms after caching
                >>> ca = SymbolicPeriodicCachingArray(sa, s)
                >>> ca[0, 0]
                LinExp({('s', 0, 0): Fraction(1, 2), AAError(id=1): Fraction(1, 2)})
                >>> ca[0, 1]
                LinExp({('s', 0, 1): Fraction(1, 2), AAError(id=1): Fraction(1, 2)})
    
    symbol_arrays : :py:class:`SymbolArray`
        The symbol arrays which the 'array' argument is constructed from.
    """
    
    def __init__(self, array, *symbol_arrays):
        self._prefix_to_symbol_array = {
            array.prefix: array
            for array in symbol_arrays
        }
        
        self._array = array
        
        # {(component_array, key): period_number_scale, ...}
        self._scale_factors = {}
        
        super(SymbolicPeriodicCachingArray, self).__init__(self._array.ndim, cache=False)
    
    def _get_scale_factors(self, component_array):
        if component_array not in self._scale_factors:
            self._scale_factors[component_array] = tuple(
                int(step * period)
                for step, period in zip(
                    self.relative_step_size_to(component_array),
                    self.period,
                )
            )
        return self._scale_factors[component_array]
    
    def get(self, keys):
        # What phase does the requested value lie on
        phase_offset = tuple(
            k % p
            for k, p in zip(keys, self._array.period)
        )
        # How many periods along each axis is the requested value
        period_number = tuple(
            k // p
            for k, p in zip(keys, self._array.period)
        )
        
        base_value = self._array[phase_offset]
        
        replacements = {}
        for sym in base_value.symbols():
            if sym is None or isinstance(sym, AAError):
                continue
            
            prefix = sym[0]
            component_keys = sym[1:]
            component_array = self._prefix_to_symbol_array[prefix]
            
            new_keys = tuple(
                (p*s) + o
                for p, s, o in zip(
                    period_number,
                    self._get_scale_factors(component_array),
                    component_keys,
                )
            )
            
            replacements[(prefix, ) + component_keys] = (prefix, ) + new_keys
        
        # Could use LinExp.subs but this direct implementation is substantially
        # faster.
        return LinExp({
            replacements.get(sym, sym): coeff
            for sym, coeff in base_value
        })
    
    @property
    def period(self):
        return self._array.period
    
    @property
    def nop(self):
        return self._array.nop
    
    def relative_step_size_to(self, other):
        if other is self:
            return (Fraction(1), ) * self.ndim
        else:
            return self._array.relative_step_size_to(other)
