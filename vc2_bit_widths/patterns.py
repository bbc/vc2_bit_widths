"""
:py:mod:`vc2_bit_widths.patterns`: Containers for test pattern data
===================================================================

The following datastructures are used to define test patterns.

Test pattern specification
--------------------------

These types define a test pattern along with information about required spacing
or quantisation levels.

.. autoclass:: TestPatternSpecification
    :no-members:

.. autoclass:: OptimisedTestPatternSpecification
    :no-members:

.. autofunction:: invert_test_pattern_specification


Test pattern
------------

A description of a test pattern in terms of pixel polarities.

.. autoclass:: TestPattern
    :members:
    :special-members: __len__, __neg__

"""

import numpy as np

from collections import namedtuple


class TestPattern(object):
    r"""
    A test pattern.
    
    A test pattern is described by a set of pixels with a defined polarity
    (either -1 or +1) with all other pixels being undefined (represented as 0).
    Where a pixel has +ve polarity, the maximum signal value should be used in
    test pictures, for -ve polarities, the minimum signal value should be used.
    
    This object stores test patterns as a :py:class:`numpy.array` defining a
    rectangular region where some pixels are not undefined. This region lies
    at the offset defined by :py:attr:`origin` and the values within the region
    are defined by :py:attr:`polarities`.
    
    :py:class:`TestPattern` objects can be constructed from either a dictionary
    of the form ``{(x, y): polarity, ...}`` or from a pair of arguments,
    ``origin`` and ``polarities`` giving a ``(dy, dx)`` tuple and
    2D :py:class:`np.array` respectively.
    """
    
    def __init__(self, *args):
        if len(args) == 1:
            # Convert from dictionary
            dictionary = args[0]
            if len(dictionary) == 0:
                origin = (0, 0)
                polarities = np.zeros((0, 0), dtype=np.int8)
            else:
                xs, ys = map(np.array, zip(*dictionary))
                origin = (np.min(ys), np.min(xs))
                width = np.max(xs) - origin[1] + 1
                height = np.max(ys) - origin[0] + 1
                polarities = np.zeros((height, width), dtype=np.int8)
                polarities[(ys-origin[0], xs-origin[1])] = list(dictionary.values())
        elif len(args) == 2:
            origin, polarities = args
            polarities = np.array(polarities)
        else:
            raise TypeError("TestPattern expects either 1 or 2 arguments")
        
        # Normalise to smallest possible array
        if polarities.shape == (0, 0):
            origin = (0, 0)
        else:
            ys, xs = np.nonzero(polarities)
            origin = (origin[0] + np.min(ys), origin[1] + np.min(xs))
            polarities = polarities[np.min(ys):np.max(ys)+1, np.min(xs):np.max(xs)+1]
        polarities = polarities.astype(np.int8)
        
        self._origin = origin
        self._polarities = polarities
    
    @property
    def origin(self):
        """
        The origin of the rectangular region definining the test pattern given
        as (dy, dx).
        
        .. warning::
            
            Numpy-style index ordering used!
        """
        return self._origin
    
    @property
    def polarities(self):
        """
        A :py:class:`numpy.array` definining the polarities of the pixels
        within a rectangular region definde by :py:attr:`origin`.
        """
        return self._polarities
    
    def as_dict(self):
        """
        Return a dictionary representation of the test pattern of the form
        ``{(x, y): polarity, ...}``.
        """
        dy, dx = self.origin
        return {
            (x+dx, y+dy): self.polarities[y, x]
            for y, x in zip(*np.nonzero(self._polarities))
        }
    
    def as_picture_and_slice(self, signal_min=-1, signal_max=+1, dtype=np.int64):
        """
        Convert this test pattern into a picture array with its origin at (0,
        0).
        
        Not supported for test patterns with pixels at negative coordinates.
        
        Parameters
        ==========
        signal_min : int
        signal_max : int
            The values to use for pixels with negative and positive polarities,
            respecitvely.
        dtype
            The :py:class:`numpy.array` datatype for the returned array.
        
        Returns
        =======
        picture : :py:class:`numpy.array`
            A test picture with its origin at (0, 0) containing the test
            pattern (with zeros in all undefined pixels).
        picture_slice : (:py:class:`slice`, :py:class:`slice`)
            A :py:class:`numpy.array` slice identifying the region of
            ``picture`` which contains the test pattern.
        """
        if self.origin[0] < 0 or self.origin[1] < 0:
            raise ValueError("Cannot make picture from test pattern with negative coordinates.")
        
        width = self.polarities.shape[1] + self.origin[1]
        height = self.polarities.shape[0] + self.origin[0]
        
        picture = np.zeros((height, width), dtype=dtype)
        picture_slice = (
            slice(self.origin[0], height),
            slice(self.origin[1], width),
        )
        
        view = picture[picture_slice]
        view[self.polarities > 0] = signal_max
        view[self.polarities < 0] = signal_min
        
        return (picture, picture_slice)
    
    def __len__(self):
        """The number of defined pixels in the test pattern."""
        return np.count_nonzero(self._polarities)
    
    def __neg__(self):
        """Return a :py:class:`TestPattern` with inverted polarities."""
        return TestPattern(self.origin, -self.polarities)
    
    def __eq__(self, other):
        return (
            self.origin == other.origin and
            np.array_equal(self.polarities, other.polarities)
        )
    
    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.as_dict())


TestPatternSpecification = namedtuple(
    "TestPatternSpecification",
    "target,pattern,pattern_translation_multiple,target_translation_multiple",
)
"""
A definition of a test pattern for a VC-2 filter. This test pattern is intended
to maximise the value of a particular intermediate or output value of a VC-2
filter.

Test patterns for both for analysis and synthesis filters are defined in terms
of picture test patterns. For analysis filters, the picture should be fed
directly to the encoder under test. For synthesis filters, the pattern must
first be fed to an encoder and the transform coefficients quantised before
being fed to a decoder.

Test patterns tend to be quite small (tens to low hundreds of pixels square)
and so it is usually sensible to collect together many test patterns into a
single picture (see :py:mod:`vc2_bit_widths.picture_packing`). To retain their
functionality, test patterns must remain correctly aligned with their target
filter. When relocating a test pattern, the pattern must be moved only by
multiples of the values in ``pattern_translation_multiple``. For each multiple
moved, the target value effected by the pattern moves by the same multiple of
``target_translation_multiple``.

Parameters
==========
target : (tx, ty)
    The target coordinate which is maximised by this test pattern.
pattern : :py:class:`TestPattern`
    The input pattern to be fed into a VC-2 encoder. Only those pixels defined
    in this pattern need be set -- all other pixels may be set to arbitrary
    values and have no effect.
    
    The test pattern is specified such that the pattern is as close to the
    top-left corner as possible given ``pattern_translation_multiple``, without
    any negative pixel coordinates. That is, ``0 <= min(x) < mx`` and ``0 <=
    min(y) < my``.
pattern_translation_multiple : (mx, my)
target_translation_multiple : (tmx, tmy)
    The multiples by which pattern pixel coordinates and target array
    coordinates may be translated when relocating the test pattern. Both the
    pattern and target must be translated by the same multiple of these two
    factors.
    
    For example, if the pattern is translated by (2*mx, 3*my), the target must
    be translated by (2*tmx, 3*tmy).
"""


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
pattern : :py:class:`TestPattern`
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


def invert_test_pattern_specification(test_pattern):
    """
    Given a :py:class:`TestPatternSpecification` or
    :py:class:`OptimisedTestPatternSpecification`,
    return a copy with the signal polarity inverted.
    """
    tuple_type = type(test_pattern)
    
    values = test_pattern._asdict()
    values["pattern"] = -values["pattern"]
    
    return tuple_type(**values)
