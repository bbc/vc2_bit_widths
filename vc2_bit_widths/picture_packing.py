"""
:py:mod:`vc2_bit_widths.picture_packing`: Pack test patterns into pictures
==========================================================================

This module contains a simple packing algorithm which attempts to efficiently
pack a collection of test patterns onto a small number of test pictures.


API
---

.. autofunction:: pack_test_patterns


Example output
--------------

The :py:func:`pack_test_patterns` function takes a dictionary of
:py:class:`~vc2_bit_widths.patterns.TestPatternSpecification` or
:py:class:`~vc2_bit_widths.patterns.OptimisedTestPatternSpecification`
objects and arranges these over as few pictures as possible. An example output
picture is shown below:

.. image:: /_static/example_packed_test_patterns.png
    :alt: An example HD picture containing a selection of test patterns.

In typical use, several independent sets of packed test pictures should be
created. One set can be created for all analysis transform test patterns. For
synthesis transform test patterns, the patterns should be divided up into
groups based on the quantisation index which should be used.

"""

from functools import partial

from collections import OrderedDict

import numpy as np


class PackTree(object):
    r"""
    A tree-based datastructure for allocating/packing rectangles into a fixed
    2D space.
    
    Inspired by `this lightmap packing algorithm
    <http://www.blackpawn.com/texts/lightmaps/default.html>`_.
    """
    
    def __init__(self, x, y, width, height):
        """
        Defines a rectangular region.
        
        Parameters
        ==========
        x, y : int
            The top-left corner of the rectangular region.
        width, height : int
            The dimensions of the region.
        """
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        
        # Has this region been allocated yet?
        self._allocated = False
        
        # If this region is split into several regions, a list of those
        # PackTrees, otherwise None.
        self._children = None
    
    def __eq__(self, other):
        return (
            self.x == other.x and
            self.y == other.y and
            self.width == other.width and
            self.height == other.height and
            self._allocated == other._allocated and
            self._children == other._children
        )
    
    def __repr__(self):
        return "<{} x={} y={} width={} height={} allocated={} children={}>".format(
            type(self).__name__,
            self._x,
            self._y,
            self._width,
            self._height,
            self._allocated,
            self._children,
        )
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    @property
    def x1(self):
        return self.x
    
    @property
    def y1(self):
        return self.y
    
    @property
    def x2(self):
        return self.x + self.width
    
    @property
    def y2(self):
        return self.y + self.height
    
    @property
    def area(self):
        return self.width * self.height
    
    def _h_split(self, y):
        """
        Split this region into two along a horizontal line.
        
        ::
            
              +-------+
              |       |
            --+-------+--  y
              |       |
              +-------+
        """
        assert not self._allocated and self._children is None
        assert self.y1 <= y < self.y2
        
        self._children = [
            PackTree(
                x=self.x,
                y=self.y,
                width=self.width,
                height=y - self.y,
            ),
            PackTree(
                x=self.x,
                y=y,
                width=self.width,
                height=self.y2 - y,
            ),
        ]
    
    def _v_split(self, x):
        """
        Split this region into two along a vertical line.
        
        ::
            
                  |
              +---+---+
              |   |   |
              |   |   |
              |   |   |
              +---+---+
                  |
                  x
        """
        assert not self._allocated and self._children is None
        assert self.x1 <= x < self.x2
        
        self._children = [
            PackTree(
                x=self.x,
                y=self.y,
                width=x - self.x,
                height=self.height,
            ),
            PackTree(
                x=x,
                y=self.y,
                width=self.x2 - x,
                height=self.height,
            ),
        ]
    
    def _cut_out_rectangle(self, x, y, width, height):
        """
        Divide this :py:class:`PackTree` up such that the specified region
        becomes its own :py:class:`PackTree`.
        
        Parameters
        ==========
        x, y : int
            The top left corner of the region to cut out.
        width, height : int
            The dimensions of the rectangle to cut out.
        
        Returns
        =======
        region : :py:class:`PackTree`
            A :py:class:`PackTree` exactly the size specified.
        """
        if (
            self.x == x and
            self.y == y and
            self.width == width and
            self.height == height
        ):
            # This rectangle fits the bill!
            return self
        else:
            # Cut this rectangle in line with one of the four edges of the
            # desired rectangle, choosing whichever creates the largest unused
            # area
            x1 = x
            y1 = y
            x2 = x1 + width
            y2 = y1 + height
            
            assert self.x1 <= x1 < self.x2
            assert self.x1 < x2 <= self.x2
            assert self.y1 <= y1 < self.y2
            assert self.y1 < y2 <= self.y2
            
            # [(area_of_empty_region, cut_function), ...]
            candidates = []
            
            if self.x1 < x1:
                candidates.append((
                    self.height * (x1 - self.x1),
                    partial(self._v_split, x=x1),
                ))
            if self.y1 < y1:
                candidates.append((
                    self.width * (y1 - self.y1),
                    partial(self._h_split, y=y1),
                ))
            if x2 < self.x2:
                candidates.append((
                    self.height * (self.x2 - x2),
                    partial(self._v_split, x=x2),
                ))
            if y2 < self.y2:
                candidates.append((
                    self.width * (self.y2 - y2),
                    partial(self._h_split, y=y2),
                ))
            
            # Apply the cut which produces the largest unused rectangle
            f = max(candidates, key=lambda c: c[0])[1]
            f()
            
            # Pick the child containing the target rectangle
            containing_child = (
                self._children[0]
                if (
                    self._children[0].x1 <= x < self._children[0].x2 and
                    self._children[0].y1 <= y < self._children[0].y2
                ) else
                self._children[1]
            )
            
            # Recurse (until the rectangle fits exactly)
            return containing_child._cut_out_rectangle(x, y, width, height)
    
    def _fit_rectangle(self, width, height, mx, my, ox, oy):
        """
        Does the specified rectangle fit within this rectangle? If so, return
        the top-left corner's coordinates used.
        
        Parameters
        ==========
        width, height : int
            The size of the rectangle.
        mx, my : int
        ox, oy : int
            Constrain the top-left-corner coordinates of the rectangle to be
            have coordinates of the form (x*mx + ox, y*my + oy) where x and y
            are integers.
        
        Returns
        =======
        top_left_corner : (x, y) or None
            The coordinates of the top-left corner of a valid alignment of the
            specified the rectangle within this :py:class:`PackTree`. Returns
            None if the rectangle does not fit.
        """
        # Round-up the top-left coordinate to be a whole multiple of that
        # required
        x1 = (((self.x - ox + mx - 1) // mx) * mx) + ox
        y1 = (((self.y - oy + my - 1) // my) * my) + oy
        
        x2 = x1 + width
        y2 = y1 + height
        
        if (
            self.x1 <= x1 < self.x2 and
            self.x1 < x2 <= self.x2 and
            self.y1 <= y1 < self.y2 and
            self.y1 < y2 <= self.y2
        ):
            return (x1, y1)
        else:
            return None
    
    def _find_candidates(self, width, height, mx, my, ox, oy):
        r"""
        Enumerate all candidate :py:class:`PackTree`\ s into which the
        specified rectangle could fit.
        
        Parameters
        ==========
        width, height : int
            The size of the rectangle.
        mx, my : int
        ox, oy : int
            Constrain the top-left-corner coordinates of the rectangle to be
            have coordinates of the form (x*mx + ox, y*my + oy) where x and y
            are integers.
        
        Returns
        =======
        candidates : [PackTree, ...]
        """
        if self._allocated:
            return []
        elif self._children is not None:
            out = []
            for child in self._children:
                out += child._find_candidates(width, height, mx, my, ox, oy)
            return out
        elif self._fit_rectangle(width, height, mx, my, ox, oy) is not None:
            return [self]
        else:
            return []
            
    
    def allocate(self, width, height, mx=1, my=1, ox=0, oy=0):
        """
        Allocate a rectangle of the defined size.
        
        Parameters
        ==========
        width, height : int
            The size of the rectangle to allocate
        mx, my : int
        ox, oy : int
            Constrain the top-left-corner coordinates of the rectangle to be
            allocated coordinates with the form (x*mx + ox, y*my + oy) where x
            and y are integers and ox < mx and oy < my.
        
        Returns
        =======
        allocation : (x, y) or None
            Returns the top-left coordinate of the allocated rectangular area,
            or None if no allocation could be made.
        """
        assert width >= 1
        assert height >= 1
        assert mx > 0
        assert my > 0
        assert 0 <= ox < mx
        assert 0 <= oy < my
        
        candidates = self._find_candidates(width, height, mx, my, ox, oy)
        
        if not candidates:
            return None
        
        # Pick the smallest possible candidate
        smallest_candidate = min(candidates, key=lambda c: c.area)
        
        # Allocate a rectangular region
        x, y = smallest_candidate._fit_rectangle(width, height, mx, my, ox, oy)
        rect = smallest_candidate._cut_out_rectangle(x, y, width, height)
        rect._allocated = True
        
        return x, y


def pack_test_patterns(width, height, test_pattern_specifications):
    """
    Given a picture size and a series of test patterns, pack those patterns
    onto as few pictures as possible.
    
    Parameters
    ==========
    width, height : int
        The size of the pictures on to which the test patterns should be
        allocated.
    test_pattern_specifications : {key: :py:class:`~vc2_bit_widths.patterns.TestPatternSpecification`, ...}
        The test patterns to be packed.
    
    Returns
    =======
    pictures : [:py:class:`numpy.array`, ...]
        A series of pictures containing test patterns.
    locations : {key: (picture_number, tx, ty), ...}
        For each of the supplied test patterns, gives the index of the picture
        it was assigned to and the coordinates (within the targeted filter
        array) of the array value which will be maximised/minimised.
        
        If any of the supplied test patterns are too large to fit in the
        specified width and height, those patterns will be omitted (and
        corresponding entries in the 'locations' will also be omitted).
    """
    # For each picture, a PackTree responsible for allocating space within that
    # picture
    pictures = []
    allocators = []
    
    locations = OrderedDict()
    
    for key, test_pattern_specification in test_pattern_specifications.items():
        # Find extents of the test pattern
        py, px = test_pattern_specification.pattern.origin
        ph, pw = test_pattern_specification.pattern.polarities.shape
        
        mx, my = test_pattern_specification.pattern_translation_multiple
        
        # Allocate a space for the picture
        for picture_index, (picture, allocator) in enumerate(zip(pictures, allocators)):
            allocation = allocator.allocate(pw, ph, mx, my, px, py)
            if allocation is not None:
                break
        else:
            # No space in an existing picture, add a new one
            picture = np.zeros((height, width), dtype=np.int8)
            allocator = PackTree(0, 0, width, height)
            
            allocation = allocator.allocate(pw, ph, mx, my, px, py)
            
            if allocation is None:
                # Skip this test pattern if it doesn't fit even on an empty
                # picture
                continue
            else:
                pictures.append(picture)
                allocators.append(allocator)
                picture_index = len(pictures) - 1
        
        # Copy the test pattern into the picture
        new_px, new_py = allocation
        dx = new_px - px
        dy = new_py - py
        picture[new_py:new_py+ph, new_px:new_px+pw] = test_pattern_specification.pattern.polarities
        
        # Adjust the target coordinates according to the position of the test
        # pattern
        tx, ty = test_pattern_specification.target
        tmx, tmy = test_pattern_specification.target_translation_multiple
        
        tx += (dx // mx) * tmx
        ty += (dy // my) * tmy
        
        # NB: Convert numbers to native python ints to ensure JSON
        # serialisability
        locations[key] = (picture_index, int(tx), int(ty))
    
    return (pictures, locations)
