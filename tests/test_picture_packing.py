import pytest

import numpy as np

from collections import OrderedDict

from vc2_bit_widths.patterns import TestPattern as TP
from vc2_bit_widths.patterns import TestPatternSpecification as TPS

from vc2_bit_widths.picture_packing import (
    PackTree,
    pack_test_patterns,
)


class TestPackTree(object):

    def test_properties(self):
        t = PackTree(1, 2, 3, 4)
        
        assert t.x == 1
        assert t.y == 2
        assert t.width == 3
        assert t.height == 4
        
        assert t.x1 == 1
        assert t.x2 == 1 + 3
        assert t.y1 == 2
        assert t.y2 == 2 + 4
        
        assert t.area == 3 * 4
    
    def test_h_split(self):
        t = PackTree(5, 10, 50, 100)
        t._h_split(y=15)
        assert t._children == [
            PackTree(5, 10, 50, 5),
            PackTree(5, 15, 50, 95),
        ]
        
        t = PackTree(5, 10, 50, 100)
        t._h_split(y=11)
        assert t._children == [
            PackTree(5, 10, 50, 1),
            PackTree(5, 11, 50, 99),
        ]
        
        t = PackTree(5, 10, 50, 100)
        t._h_split(y=109)
        assert t._children == [
            PackTree(5, 10, 50, 99),
            PackTree(5, 109, 50, 1),
        ]
    
    def test_v_split(self):
        t = PackTree(5, 10, 50, 100)
        t._v_split(x=15)
        assert t._children == [
            PackTree(5, 10, 10, 100),
            PackTree(15, 10, 40, 100),
        ]
        
        t = PackTree(5, 10, 50, 100)
        t._v_split(x=6)
        assert t._children == [
            PackTree(5, 10, 1, 100),
            PackTree(6, 10, 49, 100),
        ]
        
        t = PackTree(5, 10, 50, 100)
        t._v_split(x=54)
        assert t._children == [
            PackTree(5, 10, 49, 100),
            PackTree(54, 10, 1, 100),
        ]
    
    def test_cut_out_rectangle_exact_size_match(self):
        t = PackTree(5, 10, 50, 100)
        
        assert t._cut_out_rectangle(5, 10, 50, 100) is t
        assert t._children is None
    
    def test_cut_out_rectangle_recurse(self):
        t = PackTree(5, 10, 50, 100)
        
        out = t._cut_out_rectangle(10, 30, 30, 40)
        
        # Sanity check: allocated the requested region
        assert out.x == 10
        assert out.y == 30
        assert out.width == 30
        assert out.height == 40
        
        # Check allocated it the expected way
        #
        #  +---------------+
        #  |      2        |
        #  |               |
        #  |-+-------+- - -|
        #  |4|       |  3  |
        #  |-+-------+- - -|
        #  |               |
        #  |               |
        #  |      1        |
        #  |               |
        #  |               |
        #  |               |
        #  +---------------+
        
        t2 = PackTree(5, 10, 50, 100)
        t2._h_split(y=70)
        
        t2t = t2._children[0]
        t2t._h_split(y=30)
        
        t2tb = t2t._children[1]
        t2tb._v_split(x=40)
        
        t2tbl = t2tb._children[0]
        t2tbl._v_split(x=10)
        
        t2tblr = t2tbl._children[1]
        
        assert t2 == t
    
    def test_fit_rectangle(self):
        t = PackTree(5, 10, 50, 100)
        
        # Same size
        assert t._fit_rectangle(50, 100, 1, 1, 0, 0) == (5, 10)
        
        # Smaller
        assert t._fit_rectangle(40, 90, 1, 1, 0, 0) == (5, 10)
        
        # Larger
        assert t._fit_rectangle(51, 101, 1, 1, 0, 0) is None
        
        # Multiple constraint (fits exactly)
        assert t._fit_rectangle(47, 98, 8, 3, 0, 0) == (8, 12)
        
        # Multiple constraint (smaller)
        assert t._fit_rectangle(40, 90, 8, 3, 0, 0) == (8, 12)
        
        # Multiple constraint (larger)
        assert t._fit_rectangle(48, 99, 8, 3, 0, 0) is None
        
        # Multiple and offset constraint (fits exactly)
        assert t._fit_rectangle(44, 100, 8, 3, 3, 1) == (11, 10)
        
        # Multiple and offset constraint (smaller)
        assert t._fit_rectangle(43, 99, 8, 3, 3, 1) == (11, 10)
        
        # Multiple and offset constraint (larger)
        assert t._fit_rectangle(45, 101, 8, 3, 3, 1) is None
    
    def test_fit_rectangle_multiple_and_offset(self):
        for x, exp, m, o in [
            # No multiple or offset
            (0, 0, 1, 0),
            (1, 1, 1, 0),
            (2, 2, 1, 0),
            (3, 3, 1, 0),
            (4, 4, 1, 0),
            (5, 5, 1, 0),
            (6, 6, 1, 0),
            (7, 7, 1, 0),
            # Multiple of 3
            (0, 0, 3, 0),
            (1, 3, 3, 0),
            (2, 3, 3, 0),
            (3, 3, 3, 0),
            (4, 6, 3, 0),
            (5, 6, 3, 0),
            (6, 6, 3, 0),
            (7, 9, 3, 0),
            # Odd numbers
            (0, 1, 2, 1),
            (1, 1, 2, 1),
            (2, 3, 2, 1),
            (3, 3, 2, 1),
            (4, 5, 2, 1),
            (5, 5, 2, 1),
            (6, 7, 2, 1),
            (7, 7, 2, 1),
        ]:
            t = PackTree(x, x, 1000, 1000)
            assert t._fit_rectangle(1, 1, m, m, o, o) == (exp, exp)
    
    def test_find_candidates_already_allocated(self):
        t = PackTree(5, 10, 50, 100)
        t._allocated = True
        
        assert t._find_candidates(1, 1, 1, 1, 0, 0) == []
    
    def test_find_candidates_too_large(self):
        t = PackTree(5, 10, 50, 100)
        
        assert t._find_candidates(1000, 1000, 1, 1, 0, 0) == []
    
    def test_find_candidates_awkward_multiple(self):
        t = PackTree(5, 10, 50, 100)
        
        assert t._find_candidates(1, 1, 1000, 1000, 0, 0) == []
    
    def test_find_candidates_fits(self):
        t = PackTree(5, 10, 50, 100)
        
        assert t._find_candidates(1, 1, 1, 1, 0, 0) == [t]
    
    def test_find_candidates_child_fits(self):
        t = PackTree(5, 10, 50, 100)
        t._h_split(y=20)
        
        c0, c1 = t._children
        
        # Fits in either
        assert t._find_candidates(1, 1, 1, 1, 0, 0) == [c0, c1]
        
        # Fits in larger one only
        assert t._find_candidates(1, 20, 1, 1, 0, 0) == [c1]
    
    def test_allocate_sets_allocated(self):
        t = PackTree(5, 10, 50, 100)
        assert t.allocate(50, 100) == (5, 10)
        assert t._allocated
    
    def test_allocate_with_multiples(self):
        t = PackTree(5, 10, 50, 100)
        
        # An awkward allocation
        assert t.allocate(10, 20, 8, 3) == (8, 12)
    
    def test_allocate_in_smallest_available(self):
        t = PackTree(5, 10, 50, 100)
        t._v_split(x=50)
        
        assert t.allocate(5, 5, 1, 1) == (50, 10)
    
    def test_allocate_doesnt_fit(self):
        t = PackTree(5, 10, 50, 100)
        t._v_split(x=50)
        
        assert t.allocate(50, 100, 1, 1) is None
    
    def test_fuzz(self):
        # A simple fuzz test: allocate a number of randomly sized rectangles
        
        width, height = 1920, 1080
        num_rects = 100
        
        t = PackTree(0, 0, width, height)
        
        # Generate a set of random rectangles to allocate
        rand = np.random.RandomState(1)
        sizes = list(zip(
            rand.randint(1, 100, num_rects),
            rand.randint(1, 100, num_rects),
        ))
        placements = [
            t.allocate(w, h, 4, 8, 3, 5)
            for w, h in sizes
        ]
        
        # All should be successfully allocated
        assert all(p is not None for p in placements)
        
        # All allocations should be within the specified size
        assert all(
            0 <= x < width and
            0 < x + w <= width and
            0 <= y < height and
            0 < y + h <= height
            for (x, y), (w, h) in zip(placements, sizes)
        )
        
        # All allocations should be on the correct multiple
        assert all(
            x % 4 == 3 and
            y % 8 == 5
            for x, y in placements
        )
        
        # No allocations should overlap
        a = np.full((height, width), -1)
        for i, ((x, y), (w, h)) in enumerate(zip(placements, sizes)):
            s = (slice(y, y + h), slice(x, x + w))
            
            assert np.all(a[s] == -1)
            a[s] = i


class TestPackTestPatterns(object):
    
    def test_simple_picture(self):
        test_patterns = OrderedDict([
            (
                (1, "Output", 0, 0),
                TPS(
                    pattern=TP({
                        (1, 0): +1,
                        (0, 1): -1,
                    }),
                    pattern_translation_multiple=(1, 1),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
        ])
        
        width, height = 20, 10
        
        pictures, locations = pack_test_patterns(width, height, test_patterns)
        
        assert len(pictures) == 1
        assert list(locations) == list(test_patterns)
        
        # Should have been assigned to the top-left of the picture
        exp_picture = np.zeros((height, width), dtype=np.int8)
        exp_picture[0, 1] = +1
        exp_picture[1, 0] = -1
        assert np.array_equal(pictures[0], exp_picture)
        
        assert locations[(1, "Output", 0, 0)] == (0, 1, 1)
    
    def test_offset_picture(self):
        test_patterns = OrderedDict([
            (
                (1, "Output", 0, 0),
                TPS(
                    pattern=TP({
                        (2, 1): +1,
                        (1, 2): -1,
                    }),
                    pattern_translation_multiple=(4, 4),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
        ])
        
        width, height = 20, 10
        
        pictures, locations = pack_test_patterns(width, height, test_patterns)
        
        assert len(pictures) == 1
        assert list(locations) == list(test_patterns)
        
        # Should have been assigned to the top-left most point possible
        exp_picture = np.zeros((height, width), dtype=np.int8)
        exp_picture[1, 2] = +1
        exp_picture[2, 1] = -1
        assert np.array_equal(pictures[0], exp_picture)
        
        assert locations[(1, "Output", 0, 0)] == (0, 1, 1)
    
    def test_simple_packing(self):
        test_patterns = OrderedDict([
            (
                (1, "Output", 0, 0),
                TPS(
                    pattern=TP({
                        (2, 1): +1,
                        (1, 2): -1,
                    }),
                    pattern_translation_multiple=(4, 4),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
            (
                (1, "Output", 0, 1),
                TPS(
                    pattern=TP({
                        (2, 1): +1,
                        (1, 2): +1,
                    }),
                    pattern_translation_multiple=(4, 4),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
        ])
        
        width, height = 20, 10
        
        pictures, locations = pack_test_patterns(width, height, test_patterns)
        
        assert len(pictures) == 1
        assert list(locations) == list(test_patterns)
        
        exp_picture = np.zeros((height, width), dtype=np.int8)
        # First pattern, top left
        exp_picture[1, 2] = +1
        exp_picture[2, 1] = -1
        # Second pattern, just underneath
        exp_picture[5, 2] = +1
        exp_picture[6, 1] = +1
        assert np.array_equal(pictures[0], exp_picture)
        
        assert locations[(1, "Output", 0, 0)] == (0, 1, 1)
        assert locations[(1, "Output", 0, 1)] == (0, 1, 2)
    
    def test_target_translation(self):
        # The test patterns should be allocated like so:
        #
        #     +--+----+-----+
        #     |0 | 1  |     |
        #     +--+    |     |
        #     |  +----+-+---+
        #     |  | 2    |   |
        #     |  |      |   |
        #     +--+------+---+
        #
        test_patterns = OrderedDict([
            (
                (1, "Output", 0, 0),
                TPS(
                    pattern=TP({
                        (2, 1): +1,
                        (1, 2): -1,
                    }),
                    pattern_translation_multiple=(4, 4),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
            (
                (1, "Output", 0, 1),
                TPS(
                    pattern=TP({
                        (10, 1): +1,
                        (1, 5): +1,
                    }),
                    pattern_translation_multiple=(4, 4),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
            (
                (1, "Output", 0, 2),
                TPS(
                    pattern=TP({
                        (5, 1): -1,
                        (1, 5): -1,
                    }),
                    pattern_translation_multiple=(4, 4),
                    target=(2, 3),
                    target_translation_multiple=(4, 5),
                ),
            ),
        ])
        
        width, height = 20, 15
        
        pictures, locations = pack_test_patterns(width, height, test_patterns)
        
        assert len(pictures) == 1
        assert list(locations) == list(test_patterns)
        
        exp_picture = np.zeros((height, width), dtype=np.int8)
        # First pattern, top left
        exp_picture[1, 2] = +1
        exp_picture[2, 1] = -1
        # Second pattern, just to the right
        exp_picture[1, 14] = +1
        exp_picture[5, 5] = +1
        # Third pattern, below-right of both
        exp_picture[13, 5] = -1
        exp_picture[9, 9] = -1
        assert np.array_equal(pictures[0], exp_picture)
        
        assert locations[(1, "Output", 0, 0)] == (0, 1, 1)
        assert locations[(1, "Output", 0, 1)] == (0, 2, 1)
        assert locations[(1, "Output", 0, 2)] == (0, 2 + 4, 3 + (2*5))
    
    def test_overspill_pictures(self):
        # Tests that allocation can overspill onto extra pictures and that
        # later allocations can be inserted onto any available picture.
        #
        #     +------+------+   +-------------+
        #     |      |      |   |      1      |
        #     |  0   |  2   |   +-------------+
        #     |      |      |   |      3      |
        #     +------+------+   +-------------+
        #
        test_patterns = OrderedDict([
            (
                (1, "Output", 0, 0),
                TPS(
                    pattern=TP({
                        (9, 0): -1,
                        (0, 9): -1,
                    }),
                    pattern_translation_multiple=(1, 1),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
            (
                (1, "Output", 0, 1),
                TPS(
                    pattern=TP({
                        (19, 0): +1,
                        (0, 4): -1,
                    }),
                    pattern_translation_multiple=(1, 1),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
            (
                (1, "Output", 0, 2),
                TPS(
                    pattern=TP({
                        (9, 0): +1,
                        (0, 9): +1,
                    }),
                    pattern_translation_multiple=(1, 1),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
            (
                (1, "Output", 0, 3),
                TPS(
                    pattern=TP({
                        (19, 0): -1,
                        (0, 4): +1,
                    }),
                    pattern_translation_multiple=(1, 1),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
        ])
        
        width, height = 20, 10
        
        pictures, locations = pack_test_patterns(width, height, test_patterns)
        
        assert len(pictures) == 2
        assert list(locations) == list(test_patterns)
        
        exp_picture_0 = np.zeros((height, width), dtype=np.int8)
        exp_picture_1 = np.zeros((height, width), dtype=np.int8)
        # Pattern 0
        exp_picture_0[0, 9] = -1
        exp_picture_0[9, 0] = -1
        # Pattern 1
        exp_picture_1[0, 19] = +1
        exp_picture_1[4, 0] = -1
        # Pattern 2
        exp_picture_0[0, 19] = +1
        exp_picture_0[9, 10] = +1
        # Pattern 3
        exp_picture_1[5, 19] = -1
        exp_picture_1[9, 0] = +1
        assert np.array_equal(pictures[0], exp_picture_0)
        assert np.array_equal(pictures[1], exp_picture_1)
        
        assert locations[(1, "Output", 0, 0)] == (0, 1, 1)
        assert locations[(1, "Output", 0, 1)] == (1, 1, 1)
        assert locations[(1, "Output", 0, 2)] == (0, 11, 1)
        assert locations[(1, "Output", 0, 3)] == (1, 1, 6)
    
    def test_large_test_patterns_skipped(self):
        test_patterns = OrderedDict([
            (
                (1, "Output", 0, 0),
                TPS(
                    pattern=TP({
                        (1, 0): +1,
                        (0, 1): -1,
                    }),
                    pattern_translation_multiple=(1, 1),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
            (
                (1, "Output", 0, 1),
                TPS(
                    pattern=TP({
                        (100, 0): -1,
                        (0, 100): -1,
                    }),
                    pattern_translation_multiple=(1, 1),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
            (
                (1, "Output", 0, 2),
                TPS(
                    pattern=TP({
                        (1, 0): -1,
                        (0, 1): +1,
                    }),
                    pattern_translation_multiple=(1, 1),
                    target=(1, 1),
                    target_translation_multiple=(1, 1),
                ),
            ),
        ])
        
        width, height = 20, 10
        
        pictures, locations = pack_test_patterns(width, height, test_patterns)
        
        assert len(pictures) == 1
        assert list(locations) == [
            (1, "Output", 0, 0),
            (1, "Output", 0, 2),
        ]
        
        exp_picture = np.zeros((height, width), dtype=np.int8)
        # First pattern, top left
        exp_picture[0, 1] = +1
        exp_picture[1, 0] = -1
        # Third pattern, underneath
        exp_picture[2, 1] = -1
        exp_picture[3, 0] = +1
        assert np.array_equal(pictures[0], exp_picture)
        
        assert locations[(1, "Output", 0, 0)] == (0, 1, 1)
        assert locations[(1, "Output", 0, 2)] == (0, 1, 3)
