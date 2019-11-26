import pytest

import numpy as np

# Names starting with 'Test' not allowed in pytest module
from vc2_bit_widths.patterns import TestPattern as TP
from vc2_bit_widths.patterns import TestPatternSpecification as TPS
from vc2_bit_widths.patterns import OptimisedTestPatternSpecification as OTPS

from vc2_bit_widths.patterns import invert_test_pattern_specification

class TestTestPattern(object):
    
    def test_empty(self):
        t = TP({})
        assert t.origin == (0, 0)
        assert t.polarities.shape == (0, 0)
        assert len(t) == 0
    
    def test_from_dict(self):
        t = TP({
            (-5, 50): +1,
            (10, -1): -1,
        })
        assert t.origin == (-1, -5)
        assert t.polarities.shape == (52, 16)
        assert t.polarities[51, 0] == +1
        assert t.polarities[0, 15] == -1
    
    def test_from_origin_and_polarities(self):
        polarities = np.array([
            [0, 0, 0, -1],
            [1, 0, 0, 0],
        ])
        t = TP((-1, -5), polarities)
        assert t.origin == (-1, -5)
        assert t.polarities.dtype == np.int8
        assert np.array_equal(t.polarities, polarities)
    
    def test_normalise_dict(self):
        t = TP({
            (-5, 50): +1,
            (10, -1): 0,
        })
        assert t.origin == (50, -5)
        assert t.polarities.shape == (1, 1)
    
    def test_normalise_origin_and_polarities(self):
        polarities = np.array([
            [0, 0, 0, 0],
            [0, 0, 1, -1],
            [0, 0, 0, 0],
        ])
        t = TP((-1, -5), polarities)
        assert t.origin == (0, -3)
        assert t.polarities.shape == (1, 2)
        assert np.array_equal(t.polarities, [[1, -1]])
    
    def test_as_dict(self):
        t = TP({
            (-5, 50): +1,
            (10, -1): -1,
        })
        assert t.as_dict() == {
            (-5, 50): +1,
            (10, -1): -1,
        }
    
    def test_as_picture_and_slice_invalid(self):
        with pytest.raises(ValueError):
            TP({(-1, 0): +1}).as_picture_and_slice()
        with pytest.raises(ValueError):
            TP({(0, -1): +1}).as_picture_and_slice()
    
    def test_as_picture_and_slice_empty(self):
        p, s = TP({}).as_picture_and_slice()
        assert p.shape == (0, 0)
        assert p[s].shape == (0, 0)
    
    def test_as_picture_and_slice(self):
        p, s = TP({
            (1, 2): +1,
            (3, 4): -1,
        }).as_picture_and_slice()
        assert np.array_equal(p, [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -1],
        ])
        
        assert np.array_equal(p[s], [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
        ])
    
    def test_len(self):
        t = TP({
            (-5, 50): +1,
            (10, -1): -1,
            (0, 0): 0,
        })
        assert len(t) == 2
    
    def test_neg(self):
        t = TP({
            (0, 0): +1,
            (1, 0): -1,
        })
        assert np.array_equal((-t).polarities, [[-1, +1]])
    
    def test_eq(self):
        t1 = TP({
            (0, 0): +1,
            (1, 0): -1,
        })
        t2 = TP({
            (0, 0): +1,
            (1, 0): -1,
        })
        t3 = TP({
            (1, 1): +1,
            (2, 1): -1,
        })
        
        assert t1 == t2
        assert t1 != (-t2)
        assert t1 != t3
    
    def test_repr(self):
        t = TP({
            (-10, 20): -1,
        })
        assert repr(t) == "TestPattern({(-10, 20): -1})"


class TestInvertTestPatternSpecification(object):
    
    def test_test_pattern(self):
        ts = TPS(
            target=(1, 2),
            pattern=TP({
                (3, 4): +1,
                (5, 6): -1,
            }),
            pattern_translation_multiple=(7, 8),
            target_translation_multiple=(9, 10),
        )
        
        its = invert_test_pattern_specification(ts)
        
        assert its == TPS(
            target=(1, 2),
            pattern=TP({
                (3, 4): -1,
                (5, 6): +1,
            }),
            pattern_translation_multiple=(7, 8),
            target_translation_multiple=(9, 10),
        )
    
    def test_optimised_test_pattern(self):
        ts = OTPS(
            target=(1, 2),
            pattern=TP({
                (3, 4): +1,
                (5, 6): -1,
            }),
            pattern_translation_multiple=(7, 8),
            target_translation_multiple=(9, 10),
            quantisation_index=11,
            decoded_value=12,
            num_search_iterations=13,
        )
        
        its = invert_test_pattern_specification(ts)
        
        assert its == OTPS(
            target=(1, 2),
            pattern=TP({
                (3, 4): -1,
                (5, 6): +1,
            }),
            pattern_translation_multiple=(7, 8),
            target_translation_multiple=(9, 10),
            quantisation_index=11,
            decoded_value=12,
            num_search_iterations=13,
        )
