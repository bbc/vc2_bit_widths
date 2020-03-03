import pytest

# The vc2_conformance library contains an implementation of the VC-2
# quantisation scheme machine verified against the VC-2 specification's
# pseudocode. Here we'll simply check the local implementation finds the same
# values.
import vc2_conformance.quantization as pseudocode

from vc2_bit_widths.quantisation import (
    forward_quant,
    inverse_quant,
    quant_factor,
    quant_offset,
    maximum_useful_quantisation_index,
    maximum_dequantised_magnitude,
)


def test_quant_factor():
    for index in range(1000):
        assert quant_factor(index) == pseudocode.quant_factor(index)

def test_quant_offset():
    for index in range(1000):
        assert quant_offset(index) == pseudocode.quant_offset(index)

def test_inverse_quant():
    for coeff in list(range(100)) + list(range(100, 10000, 997)):
        for sign in [1, -1]:
            for quant_index in range(64):
                assert (
                    inverse_quant(coeff * sign, quant_index) ==
                    pseudocode.inverse_quant(coeff * sign, quant_index)
                )

def test_forward_quant():
    for coeff in list(range(100)) + list(range(100, 10000, 997)):
        for sign in [1, -1]:
            for quant_index in range(64):
                value = coeff * sign
                quant_value = forward_quant(value, quant_index)
                dequant_value = inverse_quant(quant_value, quant_index)
                error = dequant_value - value
                assert abs(error*4) <= quant_factor(quant_index)


@pytest.mark.parametrize("value", [
    # The first ten quantisation factors
    1, 2, 3, 4, 5, 6, 8, 9, 11, 13,
    # Some non-quantisation-factor values
    14, 15,
    # A very large value to ensure runtime is actually logarithmic
    999999999999,
])
def test_maximum_useful_quantisation_index(value):
    # Empirically check that any lower quantisation index produces a non-zero
    # result
    index = maximum_useful_quantisation_index(value)
    assert forward_quant(value, index) == 0
    assert forward_quant(value, index-1) != 0


@pytest.mark.parametrize("value", [
    # The first ten quantisation factors
    1, 2, 3, 4, 5, 6, 8, 9, 11, 13,
    # Some non-quantisation-factor values
    14, 15,
    # A very large value to ensure runtime is actually logarithmic
    999999999999,
])
@pytest.mark.parametrize("sign", [-1, +1])
def test_maximum_dequantised_magnitude(value, sign):
    # Empirically check that any lower quantisation index produces a non-zero
    # result
    value *= sign
    expectation = maximum_dequantised_magnitude(value)
    
    for qi in range(maximum_useful_quantisation_index(value) + 1):
        if expectation > 0:
            assert inverse_quant(forward_quant(value, qi), qi) <= expectation
        elif expectation < 0:
            assert inverse_quant(forward_quant(value, qi), qi) >= expectation
        else:
            assert inverse_quant(forward_quant(value, qi), qi) == 0
