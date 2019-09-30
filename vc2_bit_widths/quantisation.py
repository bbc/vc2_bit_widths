r"""
VC-2 Quantisation
=================

The VC-2 quantisation related pseudocode functions are implemented as follows:

.. autofunction:: forward_quant

.. autofunction:: inverse_quant

.. autofunction:: quant_factor

.. autofunction:: quant_offset

The following functions compute incidental information about the behaviour of
the VC-2 quantisation scheme.

.. autofunction:: maximum_useful_quantisation_index

.. autofunction:: maximum_dequantised_magnitude

"""

__all__ = [
    "forward_quant",
    "inverse_quant",
    "quant_factor",
    "quant_offset",
    "maximum_useful_quantisation_index",
    "maximum_dequantised_magnitude",
]


def forward_quant(coeff, quant_index):
    """Quantise a value according to the informative description in (13.3.1)"""
    if coeff >= 0:
        return (4*coeff) // quant_factor(quant_index)
    else:
        return -((4*-coeff) // quant_factor(quant_index))


def inverse_quant(quantized_coeff, quant_index):
    """(13.3.1)"""
    magnitude = abs(quantized_coeff)
    if magnitude != 0:
        magnitude *= quant_factor(quant_index)
        magnitude += quant_offset(quant_index)
        magnitude += 2
        magnitude //= 4
    return (1 if quantized_coeff > 0 else -1) * magnitude


def quant_factor(index):
    """(13.3.2)"""
    base = 2**(index//4)
    if (index%4) == 0:
        return (4 * base)
    elif (index%4) == 1:
        return(((503829 * base) + 52958) // 105917)
    elif (index%4) == 2:
        return(((665857 * base) + 58854) // 117708)
    elif (index%4) == 3:
        return(((440253 * base) + 32722) // 65444)


def quant_offset(index):
    """(13.3.2)"""
    if index == 0:
        offset = 1
    elif index == 1:
        offset = 2
    else:
        offset = (quant_factor(index) + 1)//2
    return offset


def maximum_useful_quantisation_index(value):
    """
    Compute the smallest quantisation index which quantizes the supplied value
    to zero. This is considered to be the largest useful quantisation index
    with respect to this value.
    """
    # NB: Since quantisation indices correspond to exponentially growing
    # quantisation factors, the runtime of this loop is only logarithmic with
    # respect to the magnitude of the value.
    quant_index = 0
    while forward_quant(value, quant_index) != 0:
        quant_index += 1
    return quant_index


def maximum_dequantised_magnitude(value):
    """
    Find the value with the largest magnitude that the supplied value may be
    dequantised to for any quantisation index.
    """
    # NB: A proof of the correctness of this function is provided in the
    # documentation.
    quant_index = maximum_useful_quantisation_index(value) - 1
    return inverse_quant(forward_quant(value, quant_index), quant_index)

