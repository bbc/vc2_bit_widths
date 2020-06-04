.. _theory-affine-arithmetic-quantisation:

Quantisation and affine arithmetic
==================================

Affine arithmetic may also be used to model the effects of quantisation since
VC-2's dead-zone quantiser is essentially just truncating integer division.

Consider the following (simplified) definition of VC-2's quantiser and
dequantiser:

.. math::

    \text{quantise}(x, qf) &= x//qf \\
    \text{dequantise}(X, qf) &= X \times qf + \frac{qf}{2}

In affine arithmetic this becomes:

.. math::

    \text{quantise}(x, qf) &= \frac{x}{qf} + \frac{e_1 - 1}{2}\\
    \text{dequantise}(X, qf) &= X \times qf + \frac{qf}{2} + \frac{e_2 - 1}{2}\\

So the effect of quantising and dequantising a value, as modelled by affine
arithmetic is:

.. math::

    \text{dequantise}(\text{quantise}(x, qf), qf) &=
        \left(\frac{x}{qf} + \frac{e_1 - 1}{2}\right) \times qf + \frac{qf}{2} + \frac{e_2 - 1}{2}\\
    &=
        x + qf \frac{e_1 - 1}{2} + \frac{qf}{2} + \frac{e_2 - 1}{2}\\
    &=
        \left[x - \frac{qf}{2} - 1, x + \frac{qf}{2}\right]\\

This tells us that for large quantisation factors (where :math:`qf \approx x`),
quantisation produces a range:

.. math::

    \text{dequantise}(\text{quantise}(x, x), x) =
        \left[\frac{x}{2} - 1, \frac{3x}{2}\right]

When negative numbers are taken into account, the affine range becomes:

.. math::

    \text{dequantise}(\text{quantise}(x, x), x) =
        \left[-\frac{3}{2}x, \frac{3}{2}x\right]


That is, worst-case quantisation has the effect of replacing the quantised
value with an affine error variable with a magnitude :math:`\frac{3}{2}\times`
the quantised value. In the next section we'll attempt to bound this range.
