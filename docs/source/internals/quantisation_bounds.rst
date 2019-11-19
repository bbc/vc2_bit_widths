Modelling quantisation errors
=============================




Affine arithmetic
-----------------

Affine arithmetic may also be used to model the effects of quantisation since
VC-2's dead-zone quantiser is essentially just truncating integer division.
However, in the most extreme cases, quantisation can introduce errors with the
same magnitude as the values being quantised, resulting in an extremely broad
range being generated.

Consider the following (simplified) definition of VC-2's quantiser and
dequantiser:

.. math::

    \begin{align}
        \text{quantise}(x, qf) &= x//qf \\
        \text{dequantise}(X, qf) &= X \times qf + \frac{qf}{2}
    \end{align}

In affine arithmetic this becomes:

.. math::

    \begin{align}
        \text{quantise}(x, qf) &= \frac{x}{qf} + \frac{e_1 - 1}{2}\\
        \text{dequantise}(X, qf) &= X \times qf + \frac{qf}{2} + \frac{e_2 - 1}{2}\\
    \end{align}

So the effect of quantising and dequantising a value, as modelled by affine
arithmetic is:

.. math::

    \begin{align}
        \text{dequantise}(\text{quantise}(x, qf), qf) &=
            \left(\frac{x}{qf} + \frac{e_1 - 1}{2}\right) \times qf + \frac{qf}{2} + \frac{e_2 - 1}{2}\\
        &=
            x + qf \frac{e_1 - 1}{2} + \frac{qf}{2} + \frac{e_2 - 1}{2}\\
        &=
            \left[x - \frac{qf}{2} - 1, x + \frac{qf}{2}\right]\\
    \end{align}

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
the quantised value.


Refining worst-case quantisation errors
---------------------------------------

Though in some cases quantisation can result in the quantised value growing by
a factor of :math:`\frac{3}{2}` as predicted by affine arithmetic, for many
values the worst-case gain is slightly lower.

For any value in the range :math:`[-x_{\text{max}}, x_{\text{max}}]`, the
largest value possible after quantisation and dequantisation is found by
quantising and dequantising :math:`x_{\text{max}}` by the largest quantisation
index which doesn't quantise the value to zero. Once known, this figure may be
used to define a slightly smaller affine range to represent the target value.

A formal proof of the property above is provided below for the interested
reader.

.. toctree::
    :maxdepth: 2
    
    quantisation_proof
