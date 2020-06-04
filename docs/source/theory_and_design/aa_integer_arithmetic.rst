.. _theory-affine-arithmetic:

Computing signal bounds with Affine Arithmetic
==============================================

Though VC-2 implements the discrete wavelet transform, a linear filter, integer
rounding and quantisation make VC-2 a non-linear filter. In this section we
describe the process by which Affine Arithmetic (AA) may be used to find
upper-bounds for signal ranges in VC-2.

Analysing linear filters
------------------------

Given an algebraic description of a linear filter, it is straight-forward to
determine the inputs which produce the most extreme output values.

For example, consider the following algebraic description which could describe
how a linear filter might compute the value of a particular output, given two
input pixel values :math:`a` and :math:`b`:

.. math::

    \frac{a}{2} - \frac{b}{8} + 1

In this expression :math:`a` is weighted with a positive coefficient
(:math:`\frac{1}{2}`) while :math:`b` is weighted with a negative coefficient
(:math:`-\frac{1}{8}`). As a consequence, to produce an output with the highest
possible value we should set :math:`a` to a large positive value and :math:`b`
to a large negative value. Conversely, the opposite is true if we wish to
produce the lowest possible value.

For example, if we define the input signal range as being :math:`[-100, 100]`,
it is maximum result, 63.5, is produced when :math:`a=100` and :math:`b=-100`
and the minimum result, -61.5, when :math:`a=-100` and :math:`b=100`.

In this way, given an algebraic description of a linear filter, we can compute
a set of worst-case input values (i.e. a test pattern) and also the output
signal range.


Affine arithmetic
-----------------

`Affine arithmetic <https://en.wikipedia.org/wiki/Affine_arithmetic>`_ provides
a way to bound the effects of non-linearities due to rounding errors.

In affine arithmetic, non-linear operations are modelled as linear operations
with error terms.

For example, when a value is divided by two using truncating integer arithmetic
(:math:`//`), this is modelled with affine arithmetic as:

.. math::
    
    x//2 = \frac{x}{2} + \frac{e_1 - 1}{2}

Where :math:`e_1` is an error term representing some value in the interval
:math:`[-1, +1]`.

As a result of the use of error terms, affine arithmetic expressions
effectively specify *ranges* of possible values. In the example above, that
range would be :math:`\left[\frac{x}{2} - 1, \frac{x}{2}\right]`.

Every time a non-linear operation is modelled using affine arithmetic a new
error term must be introduced, thereby (pessimistically) modelling all errors
as being independent. In practice, rounding errors are not independent and so
affine arithmetic will tend to indicate an overly broad range of values.

For example, if we substitute :math:`x=11` into :math:`x//2`, affine arithmetic
tells us that the answer lies in the range :math:`[4.5, 5.5]` which is true
(:math:`11//2 = 5`), but imprecise.

Nevertheless, affine arithmetic's pessimism guarantees that the true result is
*always* contained in the range indicated.

As a rule of thumb, so long as the rounding errors in an expression are small,
so too is the range indicated by affine arithmetic.


Worked example
--------------

The example below demonstrates the procedure used to find the theoretical
bounds of a filter.

Consider again the following filter on an input in the range :math:`[-100,
100]`:

.. math::

    \frac{a}{2} - \frac{b}{8} + 1

As before, we can use simple linear filter analysis to determine that the
filter value is maximised when :math:`a=100` and :math:`b=-100` and minimised
when :math:`a=-100` and :math:`b=100`.

Lets assume that the filter is approximated using truncating integer arithmetic
as:

.. math::

    (a+1)//2 - (b+4)//8 + 1

Represented using affine arithmetic we have:

.. math::

    \frac{a+1}{2} + \frac{e_1 - 1}{2} - \left(\frac{b+4}{8} + \frac{e_2 - 1}{2}\right) + 1

Substituting the minimising and maximising values for :math:`a` and :math:`b`
we find that the filter's minimum and maximum values lies in the following
ranges:

.. math::

    \text{maximum value} &=
        \frac{100+1}{2} + \frac{e_1 - 1}{2} - \left(\frac{-100+4}{8} + \frac{e_2 - 1}{2}\right) + 1 \\
    &=
        50.5 + \frac{e_1 - 1}{2} - \left(-12 + \frac{e_2 - 1}{2}\right) + 1 \\
    &=
        63.5 + \frac{e_1 - 1}{2} - \frac{e_2 - 1}{2} \\
    &=
        [62.5, 64.5] \\
    \\
    \text{minimum value} &=
        \frac{-100+1}{2} + \frac{e_1 - 1}{2} - \left(\frac{100+4}{8} + \frac{e_2 - 1}{2}\right) + 1 \\
    &=
        -49.5 + \frac{e_1 - 1}{2} - \left(13 + \frac{e_2 - 1}{2}\right) + 1 \\
    &=
        -61.5 + \frac{e_1 - 1}{2} - \frac{e_2 - 1}{2} \\
    &=
        [-62.5, -60.5] \\

From this we can therefore say that the output of our integer approximation of
the filter is bounded by the range :math:`[-62.5, 64.5]`.
