Computing theoretical filter bounds
===================================

VC-2 implements an integer approximation of the wavelet transform, a linear
filter. While linear filters are easily analysed to find their worst-case
outputs, once rounding and quantisation are introduced (as in VC-2), this is
no-longer the case.


Linear filters
--------------

Given an algebraic description of a linear filter, it is straight-forward to
determine the inputs which produce the most extreme output values. For example,
consider the following algebraic filter description:

.. math::

    \frac{a}{2} - \frac{b}{8} + 1

Where :math:`a` and :math:`b` are inputs to the filter.

If we define our input signal range as being :math:`[-100, 100]`, it is
straight-forward to see that the maximum result, 63.5, is produced when
:math:`a=100` and :math:`b=-100` and the minimum result, -61.5, when
:math:`a=-100` and :math:`b=100`.


Affine Arithmetic
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


Example
-------

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

    \begin{align}
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
      \end{align}

From this we can therefore say that the output of our integer approximation of
the filter is bounded by the range :math:`[-62.5, 64.5]`.
