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


.. _theory-affine-arithmetic-quantisation:

Quantisation and affine arithmetic
----------------------------------

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



.. _theory-quantisation-proof:

Refining worst-case quantisation error bounds
---------------------------------------------

Though in some cases quantisation can result in the quantised value growing by
a factor of :math:`\frac{3}{2}`, as predicted by affine arithmetic, for many
values the worst-case gain is lower.

For any value in the range :math:`[-x_{\text{max}}, x_{\text{max}}]`, the
largest value possible after quantisation and dequantisation is found by
quantising and dequantising :math:`x_{\text{max}}` by the largest quantisation
index which doesn't quantise the value to zero. Once known, this figure may be
used to define a slightly smaller affine range to represent the target value.

A formal proof of this statement is provided in the remainder of this section.

Quantisation using VC-2's quantisation/dequantisation process can fairly simply
be shown to produce outputs up to :math:`\frac{3}{2}\times` larger than the
original value. This proof sets out to show the exact upper bound of the output
of VC-2's quantiser given a defined input range.

Definitions
```````````

VC-2 defines a dead-zone quantiser based on an integer approximation of the
following:

.. math::
    
    \text{quantise}(x,\,qf) &= \left\{\begin{array}{ll}
        \left\lfloor\frac{x}{qf}\right\rfloor & \text{if}~x \ge 0 \\
        -\left\lfloor\frac{-x}{qf}\right\rfloor & \text{if}~x < 0 \\
    \end{array}\right.
    \\
    \text{dequantise}(x,\,qf) &= \left\{\begin{array}{ll}
        x \, qf + \frac{qf}{2} & \text{if}~x > 0 \\
        0 & \text{if}~x = 0 \\
        x \, qf - \frac{qf}{2} & \text{if}~x < 0 \\
    \end{array}\right.

Where :math:`x \in \mathbb{Z}` and :math:`qf` is the quantisation factor,
defined in terms of the quantisation index, :math:`qi`, as follows:

.. math::

    qf = 2^{qi/4} \quad \text{for} \quad qi \in \mathbb{N}

Formal problem statement
````````````````````````

Given some value, :math:`x_{\text{max}}`, what is the largest-magnitude value
which may be produced by
:math:`\text{dequantise}(\text{quantise}(x,\,qf),\,qf)` for
:math:`-x_{\text{max}} \le x \le x_{\text{max}}` and any :math:`qf`?


Lemmas
``````

In these lemmas the following shorthand notation will be used:

.. math::
    
    x'_{qf} = \text{dequantise}(\text{quantise}(x,\,qf),\,qf)

Without loss of generality, only non-negative values are considered below.
Analogous lemmas may be found trivially for the negative cases due to the
symmetry of the quantisation and dequantisation processes.

**Lemma 1:** When :math:`qf > x`, :math:`x'_{qf} = 0`.

*Proof:* When :math:`qf > x`, :math:`\text{quantise}(x,\,qf) =
\left\lfloor\frac{x}{qf}\right\rfloor` is trivially equal to zero. Since
:math:`\text{dequantise}(0,\, qf) = 0` by definition, :math:`x'_{qf} = 0` for
all cases where :math:`qf > x`.  QED.

**Lemma 2:** If :math:`x_1 \le x_2` then :math:`{x_1}'_{qf} \le {x_2}'_{qf}`.

That is, for a fixed :math:`qf`, VC-2's quantiser is monotonic: changing the
input value in one direction will never produce a change the output value in
the opposite direction.

*Proof:* For the case where :math:`x > 0`, the complete quantisation and
dequantisation process combine to form:

.. math::

    x'_{qf} = \left\lfloor\frac{x}{qf}\right\rfloor qf + \frac{qf}{2}

This expression is composed of only linear parts with the exception of the
floor operator which, nevertheless, is monotonic. As a consequence, the
operation as a whole is monotonic with a lower bound of 0.

In the case where :math:`x = 0`, :math:`x'_{qf} = 0`.

QED.

**Lemma 3:** For :math:`x > 0`, there exists at least one valid quantisation
factor in the range :math:`\frac{x}{2} < qf \le x`.

*Proof:* The quantisation factor is restricted to quarter powers of two
(:math:`qf = 2^{qi/4}` for :math:`qi \in \mathbb{N}`). Allowable quantisation
factors are therefore spaced a factor of :math:`2^{-1/4}` apart. Since the
range :math:`\frac{x}{2} < qf \le x` covers a range a factor of :math:`2^{1}`
wide, several quantisation factors will fall within this region. QED.

The above proof can be visualised using a log-scaled number line.  The
following figure shows a log-scale with valid quantisation factors marked,
along with an example :math:`x` and :math:`\frac{x}{2}` value. In this
illustration it can be seen that four valid quantisation factors in
:math:`\frac{x}{2} < qf \le x` will always be available in this range.

.. image:: /_static/log_numberline_valid_qf_always_exists.svg

**Lemma 4:** For :math:`\frac{x}{2} < qf \le x`, :math:`x'_{qf} = \frac{3}{2}
qf`.

That is, for the very largest quantisation factors which may be applied to
:math:`x`, without quantising it to zero (see lemma 1), the quantised value may
be computed by the linear expression :math:`x'_{qf} = \frac{3}{2} qf`.

*Proof:* When :math:`\frac{x}{2} < qf \le x`, :math:`\text{quantise}(x,\,qf) =
\left\lfloor\frac{x}{qf}\right\rfloor = 1`. As a consequence:

.. math::

    x'_{qf} &= \left\lfloor\frac{x}{qf}\right\rfloor qf + \frac{qf}{2} \\
            &= 1\,qf + \frac{qf}{2} \\
            &= \frac{3}{2} qf

**Lemma 5:** The largest quantisation factor in :math:`\frac{x}{2} < qf \le x`,
which we will call :math:`qf_{\text{max},x}`, produces the largest dequantised
value for any quantisation factor in this range.

*Proof:* In the range :math:`\frac{x}{2} < qf \le x`, lemma 4 shows that
:math:`x'_{qf} = \frac{3}{2} qf`. This expression is linear and therefore
monotonic. Therefore, the largest :math:`qf` allowed also produces the largest
:math:`x'_{qf}` possible in this range. QED.

**Lemma 6:** :math:`2^{-1/4} x < qf_{\text{max},x} \le x`

*Proof:* By definition (lemma 5) the upper bound of :math:`qf_{\text{max},x}`
is :math:`qf_{\text{max},x} \le x`.

Since quantisation factors are spaced at intervals of :math:`2^{1/4}`,
the lower bound is therefore :math:`2^{-1/4} x`.

QED.

**Lemma 7:** If :math:`qf_{\text{max},x}` is the largest :math:`qf \le x`,
:math:`\frac{qf_{\text{max},x}}{2}` is the largest :math:`qf \le \frac{x}{2}`.

*Proof:* The following visual illustration shows a log-scaled number line on
which the above values are plotted.

.. image:: /_static/log_numberline_half_qf_max_for_half_x.svg

On a log scale, scaling :math:`x` to :math:`\frac{x}{2}` moves by a factor of
:math:`2^{-1}` to the left, or exactly four quantisation factors. As a
consequence, the nearest quantisation factor below :math:`\frac{x}{2}` is also
four quantisation factors to the left of :math:`qf_{\text{max},x}`, that is
:math:`\frac{qf_{\text{max},x}}{2}`.

QED.

**Lemma 8:** :math:`x'_{qf} \le x + \frac{qf}{2}` and therefore we have an
upper bound on :math:`x'_{qf}` which is monotonic with :math:`qf`.

*Proof:* For :math:`x > 0`:

.. math::

    x'_{qf} = \left\lfloor\frac{x}{qf}\right\rfloor qf + \frac{qf}{2}

The effect of the floor operation can be replaced with an error term,
:math:`0 \le e < 1`:

.. math::

    x'_{qf} &= \left(\frac{x}{qf} - e\right) qf + \frac{qf}{2} \\
            &= x - e\,qf + \frac{qf}{2}

Therefore we get the upper bound:

.. math::

    x'_{qf} \le x + \frac{qf}{2}

Which is linear and, consequently, monotonic with :math:`qf`.

QED.


**Lemma 9:** :math:`x'_{qf} < x'_{qf_{\text{max},x}}` for all :math:`qf` in the
region :math:`1 \le qf \le \frac{x}{2}`.

*Proof:* By lemma 7, the largest quantisation factor in the range :math:`1 \le
qf \le \frac{x}{2}` is :math:`\frac{qf_{\text{max},x}}{2}`. Lemma 8 tells us
that this quantisation factor also gives an upper-bound on :math:`x'_{qf}` for
:math:`1 \le qf \le \frac{x}{2}`:

.. math::
    x'_{\frac{qf_{\text{max},x}}{2}} &\le x + \frac{qf_{\text{max},x}/2}{2} \\

Since :math:`\frac{qf_{\text{max},x}}{2} \le \frac{x}{2}` (lemmas 6 and 7), we
can substitute the former for the latter in the inequality to get:

.. math::
    x'_{\frac{qf_{\text{max},x}}{2}} &\le x + \frac{x/2}{2} \\
                                     &\le x + \frac{x}{4} \\
                                     &\le \frac{5}{4} x \\
                                     &\le 1.25\,x

Lemma 4 states that:

.. math::

    x'_{qf_{\text{max},x}} = \frac{3}{2} qf_{\text{max},x}

Lemma 6 gives a lower-bound for :math:`qf_{\text{max},x}` in terms of
:math:`x`, leading to the inequality:

.. math::

    x'_{qf_{\text{max},x}} &> \frac{3}{2} 2^{-1/4} x \\
                           &> 1.261\ldots\,x

From this we can conclude that:

.. math::
    x'_{\frac{qf_{\text{max},x}}{2}} < x'_{qf_{\text{max},x}}

And since we considered the upper-bound for
:math:`x'_{\frac{qf_{\text{max},x}}{2}}`, which is monotonic with :math:`qf`
(lemma 8), we can therefore state that:

.. math::
    x'_{qf} < x'_{qf_{\text{max},x}} \quad \text{for} \quad 1 \le qf \le \frac{x}{2}

QED.

**Lemma 10:** The largest :math:`x'_{qf}` for any :math:`qf` is produced for
the largest non-zero-producing quantisation factor, :math:`qf_{\text{max},x}`.

*Proof:* From the lemmas above:

* There exists at least one quantisation factor in the range
  :math:`\frac{x}{2} < qf_{\text{max},x} \le x` (lemma 3).
* Within this range, the largest quantisation factor,
  :math:`qf_{\text{max},x}`, also produces the largest dequantised value,
  :math:`x'_{qf_{\text{max},x}}` (lemma 5).
* For :math:`qf > qf_{\text{max},x}` we get :math:`x'_{qf} = 0` (lemma 1).
* For :math:`qf \le \frac{x}{2}` have shown that :math:`x'_{qf} <
  x'_{qf_{\text{max},x}}` (lemma 9).

Therefore, :math:`x'_{qf} \le x'_{qf_{\text{max},x}}` for *all* :math:`qf`.

QED.


Problem solution
````````````````

Using the lemmas above we are able to define a solution to our original
problem statement, repeated here for convenience:

    Given some value, :math:`x_{\text{max}}`, what is the largest-magnitude
    value which may be produced by
    :math:`\text{dequantise}(\text{quantise}(x,\,qf),\,qf)` for
    :math:`-x_{\text{max}} \le x \le x_{\text{max}}` and any :math:`qf`?

Lemma 10 tells us that the largest dequantised value for :math:`x_{\text{max}}`
will be :math:`qf_{\text{max},x_{\text{max}}}`, that is, the largest
quantisation factor that doesn't quantise :math:`x_{\text{max}}` to zero. Any
other quantisation factor will never quantise :math:`x_{\text{max}}` to a
larger value. Lemma 2 tells us that replacing :math:`x_{\text{max}}` with any
:math:`x < x_{\text{max}}` will also never produce a larger dequantised value.

The solution to the problem, therefore, is:

.. math::

    \text{largest dequantised value} =
        \text{dequantise}(
            \text{quantise}(
                x_{\text{max}},\,
                qf_{\text{max},x_{\text{max}}}
            ),\,
            qf_{\text{max},x_{\text{max}}}
        )

QED.


Validity under integer arithmetic
`````````````````````````````````

Under VC-2's integer arithmetic, all fractional values are truncated towards
zero, that is, results are monotonically adjusted downward in magnitude. As a
consequence, the monotonicity-related results for the lemmas above hold.

The function :math:`2^{qi/4}` is approximated in fixed-point arithmetic by the
function ``quant_factor`` in the VC-2 specification. This approximation is
accurate to the full precision of the arithmetic used up to quantisation index
134, corresponding with a quantisation factor of :math:`2^{33.5}`.  In real
applications (which use substantially smaller quantisation factors), the
approximation is accurate.

Finally, to give additional confidence, this solution has been verified
empirically for all 20 bit integers.
