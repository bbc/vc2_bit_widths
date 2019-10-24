.. _introduction:

Introduction and tutorial
=========================

The VC-2 standard defines the video decoding process using infinite precision
integer arithmetic. By contrast, real implementations will use fixed-width
integers sized appropriately for the content being coded. If codec is built
with too few bits of precision, potentially significant artefacts may result in
the presence of unfortunate signals. Using too many bits may also be
undesirable for efficiency or cost reasons.

Unfortunately, using real pictures and noise signals to experimentally
determine the necessary number of bits for a codec is prone to producing
under-estimates, particularly for deeper transform depths or low bit rates.

Perhaps surprisingly, the question 'how many bits do I need?' is not a simple
one to answer. This software attempts to provide reliable figures for necessary
bit widths in VC-2 implementations as well as providing test patterns which
produce extreme signals both internally and at the outputs of codec
implementations.

Before introducing this software it is important to understand its limitations
and the terminology it uses. These will be introduced in the next few sections
before the command line and Python library interfaces of
:py:mod:`vc2_bit_widths` is introduced.


Caveats
-------

While this software aims to produce robust bit width figures, it can only go as
far as the VC-2 specification and current mathematical techniques allow.


Assumed Encoder Behaviour
`````````````````````````

The VC-2 standard only defines the behaviour of a decoder. Unfortunately,
due to the clipper at the output, it is not possible to work backwards from the
output bit width to calculate the input or intermediate signal ranges (as
illustrated below).

.. image:: /_static/decoder_alone_not_enough.svg
    :alt: A VC-2 decoder with bit widths defined by the standard annotated.

Further, since the standard does not define valid ranges for values in a
bitstream, it is not possible to work forwards through the decoder and
determine the necessary bit widths. As such it is necessary to make assumptions
about the (unspecified) behaviour of a VC-2 encoder.

This software makes the assumption that all VC-2 encoders consist of a matching
forward discrete wavelet transform followed by a dead zone quantiser as
informatively suggested by the standard. Once this assumption has been made it
becomes possible to determine the bit widths of every part of a VC-2
encoder and decoder.

.. image:: /_static/working_out_bit_widths.svg
    :alt: A VC-2 encoder and decoder with inferred bit widths shown.

In principle, VC-2 encoder implementations are free to diverge from this
assumed behaviour and so may produce bitstreams with different signal ranges to
those predicted by this software. In practice, it is relatively unlikely to be
the case. Nevertheless, you should be aware that this software relies on this
assumption.


Non-linearity
`````````````

Though VC-2 is based on the (linear) wavelet transform, its use of integer
arithmetic and quantisation make it a non-linear filter. This non-linearity
makes it difficult to assess how signals will be transformed in worst-case
scenarios. As a consequence, this software is unable to calculate true
worst-case signal levels. Instead, it provides bounds guaranteed to contain
(but possibly over-estimate) worst case signals. Likewise, test pictures
produced by this tool are not guaranteed to reach true worst-case signal
levels, although they may be close.

.. _terminology:

Terminology
-----------

This software uses the following naming convention to refer to the different
values within an encoder and decoder.

Levels and subbands are numbered similarly to the VC-2 specification:

.. image:: /_static/level_numbering.svg
    :alt: Two examples showing 2D and asymmetric transforms.

The only departure from the VC-2 convention is that the DC-band is numbered '1'
and not '0' as in the VC-2 specification.

For a given transform level, the filtering process is broken down into a series
of steps which transform an array of input values into several (subsampled)
arrays during analysis (encoding) and the reverse during synthesis (decoding).

Analysis (Encoding)
```````````````````

.. image:: /_static/encoder_names.svg
    :alt: Names for the various synthesis intermediate values.

The figure above illustrates the steps involved in a single level of a 2D or
horizontal-only analysis transform level.

In the illustration two lifting stages are shown for each filter. For filters
with more than two lifting stages, the outputs of these stages follow the same
pattern. For example, for a Daubechies (9, 7), which has four lifting stages,
the additional stages' outputs arrays would be named ``DC'''``, ``DC''''``,
``L'''``, ``L''''``, ``H'''`` and ``H''''``.

Synthesis (Decoding)
````````````````````

.. image:: /_static/decoder_names.svg
    :alt: Names for the various analysis intermediate values.

The same naming convention is used in reverse for the steps involved in a 2D or
horizontal-only synthesis filtering stage.


Using the command-line tools
----------------------------

A series of command-line tools are provided for computing signal bounds and
test patterns for arbitrary VC-2 codec configurations.

The first step is to perform a static analysis of the wavelet filter used with
the :ref:`vc2-static-filter-analysis` command::

    $ vc2-static-filter-analysis \
        --wavelet-index le_gall_5_3 \
        --wavelet-index-ho haar_no_shift \
        --dwt-depth 1 \
        --dwt-depth-ho 2 \
        --output static_analysis.json

This command constructs a detailed algebraic description of the complete filter
specified and uses this to determine the relationship between pictures and
internal signal ranges. In addition, it also constructs test patterns which use
heuristics which attempt to produce extreme signal values.

.. warning::

    This command can take some time to run (e.g. a few minutes for particularly
    deep transforms or complex wavelets). The ``--verbose`` argument will give
    an indication of progress during particularly long runs.

The ``static_analysis.json`` file generated by the above example will contain
the JSON serialised output of the analysis which includes formulae for
calculating signal ranges and descriptions of test patterns (see
:ref:`vc2-static-filter-analysis-json` for details).


Tabulating bit width requirements
`````````````````````````````````

We can now turn this JSON file into a human-readable table of signal ranges for
particular picture bit depths using the :ref:`vc2-bit-widths-table` command::

    $ vc2-bit-widths-table \
        static_analysis.json \
        --picture-bit-widths 8 10 16 \
        --output bit_widths_table.csv

In the example above, we request that the bit widths required for 8, 10 and
16 bit input pictures are computed and written to ``bit_width_table.csv``. This
can be displayed in any spreadsheet package or, on UNIX-like systems, can be
displayed in tabular form using::

    $ column -t -s, bit_widths_table.csv

The table produced in the example above is shown (truncated) below:

=========  =====  ==========  =====  =====  ====  =====  =====  ====  =======  ======  ====
type       level  array_name  lower  upper  bits  lower  upper  bits  lower    upper   bits
=========  =====  ==========  =====  =====  ====  =====  =====  ====  =======  ======  ====
analysis   3      Input       -128   127    8     -512   511    10    -32768   32767   16
...        ...    ...         ...    ...    ...   ...    ...    ...   ...      ...     ...
analysis   1      L           -195   195    9     -771   771    11    -49155   49155   17
analysis   1      H           -388   389    10    -1540  1541   12    -98308   98309   18
synthesis  1      L           -272   272    10    -1086  1086   12    -69512   69512   18
synthesis  1      H           -543   543    11    -2173  2173   13    -139023  139023  19
...        ...    ...         ...    ...    ...   ...    ...    ...   ...      ...     ...
synthesis  3      Output      -1861  1858   12    -7424  7421   14    -474661  474658  20
=========  =====  ==========  =====  =====  ====  =====  =====  ====  =======  ======  ====

This table shows, for each input picture bit width specified, lower and upper
bounds for the signal levels in different parts of an analysis filter (encoder)
and synthesis filter (decoder). The 'bits' column gives the minimum number of
bits required to represent signed two's compliment integers in that range.

Each row is labelled with the transform level and array which the bounds apply
to according to the :ref:`naming convention <terminology>` defined earlier.
For example, the first row (analysis, level 3, 'Input') contains the signal
range for the picture presented to the analysis filter (encoder).

The row labelled (analysis, level 1, 'L') gives the signal range for the output
of the final DC band of the analysis filter. The row (synthesis, level 1, 'L')
gives the signal range for the same DC band input to the synthesis filter
(encoder). Notice that the signal range is larger: (-1086 1086) at the
synthesis input vs (-771, 771) at the analysis output. This is because the
signal ranges of the synthesis filter inputs are scaled up to account for the
worst-case effects of quantisation errors.

The signal bounds displayed use a mathematical technique called
:ref:`affine-arithmetic` to bound the worst-case impact of integer rounding
errors and quantisation. This technique guarantees that the signal ranges
produced are at least as wide as the true worst-case signal, therefore using
the number of bits specified in this table will always be sufficient for
correct behaviour. Unfortunately, these bounds tend to over-estimate the signal
bounds by an amount proportional to the size of the potential rounding errors.

In the analysis filter (encoder) rounding errors, and consequently the
over-estimate of signal bounds is likely to be very slight and so the number of
bits suggested in the table is likely to be the true minimum number of bits
required.

Values entering the synthesis filter (decoder) are the product of a
quantisation step which can, in extreme cases, introduce very large errors.
Consequently, the signal bounds for synthesis filters are likely to be
non-trivial over-estimates and so it is possible that fewer bits are required
than specified in the table.


Optimising test signals
```````````````````````

The test signals created by the :ref:`vc2-static-filter-analysis` command
are designed to be likely to produce extreme signal values in codecs using the
specified wavelet filters in the general case.

For analysis filters (encoders), the test signals produced by
:ref:`vc2-static-filter-analysis` are likely to produce signal levels very
close to the true worst case. The test signals work well here because the
synthesis filter only includes very slight non-linearities due to integer
rounding errors.

The test signals for synthesis filters (decoders), however, have to contend
with the strong non-linearity introduced by quantisation. The test signals are
designed to exacerbate the effects of these non linearities in the general
case.  However, non-linear effects differ significantly at different input
picture bit widths and when different quantisation matrices are used.

The :ref:`vc2-optimise-synthesis-test-signals` command uses an optimisation
algorithm to enhance the generic synthesis test signals for a particular codec
configuration (picture bit width and quantisation matrix). The resulting test
signals are highly specific to the chosen codec configuration and typically
demonstrate significantly wider signal ranges than the generic test signals.

The command may be used as follows::

    $ vc2-optimise-synthesis-test-signals \
        static_analysis.json \
        --picture-bit-width 10 \
        --output optimised_synthesis_test_signals.json

Custom quantisation matrices may be provided but the default quantisation
matrix will be used if none are specified.

The optimisation algorithm has a number of parameters which must be tuned to
achieve the best results. (See
:ref:`vc2-optimise-synthesis-test-signals-tuning`).

The optimisation process is computationally intensive and may take many hours
depending on the parameters chosen, the transform depth and wavelet complexity.
The ``--verbose`` flag may be used to track progress.

The optimised test patterns are output in JSON format to the specified file
(see :ref:`vc2-optimise-synthesis-test-signals-json` for details).
