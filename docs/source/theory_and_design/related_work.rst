.. _theory-related-work:

Related work
============

There are numerous examples of implementations of integer wavelet transforms in
the literature, however published attempts at selecting bit-widths for
arithmetic operations have been quite limited.

In [BaFT11]_ and [FaGa08]_, the authors take the potentially wasteful approach
of adding an extra bit after every addition and summing the bits after a
multiplication. In both cases, however, the authors fail to account for
accumulated signal growth in multi-level transforms.

The other common approach used by, for example, [SZAA02]_, [BRGT06]_ and
[HTLS08]_, is to pass a series of test pictures through a codec and observe the
largest signal values encountered. This approach is error prone since the
difference between 'worst case' signals and typical natural images can be
great, as shown later.

In general, the problem of determining how many bits are required for a given
integer arithmetic problem is NP-complete. For small problems involving a few
tens of variables, SAT-Modulo-Theory (SMT) solvers may be used to find exact
answers. For example, SMT solvers have been used to automatically identify
integer-overflow bugs in C and C++ code [MoBj10]_.

Video filters unfortunately represent considerably larger problems than SMT
solvers can tackle effectively, requiring hundreds or thousands of variables.

Kinsman and Nicolici ([KiNi10]_) demonstrated an approach to applying SMT
solvers to larger problems by terminating the SMT solver's search process after
a fixed timeout. The incomplete state of the SMT solver is then used to give an
(guaranteed over-estimated) upper-bound on signal values. Unfortunately, the
performance of this approach is dependent on the timeouts chosen and the
behaviour of the SMT solver during a given run. Further, the approach was only
demonstrated for small numbers of variables (low tens) and it is unclear that
it would scale well for considerably larger problems.

In the wider fields of digital signal processing and numerical methods,
Interval Arithmetic (IA) and Affine Arithmetic (AA) are widely used techniques
for bounding errors in linear functions in the presence of rounding and
quantisation. López and Carreras give a good introduction and discussion of
both techniques in [LoCN07]_.

Like Kinsman and Nicolici's truncated SMT approach, both IA and AA give hard
upper bounds on signal levels. Unfortunately both are prone to over-estimating
these bounds, with AA giving the tighter bounds. Nevertheless, these approaches
are deterministic and easy to compute.



.. rubric:: References

.. [BaFT11] Eric J. Balster; Benjamin T. Fortener; William F. Turri: "Integer
    Computation of Lossy JPEG2000 Compression". In: IEEE Transactions on Image
    Processing, August 2011.

.. [FaGa08]  Wenbing Fan; Yingmin Gao: "FPGA Design of Fast Lifting Wavelet
    Transform". In: Proceedings of the Congress on Image and Signal
    Processing, 2008.

.. [SZAA02] Vassilis Spiliotopoulos; N. D. Zcrvas; C. E. Androulidakis; G.
    Anagnostopoulos; S. Thcoharis: "Quantizing the 9/7 Daubechies filter
    coefficients for 2D DWT VLSI implementations". In: Proceedings of the
    International Conference on Digital Signal Processing, 2002.

.. [BRGT06] Stephen Bishop; Suresh Rai; B. Gunturk;  J. Trahan; R.
    Vaidyanathan: "Reconfigurable Implementation of Wavelet Integer Lifting
    Transforms for Image Compression". In: Proceedings of IEEE International
    Conference on Reconfigurable Computing and FPGAs, 2006.

.. [HTLS08] Chin-Fa Hsieh; Tsung-Han Tsai; Chih-Hung Lai; Tai-An Shan: "A Novel
    Efficient VLSI Architecture of 2-D Discrete Wavelet Transform". In:
    Proceedings of the International Conference on Intelligent Information
    Hiding and Multimedia Signal Processing, 2008.


.. [MoBj10] Leonardo de Moura; Nikolaj Bjørner: "Applications and Challenges in
    Satisfiability Modulo Theories". In: Proceedings of the Workshop on Invariant
    Generation, 2010.

.. [KiNi10] Adam B. Kinsman; Nicola Nicolici: "Bit-Width Allocation for Hardware
    Accelerators for Scientific Computing Using SAT-Modulo Theory". In: IEEE
    Transactions on Computer-Aided Design of Integrated Circuits and Systems,
    March 2010.

.. [LoCN07]  Juan A. López; Carlos Carreras; Octavio Nieto-Taladriz: "Improved
    Interval-Based Characterization ofFixed-Point LTI Systems With Feedback
    Loops".  In: IEEE Transactions on Computer-Aided Design of Integrated
    Circuits and Systems, November 2007.
