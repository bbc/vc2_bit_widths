"""
Choose the appropriate Fractions library (and associated functions) to use the
best version available on this machine.

The native Python :py:mod:`fractions` module is relatively inefficient but
heavily used by this software. Consequently it can be advantageous to use the
third-party `quicktions <https://pypi.org/project/quicktions/>`_ drop-in
replacement. Under CPython, speed-ups of 2x have been observed, though under
PyPy, it is generally better to use the native implementation.
"""

__all__ = [
    "Fraction",
    "gcd",
]


try:
    from quicktions import Fraction
except ImportError:
    from fractions import Fraction


try:
    from math import gcd  # Python >= 3.5
except ImportError:
    # Python < 3.5
    try:
        from quicktions import _gcd as gcd
    except ImportError:
        from fractions import gcd

