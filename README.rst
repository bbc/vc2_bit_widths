SMPTE ST 2042-1 (VC-2) Bit Width Calculation Software
=====================================================

**This repository is a work-in-progress.**

This Python package, ``vc2_bit_widths``, provides routines computing how many
bits of numerical precision are required for implementations of the for the
SMPTE ST 2042-1 `VC-2 professional video codec
<https://www.bbc.co.uk/rd/projects/vc-2>`_.


Installation
------------

(**Coming soon...**) You can install the ``vc2_bit_widths`` Python
module from `PyPI <https://pypi.org/>`_ using::

    $ pip install vc2_bit_widths

Alternatively you can install it from a copy of this repository using::

    $ python setup.py install


Tests
-----

To run the test suite, first install the test suite dependencies using::

    $ pip install -r requirements-test.txt

Then run the tests::

    $ pytest tests/

To automatically run the test suite under several versions of Python ``tox``
may be used::

    $ pip install tox
    $ tox


Documentation
-------------

To build the documentation, first install the build dependencies::

    $ pip install -r requirements-doc.txt

Then build the documentation::

    $ cd docs
    $ make html  # or make latexpdf 

The built (HTML) documentation can be found in `docs/build/html/index.html
<./docs/build/html/index.html>`_.


Author
------

This module is currently being developed by `Jonathan Heathcote
<mailto:jonathan.heathcote@bbc.co.uk>`_ from BBC R&D as part of a project to
refresh VC-2's conformance testing procedures.
