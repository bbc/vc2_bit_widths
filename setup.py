import os

import sys

from setuptools import setup, find_packages

version_file = os.path.join(
    os.path.dirname(__file__),
    "vc2_bit_widths",
    "version.py",
)
with open(version_file, "r") as f:
    exec(f.read())

install_requires = [
    "vc2_data_tables",
    "enum34",
    "six",
]

# Use old versions of libraries which have deprecated Python 2.7 support
if sys.version[0] == "2":
    install_requires.append("pillow<7")
    install_requires.append("numpy<1.17")
else:
    install_requires.append("pillow")
    install_requires.append("numpy")

setup(
    name="vc2_bit_widths",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/bbc/vc2_bit_widths",
    author="BBC R&D",
    description="Software for computing required bit widths for implementations of the SMPTE ST 2042-1 VC-2 professional video codec.",
    license="GPLv2",
    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Developers",

        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",

        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    keywords="vc2 dirac dirac-pro quantisation-matrix bit-width",
    install_requires=install_requires,
    entry_points = {
        "console_scripts": [
            "vc2-static-filter-analysis=vc2_bit_widths.scripts.vc2_static_filter_analysis:main",
            "vc2-static-filter-analysis-combine=vc2_bit_widths.scripts.vc2_static_filter_analysis_combine:main",
            "vc2-bit-widths-table=vc2_bit_widths.scripts.vc2_bit_widths_table:main",
            "vc2-bit-width-test-pictures=vc2_bit_widths.scripts.vc2_bit_width_test_pictures:main",
            "vc2-maximum-quantisation-index=vc2_bit_widths.scripts.vc2_maximum_quantisation_index:main",
            "vc2-optimise-synthesis-test-patterns=vc2_bit_widths.scripts.vc2_optimise_synthesis_test_patterns:main",
            "vc2-bundle=vc2_bit_widths.scripts.vc2_bundle:main",
        ],
    },
)
