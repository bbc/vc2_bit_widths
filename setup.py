import os

from setuptools import setup, find_packages

version_file = os.path.join(
    os.path.dirname(__file__),
    "vc2_bit_widths",
    "version.py",
)
with open(version_file, "r") as f:
    exec(f.read())

readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
with open(readme_file, "r") as f:
    long_description = f.read()

setup(
    name="vc2_bit_widths",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/bbc/vc2_bit_widths",
    author="BBC R&D",
    description="Software for computing required bit widths for implementations of the SMPTE ST 2042-1 VC-2 professional video codec.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="GPL-3.0-only",
    classifiers=[
        "Development Status :: 5 - Production/Stable",

        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Telecommunications Industry",

        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",

        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="vc2 dirac dirac-pro quantisation-matrix bit-width",
    install_requires=[
        "vc2_data_tables >=0.1,<2.0",
        # Use old versions/polyfill libraries which have deprecated older Python
        # version support
        "six",
        "enum34; python_version<'3.4'",
        "pillow<7; python_version<'3.0'",
        "pillow; python_version>='3.0'",
        "numpy<1.17; python_version<'3.0'",
        "numpy<1.20; python_version>='3.0' and python_version<'3.7'",
        "numpy; python_version>='3.7'",
    ],
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
