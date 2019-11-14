from setuptools import setup, find_packages

with open("vc2_bit_widths/version.py", "r") as f:
    exec(f.read())

setup(
    name="vc2_bit_widths",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/bbc/vc2_bit_widths",
    author="BBC R&D",
    description="Software for computing required bit widths for implementations of the SMPTE ST 2042-2 VC-2 professional video codec.",
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
    keywords="smpte-RP-2042-3 vc2 dirac dirac-pro quantisation-matrix bit-width",
    install_requires=["vc2_data_tables", "enum34", "numpy", "six"],
    entry_points = {
        "console_scripts": [
            "vc2-static-filter-analysis=vc2_bit_widths.scripts.vc2_static_filter_analysis:main",
            "vc2-bit-widths-table=vc2_bit_widths.scripts.vc2_bit_widths_table:main",
            "vc2-maximum-quantisation-index=vc2_bit_widths.scripts.vc2_maximum_quantisation_index:main",
            "vc2-optimise-synthesis-test-patterns=vc2_bit_widths.scripts.vc2_optimise_synthesis_test_patterns:main",
        ],
    },
)
