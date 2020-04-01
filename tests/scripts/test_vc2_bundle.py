import pytest

import json

from tempfile import NamedTemporaryFile

from vc2_bit_widths.scripts import (
    vc2_static_filter_analysis,
    vc2_optimise_synthesis_test_patterns,
    vc2_bundle,
)


@pytest.yield_fixture(scope="module")
def analysis_1():
    with NamedTemporaryFile() as f:
        vc2_static_filter_analysis.main([
            "-w", "haar_with_shift",
            "-D", "1",
            "-o", f.name,
        ])
        yield f.name


@pytest.yield_fixture(scope="module")
def analysis_2():
    with NamedTemporaryFile() as f:
        vc2_static_filter_analysis.main([
            "-w", "le_gall_5_3",
            "-D", "1",
            "-o", f.name,
        ])
        yield f.name


@pytest.yield_fixture(scope="module")
def optimised_1(analysis_1):
    with NamedTemporaryFile() as f:
        vc2_optimise_synthesis_test_patterns.main([
            analysis_1,
            "-b", "10",
            "-a", 0,
            "-i", 0,
            "-o", f.name,
        ])
        yield f.name


@pytest.yield_fixture(scope="module")
def optimised_2(analysis_2):
    with NamedTemporaryFile() as f:
        vc2_optimise_synthesis_test_patterns.main([
            analysis_2,
            "-b", "10",
            "-a", 0,
            "-i", 0,
            "-o", f.name,
        ])
        yield f.name

@pytest.yield_fixture(scope="module")
def bundle(analysis_1, analysis_2, optimised_1, optimised_2):
    with NamedTemporaryFile() as f:
        assert vc2_bundle.main([
            "create",
            f.name,
            "-s", analysis_1, analysis_2,
            "-o", optimised_1, optimised_2,
        ]) == 0
        yield f.name


def test_list_empty(tmpdir, capsys):
    bundle_filename = str(tmpdir.join("bundle.zip"))
    
    # Empty case
    assert vc2_bundle.main(["create", bundle_filename]) == 0
    assert vc2_bundle.main(["list", bundle_filename]) == 0
    assert capsys.readouterr() == ("Bundle is empty.\n", "")


def test_list_analysis_only(tmpdir, capsys, analysis_1, analysis_2):
    bundle_filename = str(tmpdir.join("bundle.zip"))
    assert vc2_bundle.main([
        "create",
        bundle_filename,
        "-s", analysis_1, analysis_2
    ]) == 0
    assert vc2_bundle.main(["list", bundle_filename]) == 0
    out, err = capsys.readouterr()
    assert out == (
        "Static filter analyses\n"
        "======================\n"
        "\n"
        "0.\n"
        "    * wavelet_index: haar_with_shift (4)\n"
        "    * wavelet_index_ho: haar_with_shift (4)\n"
        "    * dwt_depth: 0\n"
        "    * dwt_depth_ho: 1\n"
        "1.\n"
        "    * wavelet_index: le_gall_5_3 (1)\n"
        "    * wavelet_index_ho: le_gall_5_3 (1)\n"
        "    * dwt_depth: 0\n"
        "    * dwt_depth_ho: 1\n"
    )
    assert err == ""


def test_list_analysis_only(tmpdir, capsys, optimised_1, optimised_2):
    bundle_filename = str(tmpdir.join("bundle.zip"))
    assert vc2_bundle.main([
        "create",
        bundle_filename,
        "-o", optimised_1, optimised_2,
    ]) == 0
    assert vc2_bundle.main(["list", bundle_filename]) == 0
    out, err = capsys.readouterr()
    assert out == (
        "Optimised synthesis test patterns\n"
        "=================================\n"
        "\n"
        "0.\n"
        "    * wavelet_index: haar_with_shift (4)\n"
        "    * wavelet_index_ho: haar_with_shift (4)\n"
        "    * dwt_depth: 0\n"
        "    * dwt_depth_ho: 1\n"
        "    * picture_bit_width: 10\n"
        "    * quantisation_matrix: {\n"
        "          0: {'L': 4},\n"
        "          1: {'H': 0},\n"
        "      }\n"
        "1.\n"
        "    * wavelet_index: le_gall_5_3 (1)\n"
        "    * wavelet_index_ho: le_gall_5_3 (1)\n"
        "    * dwt_depth: 0\n"
        "    * dwt_depth_ho: 1\n"
        "    * picture_bit_width: 10\n"
        "    * quantisation_matrix: {\n"
        "          0: {'L': 2},\n"
        "          1: {'H': 0},\n"
        "      }\n"
    )
    assert err == ""


def test_list_both(tmpdir, capsys, analysis_1, optimised_1):
    bundle_filename = str(tmpdir.join("bundle.zip"))
    assert vc2_bundle.main([
        "create",
        bundle_filename,
        "-s", analysis_1,
        "-o", optimised_1,
    ]) == 0
    assert vc2_bundle.main(["list", bundle_filename]) == 0
    out, err = capsys.readouterr()
    assert out == (
        "Static filter analyses\n"
        "======================\n"
        "\n"
        "0.\n"
        "    * wavelet_index: haar_with_shift (4)\n"
        "    * wavelet_index_ho: haar_with_shift (4)\n"
        "    * dwt_depth: 0\n"
        "    * dwt_depth_ho: 1\n"
        "\n"
        "\n"
        "Optimised synthesis test patterns\n"
        "=================================\n"
        "\n"
        "0.\n"
        "    * wavelet_index: haar_with_shift (4)\n"
        "    * wavelet_index_ho: haar_with_shift (4)\n"
        "    * dwt_depth: 0\n"
        "    * dwt_depth_ho: 1\n"
        "    * picture_bit_width: 10\n"
        "    * quantisation_matrix: {\n"
        "          0: {'L': 4},\n"
        "          1: {'H': 0},\n"
        "      }\n"
    )
    assert err == ""

def test_extract_sfa_success(tmpdir, bundle, analysis_2):
    output_filename = str(tmpdir.join("out.json"))
    
    assert vc2_bundle.main([
        "extract-static-filter-analysis",
        bundle,
        "-w", "le_gall_5_3",
        "-D", "1",
        "-o", output_filename,
    ]) == 0
    assert (
        json.load(open(analysis_2))
        ==
        json.load(open(output_filename))
    )

def test_extract_sfa_fail(tmpdir, capsys, bundle):
    output_filename = str(tmpdir.join("out.json"))
    
    assert vc2_bundle.main([
        "extract-static-filter-analysis",
        bundle,
        "-w", "fidelity",
        "-D", "1",
        "-o", output_filename,
    ]) != 0

    assert capsys.readouterr() == (
        "",
        "No matching static filter analysis found in bundle.\n",
    )

@pytest.mark.parametrize("quant_index_arg", [
    [],
    [
        "-q",
        "0", "L", "4",
        "1", "H", "0",
    ],
])
def test_extract_ostp_success(tmpdir, bundle, optimised_1, quant_index_arg):
    output_filename = str(tmpdir.join("out.json"))
    assert vc2_bundle.main([
        "extract-optimised-synthesis-test-patterns",
        bundle,
        "-w", "haar_with_shift",
        "-D", "1",
        "-b", "10",
        "-o", output_filename,
    ] + quant_index_arg) == 0
    assert (
        json.load(open(optimised_1))
        ==
        json.load(open(output_filename))
    )

def test_extract_ostp_fail(tmpdir, capsys, bundle):
    output_filename = str(tmpdir.join("out.json"))
    
    assert vc2_bundle.main([
        "extract-optimised-synthesis-test-patterns",
        bundle,
        "-w", "fidelity",
        "-D", "1",
        "-b", "10",
        "-o", output_filename,
    ]) != 0

    assert capsys.readouterr() == (
        "",
        "No matching optimised synthesis test patterns found in bundle.\n",
    )
