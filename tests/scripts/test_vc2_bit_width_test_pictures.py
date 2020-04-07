import pytest

import os

import csv

import json

import numpy as np

import PIL.Image

from decode_and_quantise_test_utils import (
    encode_with_vc2,
    quantise_coeffs,
    decode_with_vc2,
)

from vc2_data_tables import WaveletFilters, QUANTISATION_MATRICES

from vc2_bit_widths.helpers import TestPoint as TP

from vc2_bit_widths.scripts.vc2_bit_width_test_pictures import (
    save_picture_as_png,
    save_test_points_as_json,
)

from vc2_bit_widths.scripts.vc2_static_filter_analysis import main as vc2_static_filter_analysis
from vc2_bit_widths.scripts.vc2_bit_widths_table import main as vc2_bit_widths_table
from vc2_bit_widths.scripts.vc2_bit_width_test_pictures import main as vc2_bit_width_test_pictures


def test_save_picture_as_png(tmpdir):
    filename = str(tmpdir.join("test.png"))
    
    picture = np.array([
        [1023, 0, 512],
        [0, 512, 1023],
    ], dtype=object)
    
    save_picture_as_png(filename, picture, 10)
    
    im = PIL.Image.open(filename)
    assert im.width == 3
    assert im.height == 2
    assert im.getpixel((0, 0)) == 255
    assert im.getpixel((1, 0)) == 0
    assert im.getpixel((2, 0)) == 128
    assert im.getpixel((0, 1)) == 0
    assert im.getpixel((1, 1)) == 128
    assert im.getpixel((2, 1)) == 255


def test_save_test_points_as_json(tmpdir):
    filename = str(tmpdir.join("test.json"))
    
    save_test_points_as_json(filename, [
        TP(1, "LL", 2, 3, True, 4, 5),
        TP(6, "HL", 7, 8, False, 9, 10),
    ])
    
    assert json.load(open(filename, "r")) == [
        {"level": 1, "array_name": "LL", "x": 2, "y": 3, "maximise": True, "tx": 4, "ty": 5},
        {"level": 6, "array_name": "HL", "x": 7, "y": 8, "maximise": False, "tx": 9, "ty": 10},
    ]


def test_correctness(tmpdir):
    # This integration test runs the generated test pictures through a VC-2
    # encoder/quantiser/decoder and verifies that the (observable) signal
    # levels match those predicted by the bit-widths table tool (and in the
    # locations indicated by the provided metadata).
    
    static_analysis_filename = str(tmpdir.join("static_analysis.json"))
    bit_widths_table_filename = str(tmpdir.join("bit_widths_table.csv"))
    test_pictures_dir = str(tmpdir.join("test_pictures"))
    os.mkdir(test_pictures_dir)
    
    wavelet_index = WaveletFilters.le_gall_5_3
    wavelet_index_ho = WaveletFilters.le_gall_5_3
    dwt_depth = 1
    dwt_depth_ho = 0
    quantisation_matrix = QUANTISATION_MATRICES[(
        wavelet_index,
        wavelet_index_ho,
        dwt_depth,
        dwt_depth_ho,
    )]
    picture_bit_width = 8  # Assumed to be 8 in rest of test...
    
    vc2_static_filter_analysis([
        "--wavelet-index", wavelet_index.name,
        "--wavelet-index-ho", wavelet_index_ho.name,
        "--dwt-depth", str(dwt_depth),
        "--dwt-depth-ho", str(dwt_depth_ho),
        "--output", static_analysis_filename,
    ])
    
    vc2_bit_widths_table([
        static_analysis_filename,
        "--picture-bit-width", str(picture_bit_width),
        "--show-all-filter-phases",
        "--output", bit_widths_table_filename,
    ])
    
    # Read bit widths table (to get model answers)
    # {(type, level, array_name, x, y, maximise): value, ...}
    expected_signal_values = {}
    for row in csv.DictReader(open(bit_widths_table_filename)):
        typename = row.pop("type")
        level = int(row.pop("level"))
        array_name = row.pop("array_name")
        x = int(row.pop("x"))
        y = int(row.pop("y"))
        minimise = int(row.pop("test_pattern_min"))
        maximise = int(row.pop("test_pattern_max"))
        
        expected_signal_values[(
            typename,
            level,
            array_name,
            x, y,
            False,
        )] = minimise
        
        expected_signal_values[(
            typename,
            level,
            array_name,
            x, y,
            True,
        )] = maximise
    
    vc2_bit_width_test_pictures([
        static_analysis_filename,
        "64", "32",  # Chosen to be multiple of required power of 2
        "--picture-bit-width", str(picture_bit_width),
        "--output-directory", test_pictures_dir,
    ])
    
    test_picture_filenames = os.listdir(test_pictures_dir)
    
    # Encode the analysis test pictures
    # {filename: encoded_data, ...}
    encoded_analysis_pictures = {}
    for filename in test_picture_filenames:
        if filename.startswith("analysis_") and filename.endswith(".png"):
            picture = np.asarray(PIL.Image.open(os.path.join(test_pictures_dir, filename)))
            picture = picture.astype(int) - 128
            
            encoded_analysis_pictures[filename] = encode_with_vc2(
                picture.tolist(),
                picture.shape[1],
                picture.shape[0],
                wavelet_index,
                wavelet_index_ho,
                dwt_depth,
                dwt_depth_ho,
            )
    
    # Encode/quantise/decode the synthesis test pictures
    # {filename: encoded_data, ...}
    decoded_synthesis_pictures = {}
    for filename in test_picture_filenames:
        if filename.startswith("synthesis_") and filename.endswith(".png"):
            picture = np.asarray(PIL.Image.open(os.path.join(test_pictures_dir, filename)))
            picture = picture.astype(int) - 128
            
            qi = int(filename.partition("qi")[2].partition(".")[0])
            
            transform_coeffs = encode_with_vc2(
                picture.tolist(),
                picture.shape[1],
                picture.shape[0],
                wavelet_index,
                wavelet_index_ho,
                dwt_depth,
                dwt_depth_ho,
            )
            
            quantised_coeffs = quantise_coeffs(transform_coeffs, qi, quantisation_matrix)
            
            decoded_synthesis_pictures[filename] = decode_with_vc2(
                quantised_coeffs,
                picture.shape[1],
                picture.shape[0],
                wavelet_index,
                wavelet_index_ho,
                dwt_depth,
                dwt_depth_ho,
            )
    
    # Extract the target values for the outputs of the encoded/decoded pictures
    # {(type, level, array_name, x, y, maximise): value, ...}
    actual_signal_values = {}
    for filename in test_picture_filenames:
        if filename.startswith("analysis_") and filename.endswith(".png"):
            json_filename = filename.replace(".png", ".json")
            test_points = json.load(open(os.path.join(test_pictures_dir, json_filename)))
            for test_point in test_points:
                # Convert L'' and H'' coordinates into LL, LH, HL, HH
                # coordinates (omitted since they're just an interleaving).
                # This assumes we've chosen a 2D transform with 2 lifting
                # stages in this test...
                if test_point["array_name"] not in ("L''", "H''"):
                    continue
                
                level = test_point["level"]
                
                tx = test_point["tx"]
                ty = test_point["ty"] // 2
                
                if test_point["y"] % 2 == 0:
                    subband = "LL" if test_point["array_name"] == "L''" else "HL"
                else:
                    subband = "LH" if test_point["array_name"] == "L''" else "HH"
                
                if subband == "LL":
                    level -= 1
                
                if subband == "LL" and level != 0:
                    continue
                
                value = encoded_analysis_pictures[filename][level][subband][ty][tx]
                
                actual_signal_values[(
                    "analysis",
                    test_point["level"],
                    test_point["array_name"],
                    test_point["x"],
                    test_point["y"],
                    test_point["maximise"],
                )] = value
    
    for filename in test_picture_filenames:
        if filename.startswith("synthesis_") and filename.endswith(".png"):
            json_filename = filename.replace(".png", ".json")
            test_points = json.load(open(os.path.join(test_pictures_dir, json_filename)))
            for test_point in test_points:
                # Only consider the output picture
                if test_point["array_name"] != "Output" or test_point["level"] != dwt_depth + dwt_depth_ho:
                    continue
                
                tx = test_point["tx"]
                ty = test_point["ty"]
                value = decoded_synthesis_pictures[filename][ty][tx]
                
                actual_signal_values[(
                    "synthesis",
                    test_point["level"],
                    test_point["array_name"],
                    test_point["x"],
                    test_point["y"],
                    test_point["maximise"],
                )] = value
    
    # Check the actual signal levels observed match the expectations
    for key, value in actual_signal_values.items():
        assert value == expected_signal_values[key]
