r"""
This script uses a modified version of the VC-2 pseudocode to encode and decode
a real picture, tracking the signal ranges used throughout.

.. tip::

    This script performs *substantially* faster under PyPy than under the usual
    CPython interpreter. It is well worth setting up for anything more than a
    quick test.

Example usage to measure the signal range of at different parts of the
encoder/decoder when encoding/decoding an HD frame of saturated noise::

    $ python measure_picture_signal_ranges.py \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 4 \
        --bit-width 10 \
        --saturated-noise 1920 1080 1 \
        --output results.csv
    
    $ (head results.csv; echo ...; tail results.csv) | column -t -s,
    picture                            qi  bits_per_pixel      psnr               type       level  array_name  minimum  maximum
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      Input       -512     511
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      DC          -1024    1022
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      DC'         -2046    2046
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      DC''        -2046    2046
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      L           -1535    1534
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      H           -2046    2046
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      L'          -3069    3069
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      L''         -3069    3069
    saturated_noise[1920x1080 seed=1]  NA  NA                  NA                 analysis   4      H'          -4092    4092
    ...
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      L''         0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      H''         0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      L'          0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      L           0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      H'          0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      H           0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      DC''        0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      DC'         0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      DC          0        0
    saturated_noise[1920x1080 seed=1]  58  1.0074074074074073  6.052646670823556  synthesis  4      Output      0        0

In the resulting CSV formatted table, the following columns are defined:

* picture (str): A name identifying what picture has been measured.
* qi (int or NA): The quantisation index used to quantise all transform
  coefficients. Set to NA for all analysis transform values.
* bits_per_pixel (number or NA): The average number of bits per pixel used for
  the current QI.
* psnr (number or NA): The peak signal to noise ratio of the decoded vs
  original image for the current QI.
* type ('analysis' or 'synthesis') Identity of the signal being measured.
* level (int) Identity of the signal being measured.
* array_name (str) Identity of the signal being measured.
* minimum (int): The lowest value observed for this signal.
* maximum (int): The highest value observed for this signal.

"""

import imageio

from argparse import ArgumentParser, FileType

import sys

import csv

import copy

import numpy as np

from collections import OrderedDict

from vc2_data_tables import WaveletFilters, LIFTING_FILTERS, QUANTISATION_MATRICES

from vc2_bit_widths.signal_bounds import signed_integer_range

from vc2_bit_widths.scripts.argument_parsers import (
    wavelet_index_or_name,
    parse_quantisation_matrix_argument,
)

from vc2_conformance.bitstream.exp_golomb import signed_exp_golomb_length

from vc2_conformance.picture_encoding import (
    ANALYSIS_LIFTING_FUNCTION_TYPES,
)

from vc2_conformance.picture_decoding import (
    filter_bit_shift,
    SYNTHESIS_LIFTING_FUNCTION_TYPES,
)

from vc2_conformance.arrays import (
    new_array,
    width,
    height,
    row,
    column,
)

from vc2_bit_widths.quantisation import forward_quant, inverse_quant

from vc2_conformance import picture_decoding
from vc2_conformance import picture_encoding

################################################################################
# Modified VC-2 pseudocode
# ========================
#
# The following code monkey-patches the VC-2 pseudocode in the vc2_conformance
# module to track signal ranges at each part of the filtering process. This
# tracking relies on the following values which have been retro-fitted to the
# state dictionary:
#
# signal_range_log : {(level, array_name): (minimum, maximum), ...}
#     For each level/array name (as named by vc2_bit_widths), the minimum and
#     maximum value encountered during analysis or synthesis. Should be reset
#     to an empty dictionary before each analysis/synthesis run.
# cur_level : int
#     The current transform level. Should be set to (dwt_depth + dwt_depth_ho)
#     before analysis begins and 1 before synthesis begins.
# cur_base_array_name : str
#     Used internally to track which array is being processed.
################################################################################

def monkeypatch(module):
    """
    Decorator. Replace a function in the specified module with the decorated
    function. For example:
    
        >>> from vc2_conformance import picture_encoding
        >>> @monkeypatch(picture_encoding)
        ... def h_analysis(state, data):
        ...     # New implementation for picture_encoding.h_analysis here!
        ...     pass
    """
    def patch(f):
        setattr(module, f.__name__, f)
        return f
    return patch


def expand_range(state, array_name, a):
    """
    Expand the recorded range of values in the named array using the values
    found in the array 'a'.
    """
    level = state["cur_level"]
    
    minimum, maximum = state["signal_range_log"].get((level, array_name), (0, 0))
    
    if isinstance(a[0], list):
        for sub_a in a:
            for value in sub_a:
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
    else:
        for value in a:
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
    
    state["signal_range_log"][(level, array_name)] = (minimum, maximum)


def reset_bit_width_log_for_analysis(state):
    state["cur_level"] = state["dwt_depth"] + state["dwt_depth_ho"]
    state["cur_base_array_name"] = None
    state["signal_range_log"] = OrderedDict()


def reset_bit_width_log_for_synthesis(state):
    state["cur_level"] = 1
    state["cur_base_array_name"] = None
    state["signal_range_log"] = OrderedDict()


@monkeypatch(picture_encoding)
def h_analysis(state, data):
    # Bit shift, if required
    expand_range(state, "Input", data)
    shift = filter_bit_shift(state)
    if shift > 0:
        for y in range(0, height(data)):
            for x in range(0, width(data)):
                data[y][x] = data[y][x] << shift
    expand_range(state, "DC", data)
    
    # Analysis
    state["cur_base_array_name"] = "DC"
    for y in range(0, height(data)):
        oned_analysis(row(data, y), state["wavelet_index_ho"])
    
    # De-interleave the transform data
    L_data = new_array(width(data) // 2, height(data))
    H_data = new_array(width(data) // 2, height(data))
    for y in range(0, (height(data))):
        for x in range(0, (width(data) // 2)):
            L_data[y][x] = data[y][2*x]
            H_data[y][x] = data[y][(2*x) + 1]
    
    expand_range(state, "L", L_data)
    expand_range(state, "H", H_data)
    
    state["cur_level"] -= 1
    
    return (L_data, H_data)


@monkeypatch(picture_encoding)
def vh_analysis(state, data):
    # Bit shift, if required
    expand_range(state, "Input", data)
    shift = filter_bit_shift(state)
    if shift > 0:
        for y in range(0, height(data)):
            for x in range(0, width(data)):
                data[y][x] = data[y][x] << shift
    
    expand_range(state, "DC", data)
    
    # Analysis
    state["cur_base_array_name"] = "DC"
    for y in range(0, height(data)):
        oned_analysis(row(data, y), state["wavelet_index_ho"])
    expand_range(state, "L", [r[0::2] for r in data])
    expand_range(state, "H", [r[1::2] for r in data])
    for x in range(0, width(data)):
        state["cur_base_array_name"] = "L" if x % 2 == 0 else "H"
        oned_analysis(column(data, x), state["wavelet_index"])
    
    # De-interleave the transform data
    LL_data = new_array(width(data) // 2, height(data) // 2)
    HL_data = new_array(width(data) // 2, height(data) // 2)
    LH_data = new_array(width(data) // 2, height(data) // 2)
    HH_data = new_array(width(data) // 2, height(data) // 2)
    for y in range(0, (height(data) // 2)):
        for x in range(0, (width(data) // 2)):
            LL_data[y][x] = data[2*y][2*x]
            HL_data[y][x] = data[2*y][2*x + 1]
            LH_data[y][x] = data[2*y + 1][2*x]
            HH_data[y][x] = data[2*y + 1][2*x + 1]
    
    expand_range(state, "LL", LL_data)
    expand_range(state, "LH", LH_data)
    expand_range(state, "HL", HL_data)
    expand_range(state, "HH", HH_data)
    
    state["cur_level"] -= 1
    
    return (LL_data, HL_data, LH_data, HH_data)


@monkeypatch(picture_encoding)
def oned_analysis(A, filter_index):
    filter_params = LIFTING_FILTERS[filter_index]
    
    for i, stage in enumerate(reversed(filter_params.stages)):
        lift_fn = ANALYSIS_LIFTING_FUNCTION_TYPES[stage.lift_type]
        lift_fn(A, stage.L, stage.D, stage.taps, stage.S)
        expand_range(state, state["cur_base_array_name"] + ("'"*(i+1)), A)


@monkeypatch(picture_decoding)
def h_synthesis(state, L_data, H_data):
    synth = new_array(2 * width(L_data), height(L_data))
    
    expand_range(state, "L", L_data)
    expand_range(state, "H", H_data)
    
    # Interleave transform data (as expected by synthesis routine)
    for y in range(0, (height(synth))):
        for x in range(0, (width(synth)//2)):
            synth[y][2*x] = L_data[y][x]
            synth[y][(2*x) + 1] = H_data[y][x]
    
    filter_params = LIFTING_FILTERS[state["wavelet_index_ho"]]
    expand_range(state, "DC" + "'"*len(filter_params.stages), synth)
    
    # Synthesis
    state["cur_base_array_name"] = "DC"
    for y in range(0, height(synth)):
        oned_synthesis(row(synth, y), state["wavelet_index_ho"])
    
    # Bit shift, if required
    shift = filter_bit_shift(state)
    if shift > 0:
        for y in range(0, height(synth)):
            for x in range(0, width(synth)):
                synth[y][x] = (synth[y][x] + (1 << (shift - 1))) >> shift
    
    expand_range(state, "Output", synth)
    
    state["cur_level"] += 1
    
    return synth


@monkeypatch(picture_decoding)
def vh_synthesis(state, LL_data, HL_data, LH_data, HH_data):
    expand_range(state, "LL", LL_data)
    expand_range(state, "LH", LH_data)
    expand_range(state, "HL", HL_data)
    expand_range(state, "HH", HH_data)
    
    synth = new_array(2 * width(LL_data), 2 * height(LL_data))
    
    # Interleave transform data (as expected by synthesis routine)
    for y in range(0, (height(synth)//2)):
        for x in range(0, (width(synth)//2)):
            synth[2*y][2*x] = LL_data[y][x]
            synth[2*y][2*x + 1] = HL_data[y][x]
            synth[2*y + 1][2*x] = LH_data[y][x]
            synth[2*y + 1][2*x + 1] = HH_data[y][x]
    
    filter_params = LIFTING_FILTERS[state["wavelet_index"]]
    expand_range(state, "L" + "'"*len(filter_params.stages), [r[0::2] for r in synth])
    expand_range(state, "H" + "'"*len(filter_params.stages), [r[1::2] for r in synth])
    
    # Synthesis
    for x in range(0, width(synth)):
        state["cur_base_array_name"] = "L" if x % 2 == 0 else "H"
        oned_synthesis(column(synth, x), state["wavelet_index"])
    
    filter_params = LIFTING_FILTERS[state["wavelet_index_ho"]]
    expand_range(state, "DC" + "'"*len(filter_params.stages), synth)
    
    state["cur_base_array_name"] = "DC"
    for y in range(0, height(synth)):
        oned_synthesis(row(synth, y), state["wavelet_index_ho"])
    
    # Bit shift, if required
    shift = filter_bit_shift(state)
    if shift > 0:
        for y in range(0, height(synth)):
            for x in range(0, width(synth)):
                synth[y][x] = (synth[y][x] + (1 << (shift - 1))) >> shift
    
    expand_range(state, "Output", synth)
    
    state["cur_level"] += 1
    
    return synth


@monkeypatch(picture_decoding)
def oned_synthesis(A, filter_index):
    filter_params = LIFTING_FILTERS[filter_index]
    
    for i, stage in enumerate(filter_params.stages):
        lift_fn = SYNTHESIS_LIFTING_FUNCTION_TYPES[stage.lift_type]
        lift_fn(A, stage.L, stage.D, stage.taps, stage.S)
        expand_range(state, state["cur_base_array_name"] + ("'"*(len(filter_params.stages)-i-1)), A)


################################################################################
# Encode/Quantise/Decode routines
################################################################################

def quantise_coeffs(state, coeffs, quantisation_index):
    """
    Quantise and dequantise a complete set of transform coefficients, also
    reporting the number of bits used to encode the quantised values using exp
    golomb coding.
    """

    num_bits = [0]

    def quant_dequant(level, orient, value):
        qi = max(0, quantisation_index - state["quant_matrix"][level][orient])
        
        quantised_value = forward_quant(value, qi)
        num_bits[0] += signed_exp_golomb_length(quantised_value)
        
        return inverse_quant(quantised_value, qi)
    
    quantised_coeffs = {
        level: {
            orient: [
                [
                    quant_dequant(level, orient, value)
                    for value in row
                ]
                for row in rows
            ]
            for orient, rows in orients.items()
        }
        for level, orients in coeffs.items()
    }
    
    return quantised_coeffs, num_bits[0]


def pad_picture(state, picture):
    """
    Given an array of picture data, add zero-padding values at the bottom/right
    edges as required by the wavelet transform.
    """
    picture = np.array(picture)
    height, width = picture.shape
    
    x_multiple = 2**(state["dwt_depth"] + state["dwt_depth_ho"])
    y_multiple = 2**(state["dwt_depth"])
    
    new_width = ((width+x_multiple-1) // x_multiple) * x_multiple
    new_height = ((height+y_multiple-1) // y_multiple) * y_multiple
    
    padded_picture = np.zeros((new_height, new_width), dtype=picture.dtype)
    padded_picture[:height, :width] = picture
    
    return padded_picture.tolist()

def encode(state, picture):
    """
    Encode the supplied picture. Returns a (coeffs, signal_range_log) pair.
    """
    reset_bit_width_log_for_analysis(state)
    coeffs = picture_encoding.dwt(state, copy.deepcopy(picture))
    return (coeffs, state["signal_range_log"])

def quantise_and_decode(state, coeffs, qi):
    """
    Decode the supplied picture based on the supplied transform coefficients
    after quantisation. Returns a (picture, num_bits, signal_range_log) triple.
    """
    reset_bit_width_log_for_synthesis(state)
    coeffs, num_bits = quantise_coeffs(state, coeffs, qi)
    picture = picture_decoding.idwt(state, coeffs)
    return (picture, num_bits, state["signal_range_log"])

################################################################################
# Parse command line arguments
################################################################################

parser = ArgumentParser()

source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument(
    "--noise", "-n",
    nargs=3, metavar=("WIDTH", "HEIGHT", "SEED"), type=int,
    help="""
        Encode and decode a random noise signal. Signal values will range from
        the minimum for the specified bit width through to the maximum.
    """,
)
source_group.add_argument(
    "--saturated-noise", "-N",
    nargs=3, metavar=("WIDTH", "HEIGHT", "SEED"), type=int,
    help="""
        Encode and decode a random noise signal. Signal values will consist of
        only minimum or maximum signal values for the specified bit width.
    """,
)
source_group.add_argument(
    "--picture", "-p",
    nargs=2, metavar=("FILENAME", "CHANNEL"),
    help="""
        Encode and decode a picture loaded from the specified image file.
    """,
)
parser.add_argument(
    "--wavelet-index", "-w",
    type=wavelet_index_or_name, required=True,
    help="""
        The VC-2 wavelet index for the wavelet transform. One of: {}.
    """.format(", ".join(
        "{} or {}".format(int(index), index.name)
        for index in WaveletFilters
    )),
)
parser.add_argument(
    "--wavelet-index-ho", "-W",
    type=wavelet_index_or_name,
    help="""
        The VC-2 wavelet index for the horizontal parts of the wavelet
        transform. If not specified, assumed to be the same as
        --wavelet-index/-w.
    """,
)
parser.add_argument(
    "--dwt-depth", "-d",
    type=int, default=0,
    help="""
        The VC-2 transform depth. Defaults to 0 if not specified.
    """
)
parser.add_argument(
    "--dwt-depth-ho", "-D",
    type=int, default=0,
    help="""
        The VC-2 horizontal-only transform depth. Defaults to 0 if not
        specified.
    """
)

parser.add_argument(
    "--bit-width", "-b",
    type=int, metavar="BITS", required=True,
    help="""
        The number of bits in the pixel values.
    """
)

parser.add_argument(
    "--custom-quantisation-matrix", "-q",
    nargs="+", dest="quant_matrix",
    help="""
        Define the custom quantisation matrix used by a codec. Optional except
        for filters without a default quantisation matrix defined.  Should be
        specified as a series 3-argument tuples giving the level, orientation
        and quantisation matrix value for every entry in the quantisation
        matrix.
    """
)

parser.add_argument(
    "--verbose", "-v", default=0, action="count",
    help="""
        Show more detailed status information during execution.
    """
)

parser.add_argument(
    "--output", "-o", default=sys.stdout, type=FileType("w"),
    help="""
        Filename to write the signal range information CSV to. Defaults to
        stdout.
    """
)

args = parser.parse_args()

if args.wavelet_index_ho is None:
    args.wavelet_index_ho = args.wavelet_index

args.quant_matrix = parse_quantisation_matrix_argument(
    args.quant_matrix,
    args.wavelet_index,
    args.wavelet_index_ho,
    args.dwt_depth,
    args.dwt_depth_ho,
)

################################################################################
# Read the picture
################################################################################

state = dict(
    wavelet_index=args.wavelet_index,
    wavelet_index_ho=args.wavelet_index_ho,
    dwt_depth=args.dwt_depth,
    dwt_depth_ho=args.dwt_depth_ho,
    quant_matrix=args.quant_matrix,
)

picture_bit_width = args.bit_width
signal_min, signal_max = signed_integer_range(picture_bit_width)

if args.noise is not None:
    input_width, input_height, seed = args.noise
    
    picture_name = "noise[{}x{} seed={}]".format(input_width, input_height, seed)

    rand = np.random.RandomState(seed)
    picture = rand.randint(
        signal_min,
        signal_max,
        (input_height, input_width),
        dtype=int,
    )
elif args.saturated_noise is not None:
    input_width, input_height, seed = args.saturated_noise
    
    picture_name = "saturated_noise[{}x{} seed={}]".format(input_width, input_height, seed)

    rand = np.random.RandomState(seed)
    picture = pad_picture(state, rand.choice(
        (signal_min, signal_max),
        (input_height, input_width),
    ))
elif args.picture is not None:
    filename, channel = args.picture
    channel = int(channel)
    
    picture = imageio.imread(filename)
    picture_name = "{}[{}]".format(filename, channel)
    
    # Extract desired channel
    if len(picture.shape) == 3:
        picture = picture[:, :, channel]
    else:
        assert len(picture.shape) == 2
        assert channel == 0
    
    input_height, input_width = picture.shape
    
    # Convert to expected signal range
    native_max = 1.0
    if picture.dtype == np.uint8:
        native_max = (1<<8) - 1
    elif picture.dtype == np.uint16:
        native_max = (1<<16) - 1
    else:
        raise TypeError(picture.dtype)
    picture = np.round(
        signal_min + (picture.astype(float) / native_max) * (signal_max - signal_min)
    ).astype(int)


################################################################################
# Run the encode and repeated decode
################################################################################

picture = pad_picture(state, picture)

signal_range_data = []

coeffs, encode_signal_range_log = encode(state, picture)
for (level, array_name), (minimum, maximum) in encode_signal_range_log.items():
    signal_range_data.append({
        "picture": picture_name,
        "type": "analysis",
        "level": level,
        "qi": "NA",
        "bits_per_pixel": "NA",
        "psnr": "NA",
        "array_name": array_name,
        "minimum": minimum,
        "maximum": maximum,
    })

qi = 0
while True:
    decoded_picture, num_bits, decode_signal_range_log = quantise_and_decode(state, coeffs, qi)
    
    bits_per_pixel = num_bits / float(input_width * input_height)
    np_picture = np.array(picture)
    mse = np.mean((np_picture - np.array(decoded_picture))**2)
    if mse == 0:
        psnr = np.inf
    else:
        psnr = (
            (20 * (np.log(np.max(np_picture) - np.min(np_picture))/np.log(10))) -
            (10 * (np.log(mse)/np.log(10)))
        )
    
    for (level, array_name), (minimum, maximum) in decode_signal_range_log.items():
        signal_range_data.append({
            "picture": picture_name,
            "qi": qi,
            "bits_per_pixel": bits_per_pixel,
            "psnr": psnr,
            "type": "synthesis",
            "level": level,
            "array_name": array_name,
            "minimum": minimum,
            "maximum": maximum,
        })
    
    # Stop when quantisation forces all bits to zero (and therefore num_bits ==
    # num coeffs since 0 is the only exp-golomb code of length 1)
    if num_bits == width(decoded_picture) * height(decoded_picture):
        break
    qi += 1

################################################################################
# Output CSV data
################################################################################

csv_writer = csv.DictWriter(args.output, [
    "picture",
    "qi",
    "bits_per_pixel",
    "psnr",
    "type",
    "level",
    "array_name",
    "minimum",
    "maximum",
])
csv_writer.writeheader()
for row in signal_range_data:
    csv_writer.writerow(row)
