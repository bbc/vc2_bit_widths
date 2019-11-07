r"""
This script produces plots of number-of-bits-used against filter component.

The following example shows how this script can be used to illustrate the
signal ranges in a 4-level Le Gall (5, 3) transform with 10 bit pictures::

    # Step 1: Compute the bounds and test pattern signal ranges for this
    # filter.
    vc2-static-filter-analysis \
      --wavelet-index le_gall_5_3 \
      --dwt-depth 4 \
      --output analysis.json
    vc2-bit-widths-table \
      analysis.json \
      --bit-width 10 \
      --output bit_widths_table.csv
    
    # Step 2: Compute the signal ranges for some simple noise test pictures
    # (for example).
    for seed in `seq 10`; do
      python measure_picture_signal_ranges.py \
        --wavelet-index le_gall_5_3 \
        --dwt-depth 4 \
        --bit-width 10 \
        --saturated-noise 1920 1080 $seed \
        --output noise_signal_ranges_$seed.csv
    done
    
    # Step 3: Produce a plot!
    python plot_signal_ranges.py \
        --plot-upper-bound "Theoretical Upper Bound" bit_widths_table.csv \
        --plot-test-signal "Test Pattern" bit_widths_table.csv \
        --plot-picture "Noise Pictures (4:1 compressed)" \
                       2.5 \
                       noise_signal_ranges_*.csv \
        --title "4-Level Le Gall (5, 3) filter for 10-bit pictures" \
        --output plot.pdf

"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotnine import *  # ggplot and friends

# Suppress needless plotnine warnings on save
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"(Saving|Filename).*",
    module="plotnine",
)


################################################################################
# Argument parsing
################################################################################

parser = ArgumentParser()

parser.add_argument(
    "--plot-upper-bound", "-u",
    nargs="*", metavar="[TITLE] FILENAME",
    default=[], action="append",
    help="""
        Include a plot of a theoretical upper signal bound from a
        vc2-bit-widths-table CSV. Should be passed either a filename or a title
        then a filename.
    """,
)

parser.add_argument(
    "--plot-test-signal", "-t",
    nargs=2, metavar=("TITLE", "FILENAME"),
    default=[], action="append",
    help="""
        Include a plot of a test signal's range from a vc2-bit-widths-table
        CSV.
    """,
)

parser.add_argument(
    "--plot-picture-worst-case", "-w",
    nargs="*", metavar="TITLE FILENAME [FILENAME...]",
    default=[], action="append",
    help="""
        Plot the mean and range of worst-case signal values all of the pictures
        at any quantisation index in the supplied set of
        measure_picture_signal_ranges.py CSVs. The first argument should be a
        title for the line and the rest a series of CSV filenames.
    """,
)

parser.add_argument(
    "--plot-picture-absolute-worst-case", "-W",
    nargs="*", metavar="TITLE FILENAME [FILENAME...]",
    default=[], action="append",
    help="""
        Plot the worst-case signal values found for every picture and
        quantisation index in the supplied set of
        measure_picture_signal_ranges.py CSVs. The first argument should be a
        title for the line and the rest a series of CSV filenames.
    """,
)

parser.add_argument(
    "--plot-picture", "-p",
    nargs="*", metavar="TITLE BITS_PER_PIXEL FILENAME [FILENAME...]",
    default=[], action="append",
    help="""
        Plot the mean and range of signal values for a set of pictures at
        quantisation indices achieving the specified average number of bits per
        pixel. CSVs should be produced by measure_picture_signal_ranges.py.
        The first argument should be a title for the plot, followed by a
        maxmimum number of bits per pixel and then a series of CSV filenames.
    """,
)

parser.add_argument(
    "--title", "-T",
    type=str,
    help="""
        Title for the plot
    """,
)

parser.add_argument(
    "--output", "-o",
    nargs="*", metavar="FILENAME [WIDTH HEIGHT]",
    help="""
        Output filename (followed by optional width and height, in mm). If not
        specified, the plot will be displayed in a window.
    """,
)

args = parser.parse_args()

# The following routines clean up/validate/parse the various multi-argument
# options

def cast_plot_upper_bound(arg):
    if len(arg) == 1:
        return ("Theoretical Upper Bound", arg[0])
    elif len(arg) == 2:
        return arg
    else:
        parser.error("--plot-upper-bound/-u expected [TITLE] FILENAME")
    return (title, filename)
args.plot_upper_bound = list(map(cast_plot_upper_bound, args.plot_upper_bound))

def cast_plot_picture_worst_case(arg):
    if len(arg) < 2:
        parser.error("--plot-picture-worst-case/-w expected TITLE FILENAME [FILENAME...]")
    else:
        return (arg[0], arg[1:])
args.plot_picture_worst_case = list(map(
    cast_plot_picture_worst_case,
    args.plot_picture_worst_case,
))

def cast_plot_picture_absolute_worst_case(arg):
    if len(arg) < 2:
        parser.error("--plot-picture-absolute-worst-case/-W expected TITLE FILENAME [FILENAME...]")
    else:
        return (arg[0], arg[1:])
args.plot_picture_absolute_worst_case = list(map(
    cast_plot_picture_absolute_worst_case,
    args.plot_picture_absolute_worst_case,
))

def cast_plot_picture(arg):
    if len(arg) < 3:
        parser.error("--plot-picture/-p expected TITLE BITS_PER_PIXEL FILENAME [FILENAME...]")
    try:
        bits_per_pixel = float(arg[1])
    except ValueError:
        parser.error("--plot-picture/-p BITS_PER_PIXEL must be a number")
    return (arg[0], bits_per_pixel, arg[2:])
args.plot_picture = list(map(cast_plot_picture, args.plot_picture))

if args.output is not None:
    if len(args.output) == 1:
        args.output = (args.output[0], 297, 210)
    elif len(args.output) == 3:
        filename, width, height = args.output
        try:
            width = float(width)
            height = float(height)
        except ValueError:
            parser.error("--output/-o WIDTH and HEIGHT must be numbers")
        args.output = (filename, width, height)
    else:
        parser.error("--output/-o expected FILENAME [WIDTH HEIGHT]")


################################################################################
# Read and munge data
# -------------------
#
# Every plot will be read and munged into a DataFrame with the following
# columns:
#
# * title
# * type
# * level
# * array_name
# * num_bits
# * lower
# * upper
################################################################################


def minimum_and_maximum_to_num_bits(d):
    """
    Given a DataFrame with minimum and maximum columns, return a new frame with
    those replaced by a 'num_bits' column.
    """
    abs_max = d[["minimum", "maximum"]].abs().max(axis=1)
    num_bits = (np.log(abs_max.replace(0, np.nan)).fillna(0) / np.log(2)) + 1
    
    return (
        d
        .drop(columns=["minimum", "maximum"])
        .assign(num_bits=num_bits)
    )


def add_range_na(d):
    """
    Add 'lower' and 'upper' columns containing just np.nan.
    """
    return d.assign(
        lower=np.nan,
        upper=np.nan,
    )


def read_upper_bound(title, filename):
    """
    Read the upper/lower theoretical bounds from a vc2-bit-widths-CSV.
    """
    return (
        pd.read_csv(filename)
        .rename(columns={
            "lower_bound": "minimum",
            "upper_bound": "maximum",
        })
        .assign(title=title)
        [["title", "type", "level", "array_name", "minimum", "maximum"]]
        .pipe(minimum_and_maximum_to_num_bits)
        .pipe(add_range_na)
    )


def read_test_signal(title, filename):
    """
    Read the upper/lower test signal range from a vc2-bit-widths-CSV.
    """
    return (
        pd.read_csv(filename)
        .rename(columns={
            "test_signal_min": "minimum",
            "test_signal_max": "maximum",
        })
        .assign(title=title)
        [["title", "type", "level", "array_name", "minimum", "maximum"]]
        .pipe(minimum_and_maximum_to_num_bits)
        .pipe(add_range_na)
    )


def read_many_csvs(filenames):
    return pd.concat(
        map(pd.read_csv, filenames),
        ignore_index=True,
        sort=False,
    )


def read_picture_worst_case(title, filenames):
    """
    Read and aggregate the worst-case signal ranges for a set of picture files
    produced by ``measure_picture_signal_ranges.py``.
    
    For each picture and each component, the worst-case signal value for any
    quantisation index is found. These mean and range of these worst case
    values across all of the pictures is then returned.
    """
    return (
        read_many_csvs(filenames)
        .pipe(minimum_and_maximum_to_num_bits)
        # Take worst case on a picture-by-picture basis
        .groupby(["picture", "type", "level", "array_name"])
        ["num_bits"]
        .aggregate(num_bits="max")
        .reset_index()
        # Take average/range of worst cases over different pictures
        .groupby(["type", "level", "array_name"])
        ["num_bits"]
        .aggregate(
            num_bits="mean",
            lower="min",
            upper="max",
        )
        .reset_index()
        .assign(title=title)
    )


def read_picture_absoluteworst_case(title, filenames):
    """
    Read and aggregate the absolute worst-case signal ranges for a set of
    picture files produced by ``measure_picture_signal_ranges.py``.
    
    For each component, the worst-case signal value for any quantisation index
    and any picture is returned.
    """
    return (
        read_many_csvs(filenames)
        .pipe(minimum_and_maximum_to_num_bits)
        .groupby(["type", "level", "array_name"])
        ["num_bits"]
        .aggregate(num_bits="max")
        .reset_index()
        .pipe(add_range_na)
        .assign(title=title)
    )

def read_picture(title, bits_per_pixel, filenames):
    """
    Read and aggregate the signal ranges for a set of picture files quantised
    to achieve a particular number of bits per pixel produced by
    ``measure_picture_signal_ranges.py``.
    """
    d = (
        read_many_csvs(filenames)
        .pipe(minimum_and_maximum_to_num_bits)
    )
    
    # For each picture, find the lowest quantisation index which has no more
    # than the specified number of bits per pixel.
    achieved_bits_per_pixel = (
        # Get a table of (picture, qi, bits_per_pixel)
        d.query("type=='synthesis'")
        .groupby(["picture", "qi"])
        ["bits_per_pixel"]
        .aggregate(bits_per_pixel="first")
        .reset_index()
        # Select only those qis which achieve the desired number of bits per
        # pixel
        .query("bits_per_pixel <= {}".format(bits_per_pixel))
        # Choose smallest one for each picture
        .groupby("picture")
        ["qi"]
        .min()
        .reset_index()
    )
    
    analysis_results = (
        d.query("type=='analysis'")
        [["picture", "type", "level", "array_name", "num_bits"]]
    )
    
    synthesis_results = (
        # Select only the synthesis results for the quantisation indices chosen
        # earlier
        pd.merge(
            d.query("type=='synthesis'"),
            achieved_bits_per_pixel,
            on=["picture", "qi"],
        )
        [["picture", "type", "level", "array_name", "num_bits"]]
    )
    
    all_results = pd.concat(
        [analysis_results, synthesis_results],
        ignore_index=True,
        sort=False,
    )
    
    # Take average/range of numbers of bits over the different pictures
    return (
        all_results
        .groupby(["type", "level", "array_name"])
        ["num_bits"]
        .aggregate(
            num_bits="mean",
            lower="min",
            upper="max",
        )
        .reset_index()
        .assign(title=title)
    )


# An array of DataFrames (with the columns noted earlier) to be plotted which
# are later concatenated into a single data frame
data_frames_to_plot = []

# Read an munge all lines as required
for title, filename in args.plot_upper_bound:
    data_frames_to_plot.append(read_upper_bound(title, filename))

for title, filename in args.plot_test_signal:
    data_frames_to_plot.append(read_test_signal(title, filename))

for title, filenames in args.plot_picture_worst_case:
    data_frames_to_plot.append(read_picture_worst_case(title, filenames))

for title, filenames in args.plot_picture_absolute_worst_case:
    data_frames_to_plot.append(read_picture_absoluteworst_case(title, filenames))

for title, bits_per_pixel, filenames in args.plot_picture:
    data_frames_to_plot.append(read_picture(title, bits_per_pixel, filenames))

to_plot = pd.concat(
    data_frames_to_plot,
    ignore_index=True,
    sort=False,
)


################################################################################
# Perpare data for plotting
# -------------------------
#
# This consists of cleaning up catagorical value names and setting sensible
# orderings for them.
################################################################################

def make_title_categorical(d):
    """
    Make the 'title' column into a pandas Categorical type with the category
    ordering such that the plots which tend to be highest are listed first.
    Also adds a 'title_rank' column with the order indicated.
    """
    title_rank = (
        d
        # For each filter component, rank the different plots
        .set_index("title")
        .groupby(["type", "level", "array_name"])
        ["num_bits"]
        .rank().rename("title_rank")
        .reset_index()
        # Take the average rank over all of the filter components to be the rank
        # for each plot
        .groupby("title")
        .mean()
        # Break any ties arbitrarily
        .rank()
        # Sort the titles accordingly
        .reset_index()
        .sort_values("title_rank", ascending=False)
        .reset_index(drop=True)
    )
    
    # Create a suitably ordered category dtype
    cat_dtype = pd.CategoricalDtype(title_rank["title"])
    
    return pd.merge(
        d.astype({"title": cat_dtype}),
        title_rank.astype({"title": cat_dtype}),
        on="title",
    )


def add_filter_component(d):
    """
    Add a filter_component column, a Categorical value giving the combined
    type, level and array_name.
    """
    # Work out the transform depth and filter stage counts
    transform_depth = d["level"].max()
    
    array_names = d["array_name"]
    h_lifting_stages = array_names[array_names.str.startswith("DC'")].str.len().max() - 2
    v_lifting_stages = array_names[array_names.str.startswith("L'")].str.len().max() - 1
    
    # Create a series of category names which enumerate the analysis stages in
    # order
    analysis_component_names = [
        "A {} {}".format(level, array_name)
        for level in range(transform_depth, 0, -1)
        for array_name in (
            ["Input"] +
            [
                "DC" + ("'"*i)
                for i in range(h_lifting_stages + 1)
            ] +
            [
                base_name + ("'"*i)
                for i in range(v_lifting_stages + 1)
                for base_name in ("L", "H")
            ] +
            ["LL", "LH", "HL", "HH"]
        )
    ]
    
    # Construct the matching series of names for synthesis filters, in
    # reverse (symmetric) order
    synthesis_component_names = [
        (
            name
            .replace("A", "S")
            .replace("Input", "Output")
        )
        for name in reversed(analysis_component_names)
    ]
    
    component_names = analysis_component_names + synthesis_component_names
    
    filter_component = d.apply(
        lambda r: "{} {} {}".format(
            r["type"][0].upper(),
            r["level"],
            r["array_name"],
        ),
        axis=1,
    )
    
    return d.assign(filter_component=pd.Categorical(
        filter_component,
        categories=component_names,
        ordered=True,
    ))

def add_filter_component_type(d):
    """
    Adds a filter_component_type field giving a Categorical description of the
    type of component each row represents (e.g. internal/picture/bitstream
    value).
    """
    return d.assign(
        filter_component_type=pd.Categorical(
            # Classify each row
            d.apply(
                (
                    lambda r:
                    "Bitstream Coefficient"
                    if (
                        r["array_name"] in ("LH", "HL", "HH") or
                        r["array_name"] == "LL" and r["level"] == 1
                    ) else
                    "Picture"
                    if (
                        r["array_name"] in ("Input", "Output") and
                        r["level"] == d["level"].max()
                    ) else
                    "Intermediate Result"
                ),
                axis=1,
            ),
            categories=[
                "Intermediate Result",
                "Bitstream Coefficient",
                "Picture",
            ],
        )
    )


def prettify_type(d):
    """Make the 'type' column have more presentable values."""
    return d.assign(type=(
        pd.Categorical(
            d["type"],
            categories=["analysis", "synthesis"],
            ordered=True,
        )
        .rename_categories({
            "analysis": "Analysis (Encoding)",
            "synthesis": "Synthesis (Decoding)",
        })
    ))


plottable_data = (
    to_plot
    .pipe(make_title_categorical)
    .pipe(add_filter_component)
    .pipe(add_filter_component_type)
    .pipe(prettify_type)
)

plot = ggplot(
    plottable_data,
    aes(
        x="filter_component",
        y="num_bits",
        ymin="lower",
        ymax="upper",
        color="title",
        group="title",
        shape="filter_component_type",
    ),
)

# Display the data
plot += geom_line()
plot += geom_errorbar(na_rm=True)
plot += geom_point()
plot += facet_grid(". ~ type", scales="free_x")

# Set labels
plot += labs(
    x="Filter Component",
    y="Signal Range (Bits)",
    color="Pictures",
    shape="Signal Type"
)
if args.title is not None:
    plot += labs(title=args.title)

# Configure appearance
plot += theme_minimal()
plot += theme(axis_text_x=element_text(angle=-90))

def integer_breaks(lo_hi):
    return list(range(
        int(np.floor(lo_hi[0])),
        int(np.ceil(lo_hi[1])) + 1,
    ))
plot += scale_y_continuous(breaks=integer_breaks, minor_breaks=0)

# Save or display
if args.output is None:
    plot.draw()
    plt.show()
else:
    filename, width, height = args.output
    plot.save(filename, width=width, height=height, units="mm")
