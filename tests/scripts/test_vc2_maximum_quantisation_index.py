import pytest

import shlex

from vc2_bit_widths.scripts.vc2_static_filter_analysis import main as sfa
from vc2_bit_widths.scripts.vc2_maximum_quantisation_index import main as mqi


def test_sanity(tmpdir, capsys):
    # Just a simple sanity check that the command works as expected
    
    f = str(tmpdir.join("file.json"))
    
    # vc2-static-filter-analysis
    assert sfa(shlex.split("-w haar_with_shift -d 1 -o") + [f]) == 0
    
    # vc2-maximum-quantisation-index
    assert mqi([f, "-b", "10"]) == 0
    
    assert int(capsys.readouterr().out.strip()) == 49
