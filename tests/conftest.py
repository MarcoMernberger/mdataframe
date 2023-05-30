#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for clustering.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

# import pytest
"""
    conftest.py for mruns.
"""

import pathlib
import sys
root = pathlib.Path(".").parent.parent
sys.path.append(str(root / "src"))
sys.path.append(str(root.parent / "mreports" / "src"))
import pandas as pd

# from mruns.util import filter_function, read_toml
# from mruns.base import analysis
# from pathlib import Path
import pytest

# from pypipegraph.testing.fixtures import (  # noqa:F401
#     new_pipegraph,
#     both_ppg_and_no_ppg,
#     no_pipegraph,
#     pytest_runtest_makereport,
# )

# data_folder = Path(__file__).parent / "data"
# toml_file = data_folder / "run.toml"


# @pytest.fixture
# def toml_input():
#     return read_toml(toml_file)


@pytest.fixture
def test_frame():
    return pd.DataFrame(
        {
            "sampleA_1": [23, 12, 9, 40],
            "sampleA_2": [2, 14, 6, 80],
            "sampleA_3": [21, 14, 6, 80],
            "sampleB_1": [23, 4, 21, 90],
            "sampleB_2": [3, 4, 22, 90],
            "sampleB_3": [4, 5, 23, 80],
        },
        index=["genA", "genB", "genC", "genD"],
    )


@pytest.fixture
def test_samples_to_group():
    return {
        "sampleA_1": "A",
        "sampleA_2": "A",
        "sampleA_3": "A",
        "sampleB_1": "B",
        "sampleB_2": "B",
        "sampleB_3": "B",
    }
