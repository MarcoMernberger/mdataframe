#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
from mdataframe.filter import Filter
from pandas import DataFrame


@pytest.fixture
def test_frame2():
    return pd.DataFrame(
        {
            "log": [2, 1, -9, 0],
            "fdr": [0.1, 0.001, 0.4, 0.2],
            "type": ["A", "B", "A", "C"],
        },
        index=["A", "B", "C", "D"],
    )


def test_init():
    myfilter = Filter([("log", "|>", 1)])
    assert myfilter.filter_args == [("log", "|>", 1)]
    assert myfilter.name == "Filter"


def test_filter_log_func(test_frame2):
    myfilter = Filter._filter_abs_gtn("log", 1)
    index = myfilter(test_frame2[["log"]])
    pd.testing.assert_index_equal(index, pd.Index(["A", "C"]))


def test_filter_gt(test_frame2):
    myfilter = Filter([("log", ">", 0.9)])
    df = myfilter(test_frame2)
    pd.testing.assert_index_equal(df.index, pd.Index(["A", "B"]))


def test_filter_abs_gtn(test_frame2):
    myfilter = Filter([("log", "|>", 0.9)])
    df = myfilter(test_frame2)
    pd.testing.assert_index_equal(df.index, pd.Index(["A", "B", "C"]))


def test_filter_gt_lw(test_frame2):
    myfilter = Filter([("log", ">", 0.5), ("fdr", "<", 0.2)])
    df = myfilter(test_frame2)
    pd.testing.assert_index_equal(df.index, pd.Index(["A", "B"]))


def test_filter_isin(test_frame2):
    myfilter = Filter([("type", "in", ["A"])])
    df = myfilter(test_frame2)
    pd.testing.assert_index_equal(df.index, pd.Index(["A", "C"]))


def test_filter_operator_unknown(test_frame2):
    with pytest.raises(ValueError):
        myfilter = Filter([("type", "xxx", ["A"])])
