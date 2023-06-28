#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from mdataframe.mframe import MDF2
from pandas import DataFrame, Series
from mdataframe.transformations import TMM

__author__ = "MarcoMernberger"
__copyright__ = "MarcoMernberger"
__license__ = "mit"


def test_mdf_init(test_frame):
    mdf = MDF2(test_frame)
    assert isinstance(mdf, DataFrame)


def test_mdf_mean(test_frame):
    mdf = MDF2(test_frame)
    means = mdf.mean()
    assert isinstance(mdf, DataFrame)
    assert isinstance(means, Series)
    print(test_frame)
    print(test_frame.mean().values)
    np.testing.assert_equal(means, test_frame.mean().values)


# def test_mdf_morph(test_frame):
#     mdf = MDF2(test_frame)
#     means = mdf.morph(TMM())
