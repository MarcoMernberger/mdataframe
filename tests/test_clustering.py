#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import sklearn
from mdataframe.clustering import Agglo


def test_init():
    cluster = Agglo()
    assert cluster.name == "Agglo"
    print(dir(cluster.model))
    assert cluster.model.n_clusters == 2
    assert isinstance(cluster.model, sklearn.cluster.AgglomerativeClustering)
    cluster = Agglo("other", n_clusters=3)
    assert cluster.name == "other"
    assert cluster.model.n_clusters == 3


def test_cluster(test_frame):
    agglo = Agglo()
    df_clustered = agglo(test_frame)
    assert agglo.name in df_clustered.columns
    assert df_clustered[agglo.name].is_monotonic
    assert hasattr(agglo.model, "children_")


def test_sort(test_frame):
    agglo = Agglo()
    df_clustered = agglo(test_frame, sort=True, ascending=False)
    assert not df_clustered[agglo.name].is_monotonic
    df_clustered = agglo(test_frame, sort=True, ascending=True)
    assert df_clustered[agglo.name].is_monotonic
    df_rev = test_frame.loc[::-1]
    df_clustered = agglo(df_rev, sort=False)
    assert df_clustered.index.equals(test_frame.index[::-1])


def test_add(test_frame):
    test_frame.index = ["B", "C", "D", "A"]
    agglo = Agglo()
    df1_index = agglo(test_frame).index
    agglo = Agglo()
    df_clustered = agglo(test_frame, sort=True, add=False)
    assert agglo.name not in df_clustered.columns
    assert df_clustered.index.equals(df1_index)
