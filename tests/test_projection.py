#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest


__author__ = "MarcoMernberger"
__copyright__ = "MarcoMernberger"
__license__ = "mit"


from mdataframe.projection import PCA, TSNE, UMAP
from pandas import DataFrame


def test_pca_init():
    pca = PCA()
    assert pca.n_components == 2
    assert pca.suffix == " (PCA)"
    pca = PCA("test", 4)
    assert pca.n_components == 4
    assert pca.suffix == " (test)"


def test_pca_call(test_frame):
    pca = PCA()
    result = pca(test_frame)
    assert isinstance(result, DataFrame)
    assert hasattr(pca, "model")
    assert hasattr(pca.model, "explained_variance_ratio_")
    assert hasattr(pca.model, "explained_variance_")


def test_tsne_init():
    tsne = TSNE()
    assert tsne.n_components == 2
    assert tsne.perplexity == 30.0
    assert tsne.suffix == " (TSNE)"
    tsne = TSNE("test", 4, 2)
    assert tsne.n_components == 4
    assert tsne.suffix == " (test)"
    assert tsne.perplexity == 2


def test_tsne_call(test_frame):
    tsne = TSNE(n_components=2, perplexity=2)
    result = tsne(test_frame)
    assert isinstance(result, DataFrame)
    assert hasattr(tsne, "model")
    print(tsne.model)
    print(dir(tsne.model))
    assert hasattr(tsne.model, "embedding_")
    assert hasattr(tsne.model, "learning_rate")


def test_umap_init():
    ump = UMAP()
    assert ump.suffix == " (UMAP)"
    assert ump.n_components == 2
    ump = UMAP("test", 4)
    assert ump.suffix == " (test)"
    assert ump.n_components == 4


def test_UMAP_call(test_frame):
    ump = UMAP()
    result = ump(test_frame)
    assert isinstance(result, DataFrame)
    assert hasattr(ump, "model")
    assert hasattr(ump.model, "embedding_")
