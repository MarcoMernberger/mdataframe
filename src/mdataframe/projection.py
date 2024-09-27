#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""decomposition.py: Contains matrix decompositon methods."""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
import pandas as pd
import pypipegraph as ppg
import sklearn

#from umap.umap_ import UMAP as UMAPu
from pandas import DataFrame, Index
from .transformations import _Transformer
from sklearn.decomposition import PCA as PCAsk
from sklearn.manifold import TSNE as TSNEsk
from abc import abstractmethod, ABC

__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class _Reducer(_Transformer):
    def __init__(self, name: str, **kwargs):
        super().__init__(self.__class__.__name__, **kwargs)
        self.suffix = f" ({name})"
        self.model_kwargs = kwargs
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @property
    def model(self):
        return self._model

    @abstractmethod
    def _set_model(self):
        raise NotImplementedError("should not be called")

    def _new_columns(self):
        return [f"{self.name} dim {ii+1}" for ii in range(self.n_components)]

    def call(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        """The dataframe should already be scaled."""
        df_transposed = df.transpose()
        self._set_model()
        print
        matrix = self.model.fit_transform(df_transposed)
        df_result = pd.DataFrame(matrix, columns=self._new_columns(), index=df.columns)
        return df_result


class PCA(_Reducer):
    def __init__(self, name: str = "PCA", n_components: int = 2, **kwargs):
        super().__init__(name, n_components=n_components, **kwargs)
        self.n_components = n_components

    def _set_model(self):
        self._model = PCAsk(**self.model_kwargs)

    def __call__(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        return self.call(df)

    def _new_columns(self):
        return [
            f"PC{ii+1} (expl.var = {100*self.model.explained_variance_ratio_[ii]:1.1f}%)"
            for ii in range(self.n_components)
        ]


class TSNE(_Reducer):
    def __init__(
        self,
        name: str = "TSNE",
        n_components: int = 2,
        perplexity: float = 30,
        **kwargs,
    ):
        super().__init__(
            name, n_components=n_components, perplexity=perplexity, **kwargs
        )
        self.perplexity = perplexity

    def _set_model(self):
        self._model = TSNEsk(**self.model_kwargs)

    def __call__(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        return self.call(df)


class UMAP(_Reducer):
    def __init__(self, name: str = "UMAP", **kwargs):
        super().__init__(name, **kwargs)

    def _set_model(self):
        self._model = UMAPu()

    def __call__(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        return self.call(df)
