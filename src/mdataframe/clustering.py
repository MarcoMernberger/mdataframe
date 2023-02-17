from typing import Optional, Iterable, List, Tuple, Any, Optional
from pandas import DataFrame, Index
from .transformations import _Transformer
from abc import abstractmethod, ABC
import pandas as pd
import sklearn


class Cluster(_Transformer):
    def __init__(self, name: str, column_name: Optional[str] = None, **kwargs):
        super().__init__(
            name,
            **kwargs,
        )
        self.new_column = self.name if column_name is None else column_name
        self._set_model(**kwargs)

    @property
    def model(self):
        return self._model

    @abstractmethod
    def _set_model(self, **kwargs):
        self._model = None

    def __sort(self, df, ascending, sort):
        if sort:
            df = df.sort_values(by=self.new_column, ascending=ascending)
        return df

    def call(self, df: DataFrame, **kwargs) -> DataFrame:
        """The dataframe should already be scaled."""
        ascending = kwargs.pop("ascending", True)
        add = kwargs.pop("add", True)
        sort = kwargs.pop("sort", True)
        labels = self._model.fit_predict(df)
        df[self.new_column] = labels
        df = self.__sort(df, ascending, sort)
        if not add:
            df = df.drop(self.new_column, axis="columns")
        return df

    def __call__(self, df: DataFrame, **kwargs) -> DataFrame:
        return self.call(df, **kwargs)


class Agglo(Cluster):
    def __init__(self, name: str = "Agglo", **kwargs):
        super().__init__(name, **kwargs)

    def _set_model(self, **kwargs):
        self._model = sklearn.cluster.AgglomerativeClustering(**kwargs)
