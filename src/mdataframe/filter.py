from typing import Optional, Iterable, List, Tuple, Any
from pandas import DataFrame, Index
from .transformations import _Transformer
import pandas as pd


class Filter(_Transformer):
    def __init__(
        self,
        filter_args: List[Tuple[str, str, List[Any]]],
    ):
        super().__init__(
            "Filter",
            filter_args,
        )
        self.filter_args = filter_args
        self.operator_lookup = {
            "|>": Filter._filter_abs_gtn,
            ">": Filter._filter_gtn,
            "<": Filter._filter_lwr,
            "in": Filter._filter_isin,
        }
        self._set_filters()

    def _set_filters(self):
        self.__filters = []
        for filter_arg in self.filter_args:
            ff = self.interpret_filter(filter_arg)
            print(ff)
            self.__filters.append(ff)

    @classmethod
    def _filter_abs_gtn(self, column: str, threshold):
        def __filter(df):
            return df[df[column].abs() > threshold].index

        return __filter

    @classmethod
    def _filter_gtn(self, column: str, threshold):
        def __filter(df):
            return df[df[column] > threshold].index

        return __filter

    @classmethod
    def _filter_lwr(self, column: str, threshold):
        def __filter(df):
            return df[df[column] < threshold].index

        return __filter

    @classmethod
    def _filter_isin(self, column: str, allowed: List[str]):
        def __filter(df):
            return df[df[column].isin(allowed)].index

        return __filter

    def interpret_filter(self, filter_arg):
        column, operator, arguments = filter_arg
        if operator not in self.operator_lookup:
            raise ValueError(f"The operator '{operator}' is not valid.")
        getter = self.operator_lookup[operator]
        return getter(column, arguments)

    def __call__(self, df: DataFrame) -> DataFrame:
        """Call edgeR exactTest comparing two groups unpaired."""
        to_keep = df.index
        for myfilter in self.__filters:
            to_keep = to_keep.intersection(myfilter(df))
        return df.loc[to_keep]


"""
expected behaviour:
myfilter = Filter(
    [("log", "|>", 1)],
    [("fdr", "<", 0,05),
    [("type", "in", ["protein_coding, lincRNA"])
    ]
)
df_filter = myfilter(df)
df_filter = df.myfilter()
"""
