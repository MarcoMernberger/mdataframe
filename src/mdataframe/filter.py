from typing import Optional, Iterable, List, Tuple, Any
from pandas import DataFrame, Index
from .transformations import _Transformer
import pandas as pd
import re


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
        self.operator_pattern = re.compile("^(?P<at_least>[\\d,a]?)(?P<operator>[<>=]+)$")
        self.operator_lookup = {
            "|>": Filter._filter_abs_gtn,
            ">": Filter._filter_gtn,
            "<": Filter._filter_ltn,
            ">=": Filter._filter_geq,
            "<=": Filter._filter_leq,
            "in": Filter._filter_isin,
            "notin": Filter._filter_notin,
        }
        self._set_filters()

    def _set_filters(self):
        self.__filters = []
        for filter_arg in self.filter_args:
            filter_callable = self.generate_filter_callable(filter_arg)
            self.__filters.append(filter_callable)

    def match_operator(self, operator: str):
        pattern_match = re.match(self.operator_pattern, operator)
        if pattern_match is None:
            raise ValueError(f"The operator '{operator}' is not valid.")
        operator = pattern_match.group("operator")
        at_least = pattern_match.group("at_least")
        return operator, at_least

    def interpret_filter_args(self, filter_arg):
        operator = filter_arg[1]
        if operator in self.operator_lookup:
            at_least = filter_arg[3] if len(filter_arg) > 3 else "1"
        else:
            operator, at_least = self.match_operator(operator)
        columns = [filter_arg[0]] if isinstance(filter_arg[0], str) else filter_arg[0]
        arguments = filter_arg[2]
        return columns, operator, arguments, at_least

    def generate_filter_callable(self, filter_arg: List[Any]):
        columns, operator, arguments, at_least = self.interpret_filter_args(filter_arg)
        filter_callable = self.get_filter_by_operator(columns, operator, arguments, at_least)
        return filter_callable

    def get_filter_by_operator(
        self, columns: List[str], operator: str, arguments: Any, at_least: str
    ):
        operate = self.operator_lookup[operator](columns, arguments)
        aggregate = self.get_aggregator(at_least)

        def __filter(df: DataFrame) -> pd.Index:
            index = aggregate(operate(df))
            return index

        return __filter

    def get_aggregator(self, at_least: str):
        def __all(df):
            return df[df.all(axis="columns")].index

        def __some(df):
            return df[df.sum(axis="columns") >= int(at_least)].index

        if at_least == "a":
            return __all
        elif at_least.isdigit():
            return __some
        else:
            raise ValueError("Don't know how to aggregate.")

    @classmethod
    def _filter_abs_gtn(self, columns: List[str], threshold: float):
        def __filter(df):
            return df[columns].abs() > threshold

        return __filter

    @classmethod
    def _filter_gtn(self, columns: List[str], threshold: float):
        def __filter(df):
            return df[columns] > threshold

        return __filter

    @classmethod
    def _filter_ltn(self, columns: List[str], threshold: float):
        def __filter(df):
            return df[columns] < threshold

        return __filter

    @classmethod
    def _filter_geq(self, columns: List[str], threshold: float):
        def __filter(df):
            return df[columns] >= threshold

        return __filter

    @classmethod
    def _filter_leq(self, columns: List[str], threshold: float):
        def __filter(df):
            return df[columns] <= threshold

        return __filter

    @classmethod
    def _filter_isin(self, column: str, allowed: List[str]):
        def __filter(df):
            return df[column].isin(allowed)

        return __filter

    @classmethod
    def _filter_notin(self, column: str, allowed: List[str]):
        def __filter(df):
            return ~df[column].isin(allowed)

        return __filter

    def __call__(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        """Call edgeR exactTest comparing two groups unpaired."""
        to_keep = df.index
        for myfilter in self.__filters:
            to_keep = to_keep.intersection(myfilter(df))
        return df.loc[to_keep]

    def __and__(self, other_filter):
        return CombinedFilter(self, other_filter, "intersection")

    def __or__(self, other_filter):
        return CombinedFilter(self, other_filter, "union")


class CombinedFilter(Filter):

    def __init__(self, filter1: Filter, filter2: Filter, combine_operation: str = "union"):
        self.filter1 = filter1
        self.filter2 = filter2
        self.combine_operation = combine_operation

    def __call__(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        index1 = self.filter1(df).index
        index2 = self.filter2(df).index
        if not hasattr(index1, self.combine_operation):
            raise ValueError(f"Combine operation {self.combine_operation} not supported for pandas.Index class.")
        combine = getattr(index1, self.combine_operation)
        index_combined = combine(index2)
        return df.loc[index_combined]


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
