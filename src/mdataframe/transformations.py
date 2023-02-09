from pandas import DataFrame, Index
from typing import Optional, Union, Dict
from mbf.r import convert_dataframe_to_r, convert_dataframe_from_r
from abc import ABC
import hashlib
import pandas as pd
import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as numpy2ri

__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class _Transformer(ABC):
    """The transformer class needs to have __call__, a name, a hash function"""

    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.__set_params(args, kwargs)
        self.__calculate_hash()
        self.suffix = False

    def __set_params(self, args_from_init, kwargs_from_init):
        self._parameter_as_string = ",".join([str(x) for x in args_from_init])
        if len(kwargs_from_init) > 0:
            self._parameter_as_string += "," + ",".join(
                [f"({key}={value})" for key, value in kwargs_from_init.items()]
            )

    @property
    def hash(self):
        return self.__hash

    def __calculate_hash(self):
        m = hashlib.sha256()
        m.update(self._parameter_as_string.encode(encoding="UTF-8"))
        self.__hash = m.digest()

    def __call__(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError(
            """
            You invoked the base class __call__() method. Make sure to override
            __call__ in the child class.
            """
        )

    def _post_call(self, df: DataFrame, index: Index) -> DataFrame:
        df = df.reset_index(drop=True)
        if self.suffix:
            df.columns = [f"{col}{self.suffix}" for col in df.columns]
        df.index = index
        return df


class TMM(_Transformer):
    def __init__(
        self,
        samples_to_group: Optional[Dict[str, str]] = None,
        batch_effects: Optional[Dict[str, str]] = None,
        suffix: Union[bool, str] = False,
    ):
        super().__init__("TMM", samples_to_group, batch_effects)
        self.samples_to_group = samples_to_group
        self.batch_effects = batch_effects
        self.suffix = suffix
        if suffix is True:
            self.suffix = " (TMM)" if self.batch_effects is None else " (TMM batch-corrected)"

    def __call__(self, df_raw_counts: DataFrame) -> DataFrame:
        """
        Call to edgeR via r2py to get TMM (trimmed mean of M-values)
        normalization for raw counts.

        Prepare the edgeR input in python and call edgeR calcNormFactors via
        r2py. The TMM normalized values are returned in a DataFrame which
        is converted back to pandas DataFrame via r2py.

        Parameters
        ----------
        df_raw_counts : DataFrame
            The dataframe containing the raw counts.

        Returns
        -------
        DataFrame
            A dataframe with TMM values (trimmed mean of M-values).
        """
        if not isinstance(df_raw_counts, DataFrame):
            raise ValueError(
                f"Transformer calls need a DataFrame as first parameter, was {type(df_raw_counts)}."
            )
        ro.r("library(edgeR)")
        ro.r("library(base)")
        # create the df_samples dataframe
        to_df = {"lib.size": df_raw_counts.sum(axis=0).values}
        if self.samples_to_group is not None:
            to_df["group"] = [
                self.samples_to_group[sample_name] for sample_name in self.samples_to_group
            ]
        if self.batch_effects is not None:
            to_df["batch"] = [
                self.batch_effects[sample_name] for sample_name in df_raw_counts.columns
            ]
        columns_for_r = {col: "X" + col for col in df_raw_counts.columns}
        df_raw_counts = df_raw_counts.rename(columns=columns_for_r)
        df_samples = pd.DataFrame(to_df)
        df_samples["lib.size"] = df_samples["lib.size"].astype(int)
        df_samples = df_samples.rename(columns_for_r)
        r_counts = convert_dataframe_to_r(df_raw_counts)
        ro.r("head")(r_counts)

        r_samples = convert_dataframe_to_r(df_samples)
        dgelist = ro.r("DGEList")(
            counts=r_counts,
            samples=r_samples,
        )
        # apply TMM normalization
        dgelist = ro.r("calcNormFactors")(dgelist)  # default is TMM
        logtmm = ro.r(
            """function(dgelist){
                cpm(dgelist, log=TRUE, prior.count=5)
                }"""
        )(
            dgelist
        )  # apparently removeBatchEffects works better on log2-transformed values
        if self.batch_effects is not None:
            batches = np.array(list(self.batch_effects.values()))
            batches = numpy2ri.py2rpy(batches)
            logtmm = ro.r(
                """
                function(logtmm, batch) {
                    tmm = removeBatchEffect(logtmm,batch=batch)
                }
                """
            )(logtmm=logtmm, batch=batches)
        cpm = ro.r("data.frame")(logtmm)
        df = convert_dataframe_from_r(cpm)
        df = df.rename(columns={k: v for v, k in columns_for_r.items()})
        return self._post_call(df, df_raw_counts.index)


class VST(_Transformer):
    def __init__(
        self,
        samples_to_group: Optional[Dict[str, str]] = None,
        nsub: int = 1000,
        suffix: Union[bool, str] = False,
    ):
        super().__init__("VST", samples_to_group, nsub)
        self.samples_to_group = samples_to_group
        self.nsub = nsub
        self.suffix = suffix
        if self.suffix is True:
            self.suffix = " (VST)"

    def __call__(self, df_raw_counts: DataFrame) -> DataFrame:
        """
        Call to DESeq2 via r2py to get variance-stabilizing transformation of
        the raw counts.

        Prepare the DESeq2 input in python and call vst via
        r2py.

        Parameters
        ----------
        df_raw_counts : DataFrame
            The dataframe containing the raw counts.

        Returns
        -------
        DataFrame
            A dataframe with VST normalized counts.
        """
        if not isinstance(df_raw_counts, DataFrame):
            raise ValueError(
                f"Transformer calls need a DataFrame as first parameter, was {type(df_raw_counts)}."
            )
        ro.r("library(DESeq2)")
        columns = df_raw_counts.columns
        to_df = {}
        if self.samples_to_group is not None:
            to_df["condition"] = [
                self.samples_to_group[sample_name] for sample_name in self.samples_to_group
            ]
        formula = "~ condition"
        df_samples = pd.DataFrame(to_df)
        df_samples.index = columns
        r_counts = convert_dataframe_to_r(df_raw_counts)
        r_samples = convert_dataframe_to_r(df_samples)
        deseq_dataset = ro.r("DESeqDataSetFromMatrix")(
            countData=r_counts, colData=r_samples, design=ro.Formula(formula)
        )
        nsub = min(len(df_raw_counts), self.nsub)
        var_stabilized = ro.r(" vst")(deseq_dataset, nsub=nsub)
        var_stabilized = ro.r(
            """
            function(var_stabilized){
                vsd = as.data.frame(assay(var_stabilized));
                vsd
            }
            """
        )(var_stabilized=var_stabilized)
        df = convert_dataframe_from_r(var_stabilized)
        return self._post_call(df, df_raw_counts.index)
