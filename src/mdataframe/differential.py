import math
import rpy2.robjects as ro
import pandas as pd
from typing import Optional, Iterable, List, Dict
from collections.abc import Collection
from mbf.r import convert_dataframe_to_r, convert_dataframe_from_r
from pandas import DataFrame, Index
from .transformations import _Transformer
from abc import ABC


class Differential(_Transformer, ABC):
    def __init__(self, name, *args):
        super().__init__(name, *args)
        self.suffix = f" ({self.name})"

    @property
    def logFC(self) -> str:
        return self.logFC_column

    @property
    def P(self) -> str:
        return self.p_column

    @property
    def FDR(self) -> str:
        return self.fdr_column

    @property
    def logFC_column(self) -> str:
        return "log2FC" + self.suffix

    @property
    def p_column(self) -> str:
        return "p" + self.suffix

    @property
    def fdr_column(self) -> str:
        return "FDR" + self.suffix

    @property
    def columns(self) -> List[str]:
        return self._columns


class EdgeR_Unpaired(Differential):
    def __init__(
        self,
        columns_a: Collection,
        columns_b: Collection,
        comparison_name: Optional[str] = None,
        library_sizes: Optional[Iterable] = None,
        manual_dispersion_value: float = 0.4,
        **parameters,
    ):
        super().__init__(
            "EdgeR_Unpaired",
            columns_a,
            columns_b,
            comparison_name,
            library_sizes,
            manual_dispersion_value,
        )
        self.suffix = f" ({self.name})"
        if comparison_name is not None:
            self.suffix = f" ({comparison_name})"
        self.columns_a = columns_a
        self.columns_b = columns_b
        self.library_sizes = library_sizes
        self.manual_dispersion_value = manual_dispersion_value
        self.parameters = parameters
        if len(parameters) > 0:
            raise NotImplementedError
        self._columns = [self.logFC_column, self.p_column, self.fdr_column, self.logCPM_column]

    @property
    def logCPM(self) -> str:
        return self.logCPM_column

    @property
    def logCPM_column(self) -> str:
        return "logCPM" + self.suffix

    def __prepare_input(self, df: DataFrame) -> DataFrame:
        input_df = df[list(self.columns_a) + list(self.columns_b)]
        input_df.columns = ["X_%i" % x for x in range(len(input_df.columns))]
        return input_df

    def __prepare_sample_df(self, input_df: DataFrame) -> DataFrame:
        if self.library_sizes is not None:  # pragma: no cover
            samples = pd.DataFrame({"lib.size": self.library_sizes})
        else:
            samples = pd.DataFrame({"lib.size": input_df.sum(axis=0)})
        # this looks like it inverts the columns, but it doesnt'
        samples.insert(0, "group", ["z"] * len(self.columns_a) + ["x"] * len(self.columns_b))
        samples.index = input_df.columns
        return samples

    def __post_call(self, result: DataFrame, index: Index) -> DataFrame:
        result = result.rename(
            columns={
                "logFC": self.logFC_column,
                "PValue": self.p_column,
                "logCPM": self.logCPM_column,
                "FDR": self.fdr_column,
            }
        )
        result = result.loc[index]
        return result

    def __call__(self, df: DataFrame, *args, **kwargs) -> DataFrame:
        """Call edgeR exactTest comparing two groups unpaired."""
        ro.r("library(edgeR)")
        input_df = self.__prepare_input(df)
        samples = self.__prepare_sample_df(input_df)
        r_counts = convert_dataframe_to_r(input_df)
        r_samples = convert_dataframe_to_r(samples)
        dgelist = ro.r("DGEList")(counts=r_counts, samples=r_samples)
        # apply TMM normalization
        dgelist = ro.r("calcNormFactors")(dgelist)
        if len(self.columns_a) == 1 and len(self.columns_b) == 1:  # pragma: no cover
            # not currently used.
            dispersion = self.manual_dispersion_value
            exact_tested = ro.r("exactTest")(
                dgelist, dispersion=math.pow(self.manual_dispersion_value, 2)
            )
            print(
                """
            you are attempting to estimate dispersions without any replicates.
            Since this is not possible, there are several inferior workarounds to come up with something
            still semi-useful.
            1. pick a reasonable dispersion value from "Experience": 0.4 for humans, 0.1 for genetically identical model organisms, 0.01 for technical replicates. We'll try this for now.
            2. estimate dispersions on a number of genes that you KNOW to be not differentially expressed.
            3. In case of multiple factor experiments, discard the least important factors and treat the samples as replicates.
            4. just use logFC and forget about significance.
            """
            )
        else:
            dispersion = ro.r("estimateDisp")(dgelist, robust=True)
            exact_tested = ro.r("exactTest")(dispersion)
        res = ro.r("topTags")(exact_tested, n=len(input_df), **{"sort.by": "none"})
        return self.__post_call(convert_dataframe_from_r(res[0]), input_df.index)


class DESeq2UnpairedSimple(Differential):
    def __init__(
        self,
        columns_a: Collection,
        columns_b: Collection,
        comparison_name: Optional[str] = None,
        **parameters,
    ):
        super().__init__("DESeq2UnpairedSimple", columns_a, columns_b, comparison_name)
        if comparison_name is not None:
            self.suffix = f" ({comparison_name})"
        self.columns_a = columns_a
        self.columns_b = columns_b
        self.parameters = parameters
        if len(parameters) > 0:
            raise NotImplementedError
        self._columns = [
            self.logFC_column,
            self.p_column,
            self.fdr_column,
            self.baseMean_column,
            self.lfcSE_column,
            self.stat_column,
        ]

    @property
    def baseMean(self) -> str:
        return self.baseMean_column

    @property
    def lfcSE(self) -> str:
        return self.lfcSE_column

    @property
    def stat(self) -> str:
        return self.stat_column

    @property
    def baseMean_column(self) -> str:
        return "baseMean" + self.suffix

    @property
    def lfcSE_column(self) -> str:
        return "lfcSE" + self.suffix

    @property
    def stat_column(self) -> str:
        return "stat" + self.suffix

    def __prepare_sample_df(self) -> DataFrame:
        df_samples = pd.DataFrame(
            {
                "samples": list(self.columns_a) + list(self.columns_b),
                "condition": ["z"] * len(self.columns_a) + ["x"] * len(self.columns_b),
            }
        )
        df_samples = df_samples.set_index("samples")
        return df_samples

    def __call__(self, df_raw_counts: DataFrame, *args, **kwargs) -> DataFrame:
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
        df_samples = self.__prepare_sample_df()
        formula = "~ condition"
        r_counts = convert_dataframe_to_r(df_raw_counts)
        r_samples = convert_dataframe_to_r(df_samples)
        deseq_dataset = ro.r("DESeqDataSetFromMatrix")(
            countData=r_counts, colData=r_samples, design=ro.Formula(formula)
        )
        ro.r("print")(deseq_dataset)
        deseq_dataset = ro.r("DESeq")(deseq_dataset)
        result = ro.r("results")(deseq_dataset)
        result = ro.r("as.data.frame")(result)
        df = convert_dataframe_from_r(result)
        return self.__post_call(df, df_raw_counts.index)

    def __post_call(self, result: DataFrame, index: Index) -> DataFrame:
        if isinstance(index, pd.RangeIndex):
            result.index = result.index.astype(int)

        result = result.rename(
            columns={
                "log2FoldChange": self.logFC_column,
                "pvalue": self.p_column,
                "padj": self.fdr_column,
                "lfcSE": self.lfcSE_column,
                "baseMean": self.baseMean_column,
                "stat": self.stat_column,
            }
        )
        result = result.loc[index]
        return result


class DESeq2Unpaired(Differential):
    def __init__(
        self,
        condition_a,
        condition_b,
        condition_to_columns: Dict[str, Collection],
        comparison_name: Optional[str] = None,
        **parameters,
    ):
        super().__init__(
            "DESeq2Unpaired",
            condition_a,
            condition_b,
            condition_to_columns,
            comparison_name,
        )
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.condition_to_columns = condition_to_columns
        if comparison_name is not None:
            self.suffix = f" ({comparison_name})"
        self.parameters = parameters
        self._columns = [
            self.logFC_column,
            self.p_column,
            self.fdr_column,
            self.baseMean_column,
            self.lfcSE_column,
            self.stat_column,
        ]
        self.include_other_columns_for_variance = self.parameters.get(
            "include_other_columns_for_variance", False
        )
        self.columns_a = self.condition_to_columns[self.condition_a]
        self.columns_b = self.condition_to_columns[self.condition_b]

    @property
    def baseMean(self) -> str:
        return self.baseMean_column

    @property
    def lfcSE(self) -> str:
        return self.lfcSE_column

    @property
    def stat(self) -> str:
        return self.stat_column

    @property
    def baseMean_column(self) -> str:
        return "baseMean" + self.suffix

    @property
    def lfcSE_column(self) -> str:
        return "lfcSE" + self.suffix

    @property
    def stat_column(self) -> str:
        return "stat" + self.suffix

    def __prepare_sample_df(self) -> DataFrame:
        to_df = {
            "samples": list(self.columns_a) + list(self.columns_b),
            "condition": [f"z_{self.condition_a}"] * len(self.columns_a)
            + [f"x_{self.condition_b}"] * len(self.columns_b),
        }
        if self.include_other_columns_for_variance:
            for condition in self.condition_to_columns:
                if condition not in [self.condition_a, self.condition_b]:
                    to_df["samples"] += list(self.condition_to_columns[condition])
                    to_df["condition"] += [f"o_{condition}"] * len(
                        self.condition_to_columns[condition]
                    )
        df_samples = pd.DataFrame(to_df)
        df_samples = df_samples.set_index("samples")
        return df_samples

    def __call__(self, df_raw_counts: DataFrame, *args, **kwargs) -> DataFrame:
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
        df_samples = self.__prepare_sample_df()
        formula = "~ condition"
        r_counts = convert_dataframe_to_r(df_raw_counts)
        r_samples = convert_dataframe_to_r(df_samples)
        deseq_dataset = ro.r("DESeqDataSetFromMatrix")(
            countData=r_counts, colData=r_samples, design=ro.Formula(formula)
        )
        deseq_dataset = ro.r("DESeq")(deseq_dataset)
        result = ro.r("results")(
            deseq_dataset,
            contrast=ro.r("c")("condition", f"z_{self.condition_a}", f"x_{self.condition_b}"),
        )
        result = ro.r("as.data.frame")(result)
        df = convert_dataframe_from_r(result)
        return self.__post_call(df, df_raw_counts.index)

    def __post_call(self, result: DataFrame, index: Index) -> DataFrame:
        if isinstance(index, pd.RangeIndex):
            result.index = result.index.astype(int)

        result = result.rename(
            columns={
                "log2FoldChange": self.logFC_column,
                "pvalue": self.p_column,
                "padj": self.fdr_column,
                "lfcSE": self.lfcSE_column,
                "baseMean": self.baseMean_column,
                "stat": self.stat_column,
            }
        )
        result = result.loc[index]
        return result


class DESeq2Timeseries(Differential):
    def __init__(
        self,
        sample_columns: Collection,
        factors: Collection,
        formula: str,
        reduced: str,
        comparison_name: Optional[str] = None,
        **parameters,
    ):
        super().__init__(
            "DESeq2TimeSeries", sample_columns, factors, formula, reduced, comparison_name
        )
        if comparison_name is not None:
            self.suffix = f" ({comparison_name})"
        self.sample_columns = sample_columns
        self.factors = factors
        self.formula = formula
        self.reduced = reduced
        self.parameters = parameters
        if len(parameters) > 0:
            raise NotImplementedError
        self._columns = [
            self.logFC_column,
            self.p_column,
            self.fdr_column,
            self.baseMean_column,
            self.lfcSE_column,
            self.stat_column,
        ]

    @property
    def baseMean(self) -> str:
        return self.baseMean_column

    @property
    def lfcSE(self) -> str:
        return self.lfcSE_column

    @property
    def stat(self) -> str:
        return self.stat_column

    @property
    def baseMean_column(self) -> str:
        return "baseMean" + self.suffix

    @property
    def lfcSE_column(self) -> str:
        return "lfcSE" + self.suffix

    @property
    def stat_column(self) -> str:
        return "stat" + self.suffix

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
        df_samples = self.__prepare_sample_df()
        r_counts = convert_dataframe_to_r(df_raw_counts)
        r_samples = convert_dataframe_to_r(df_samples)
        deseq_dataset = ro.r("DESeqDataSetFromMatrix")(
            countData=r_counts, colData=r_samples, design=ro.Formula(self.formula)
        )

        # dds <- DESeq(dds, test="LRT", reduced=~batch)
        # res <- results(dds)

        deseq_dataset = ro.r("DESeq")(deseq_dataset)
        result = ro.r("results")(deseq_dataset)
        result = ro.r("as.data.frame")(result)
        df = convert_dataframe_from_r(result)
        return self.__post_call(df, df_raw_counts.index)

        #    def __prepare_sample_df(self) -> DataFrame:
        df_samples = pd.DataFrame(
            {
                "samples": list(self.sample_columns),
                "condition": ["z"] * len(self.columns_a) + ["x"] * len(self.columns_b),
            }
        )
        df_samples = df_samples.set_index("samples")
        return df_samples

    def __post_call(self, result: DataFrame, index: Index) -> DataFrame:
        result = result.rename(
            columns={
                "log2FoldChange": self.logFC_column,
                "pvalue": self.p_column,
                "padj": self.fdr_column,
                "lfcSE": self.lfcSE_column,
                "baseMean": self.baseMean_column,
                "stat": self.stat_column,
            }
        )
        result = result.loc[index]
        return result
