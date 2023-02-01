import math
import rpy2.robjects as ro
import pandas as pd
from typing import Optional, Iterable, List
from collections.abc import Collection
from mbf.r import convert_dataframe_to_r, convert_dataframe_from_r
from pandas import DataFrame, Index
from .transformations import _Transformer


class EdgeR_Unpaired(_Transformer):
    def __init__(
        self,
        columns_a: Collection,
        columns_b: Collection,
        comparison_name: Optional[str] = None,
        library_sizes: Optional[Iterable] = None,
        manual_dispersion_value: float = 0.4,
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
    def logCPM_column(self) -> str:
        return "logCPM" + self.suffix

    @property
    def columns(self) -> List[str]:
        return [self.logFC_column, self.p_column, self.fdr_column, self.logCPM_column]

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

    def __call__(self, df: DataFrame) -> DataFrame:
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


class DESeq2_Unpaired(_Transformer):
    def __init__(
        self,
        columns_a: Collection,
        columns_b: Collection,
        comparison_name: Optional[str] = None,
    ):
        super().__init__("DESeq2_Unpaired", columns_a, columns_b, comparison_name)
        self.suffix = f" ({self.name})"
        if comparison_name is not None:
            self.suffix = f" ({comparison_name})"
        self.columns_a = columns_a
        self.columns_b = columns_b

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
    def baseMean_column(self) -> str:
        return "baseMean" + self.suffix

    @property
    def lfcSE_column(self) -> str:
        return "lfcSE" + self.suffix

    @property
    def stat_column(self) -> str:
        return "stat" + self.suffix

    @property
    def columns(self) -> List[str]:
        return [
            self.logFC_column,
            self.p_column,
            self.fdr_column,
            self.baseMean_column,
            self.lfcSE_column,
            self.stat_column,
        ]

    def __prepare_sample_df(self) -> DataFrame:
        df_samples = pd.DataFrame(
            {
                "samples": list(self.columns_a) + list(self.columns_b),
                "condition": ["z"] * len(self.columns_a) + ["x"] * len(self.columns_b),
            }
        )
        df_samples = df_samples.set_index("samples")
        return df_samples

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
        formula = "~ condition"
        r_counts = convert_dataframe_to_r(df_raw_counts)
        r_samples = convert_dataframe_to_r(df_samples)
        ro.r("print")(r_counts)
        deseq_dataset = ro.r("DESeqDataSetFromMatrix")(
            countData=r_counts, colData=r_samples, design=ro.Formula(formula)
        )
        deseq_dataset = ro.r("DESeq")(deseq_dataset)
        ro.r("print")(deseq_dataset)
        result = ro.r("results")(deseq_dataset)
        ro.r("print")(result)
        result = ro.r("as.data.frame")(result)
        df = convert_dataframe_from_r(result)
        return self.__post_call(df, df_raw_counts.index)

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
