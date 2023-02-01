#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""mbf_compliance.py: Contains fucntions needed work with mbf."""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Any, Union
from mbf.gemomics.annotators import Annotator
import pandas as pd
import pypipegraph2 as ppg2
import hashlib


__author__ = "Marco Mernberger"
__copyright__ = "Copyright (c) 2020 Marco Mernberger"
__license__ = "mit"


class FromFile(Annotator):
    def __init__(
        self,
        tablepath: Path,
        columns_to_add: List[str],
        index_column_table: str = "gene_stable_id",
        index_column_genes: str = "gene_stable_id",
        fill_value: Optional[float] = None,
        is_tsv: bool = False,
    ):
        """
        Adds arbitrary columns from a table.

        This requires that both the table and the ddf have a common column on
        which we can index.

        Parameters
        ----------
        tablepath : Path
            Path to table with additional columns.
        columns_to_add : List[str]
            List of columns to append.
        index_column_table : str, optional
            Index column in table, by default "gene_stable_id".
        index_column_genes : str, optional
            Index column in ddf to append to, by default "gene_stable_id".
        fill_value : float, optonal
            Value to fill for missing rows, defaults to np.NaN.
        is_tsv : bool
            If the input file is a .tsv file regardless of the suffix.
        """
        self.tablepath = tablepath
        self.columns = columns_to_add
        self.index_column_table = index_column_table
        self.index_column_genes = index_column_genes
        self.fill = fill_value if fill_value is not None else np.NaN
        self.is_tsv = is_tsv

    def parse(self):
        if (
            (self.tablepath.suffix == ".xls") or (self.tablepath.suffix == ".xlsx")
        ) and not self.is_tsv:
            return pd.read_excel(self.tablepath)
        else:
            return pd.read_csv(self.tablepath, sep="\t")

    def get_cache_name(self):
        suffix = f"{self.tablepath.name}_{self.columns[0]}".encode("utf-8")
        return f"FromFile_{hashlib.md5(suffix).hexdigest()}"

    def calc_ddf(self, ddf):
        """Calculates the ddf to append."""
        df_copy = ddf.df.copy()
        if self.index_column_genes not in df_copy.columns:
            raise ValueError(
                f"Column {self.index_column_genes} not found in ddf index, found was:\n{[str(x) for x in df_copy.columns]}."
            )
        df_in = self.parse()
        if self.index_column_table not in df_in.columns:
            raise ValueError(
                f"Column {self.index_column_table} not found in table, found was:\n{[str(x) for x in df_in.columns]}."
            )
        for column in self.columns:
            if column not in df_in.columns:
                raise ValueError(
                    f"Column {column} not found in table, found was:\n{[str(x) for x in df_in.columns]}."
                )
        df_copy.index = df_copy[self.index_column_genes]
        df_in.index = df_in[self.index_column_table]
        df_in = df_in.reindex(df_copy.index, fill_value=self.fill)
        df_in = df_in[self.columns]
        df_in.index = ddf.df.index
        return df_in

    def deps(self, ddf):
        """Return ppg.jobs"""
        return ppg2.FileInvariant(self.tablepath)
