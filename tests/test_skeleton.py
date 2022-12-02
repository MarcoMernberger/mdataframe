#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


__author__ = "MarcoMernberger"
__copyright__ = "MarcoMernberger"
__license__ = "mit"


test_heatmap():

    fig = heatmap(
    df_plot: DataFrame,
    cmap: Union[str, Colormap] = "seismic",
    center: bool = True,
    shrink: float = 0.5,
    figsize: Tuple[int, int] = (10, 10),
    **kwargs)
) -> Figure:


    genes_or_df: Union[Genes, DataFrame],
    fdr_column: str,
    logFC_column: str,
    significance_threshold: float = 0.05,
    fc_threhold: float = 1,
    outfile: Path = None,
    dependencies: List[Job] = [],
    **kwargs,
)