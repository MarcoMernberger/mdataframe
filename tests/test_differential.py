#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from mdataframe.differential import EdgeR_Unpaired, DESeq2_Unpaired
from pandas import DataFrame


def test_edger_unpaired_init(test_frame):
    columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
    columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
    lib_sizes = test_frame.sum(axis=0).values
    edger = EdgeR_Unpaired(columns_a, columns_b)
    assert edger.name == "EdgeR_Unpaired"
    assert edger.columns_a == columns_a
    assert edger.columns_b == columns_b
    assert edger.library_sizes is None
    assert edger.manual_dispersion_value == 0.4
    assert edger.suffix == f" ({edger.name})"
    edger2 = EdgeR_Unpaired(
        columns_a,
        columns_b,
        comparison_name="A_vs_B",
        library_sizes=lib_sizes,
        manual_dispersion_value=0.3,
    )
    assert edger2.suffix == " (A_vs_B)"
    np.testing.assert_equal(edger2.library_sizes, lib_sizes)
    assert edger2.manual_dispersion_value == 0.3
    assert edger.hash != edger2.hash


def test_edger_column_names():
    columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
    columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
    edger = EdgeR_Unpaired(columns_a, columns_b)
    assert edger.logFC_column == f"log2FC ({edger.name})"
    assert edger.p_column == f"p ({edger.name})"
    assert edger.fdr_column == f"FDR ({edger.name})"
    assert edger.logCPM_column == f"logCPM ({edger.name})"


def test_edger_unpaired_call(test_frame):
    columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
    columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
    lib_sizes = test_frame.sum(axis=0).values
    edger = EdgeR_Unpaired(columns_a, columns_b, library_sizes=lib_sizes)
    assert edger.suffix == " (EdgeR_Unpaired)"
    assert callable(edger)
    result = edger(test_frame)
    assert isinstance(result, DataFrame)
    r_result = DataFrame(
        {
            "genA": [0.7643766, 16.94768, 5.263604e-01, 5.263604e-01],
            "genB": [1.7107356, 16.53769, 1.429378e-04, 2.858756e-04],
            "genC": [-1.5167703, 17.10778, 6.638539e-06, 2.655416e-05],
            "genD": [-0.2533613, 19.37526, 4.473013e-01, 5.263604e-01],
        },
        index=["logFC", "logCPM", "PValue", "FDR"],
    ).transpose()
    np.testing.assert_almost_equal(
        result["log2FC" + edger.suffix].values, r_result["logFC"].values, decimal=5
    )
    np.testing.assert_almost_equal(
        result["FDR" + edger.suffix].values, r_result["FDR"].values, decimal=5
    )
    np.testing.assert_almost_equal(
        result["p (EdgeR_Unpaired)"].values, r_result["PValue"].values, decimal=5
    )
    np.testing.assert_almost_equal(
        result["logCPM (EdgeR_Unpaired)"].values, r_result["logCPM"].values, decimal=5
    )
    for col in edger.columns:
        assert col in result.columns


def test_deseq2_unpaired_init():
    columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
    columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
    deseq = DESeq2_Unpaired(columns_a, columns_b)
    assert deseq.name == "DESeq2_Unpaired"
    assert deseq.suffix == " (DESeq2_Unpaired)"
    assert deseq.columns_a == columns_a
    assert deseq.columns_b == columns_b
    deseq = DESeq2_Unpaired(columns_a, columns_b, "othername")
    assert deseq.suffix == " (othername)"


def test_deseq2_unpaired_call(test_frame):
    columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
    columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
    deseq = DESeq2_Unpaired(columns_a, columns_b)
    assert callable(deseq)
    result = deseq(test_frame)
    assert isinstance(result, DataFrame)
    print(result.head())
    r_result = DataFrame(
        {
            "genA": [10.56910, 0.651101, 0.964211, 0.675268, 0.4995057, 0.4995057],
            "genB": [9.00409, 1.574702, 0.923656, 1.704858, 0.0882209, 0.1764419],
            "genC": [15.02513, -1.746445, 0.760585, -2.296187, 0.0216652, 0.0866609],
            "genD": [78.92420, -0.409600, 0.505823, -0.809770, 0.4180723, 0.4995057],
        },
        index=["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"],
    ).transpose()
    np.testing.assert_almost_equal(
        result["log2FC" + deseq.suffix].values, r_result["log2FoldChange"].values, decimal=5
    )
    np.testing.assert_almost_equal(
        result["FDR" + deseq.suffix].values, r_result["padj"].values, decimal=5
    )
    np.testing.assert_almost_equal(
        result["p" + deseq.suffix].values, r_result["pvalue"].values, decimal=5
    )
    np.testing.assert_almost_equal(
        result["baseMean" + deseq.suffix].values, r_result["baseMean"].values, decimal=5
    )
    np.testing.assert_almost_equal(
        result["lfcSE" + deseq.suffix].values, r_result["lfcSE"].values, decimal=5
    )
    np.testing.assert_almost_equal(
        result["stat" + deseq.suffix].values, r_result["stat"].values, decimal=5
    )
    for col in deseq.columns:
        assert col in deseq.columns


def test_deseq2_column_names():
    columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
    columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
    deseq = DESeq2_Unpaired(columns_a, columns_b)
    assert deseq.logFC_column == f"log2FC ({deseq.name})"
    assert deseq.p_column == f"p ({deseq.name})"
    assert deseq.fdr_column == f"FDR ({deseq.name})"
    assert deseq.baseMean_column == "baseMean" + deseq.suffix
    assert deseq.stat_column == "stat" + deseq.suffix
    assert deseq.lfcSE_column == "lfcSE" + deseq.suffix
