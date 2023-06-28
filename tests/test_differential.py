#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
from mdataframe.differential import EdgeR_Unpaired, DESeq2Unpaired, DESeq2UnpairedAB, NOIseq
from pandas import DataFrame


class TestSuite_EdgeR_Unpaired:

    condition_to_columns = {
        "A": ["sampleA_1", "sampleA_2", "sampleA_3"],
        "B": ["sampleB_1", "sampleB_2", "sampleB_3"],
    }

    def test_edger_unpaired_init(self, test_frame):
        columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
        columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
        lib_sizes = test_frame.sum(axis=0).values
        edger = EdgeR_Unpaired(columns_a, columns_b, self.condition_to_columns)
        assert edger.name == "EdgeR_Unpaired"
        assert edger.columns_a == columns_a
        assert edger.columns_b == columns_b
        assert edger.library_sizes is None
        assert edger.manual_dispersion_value == 0.4
        assert edger.suffix == f" ({edger.name})"
        edger2 = EdgeR_Unpaired(
            columns_a,
            columns_b,
            self.condition_to_columns,
            comparison_name="A_vs_B",
            library_sizes=lib_sizes,
            manual_dispersion_value=0.3,
        )
        assert edger2.suffix == " (A_vs_B)"
        np.testing.assert_equal(edger2.library_sizes, lib_sizes)
        assert edger2.manual_dispersion_value == 0.3
        assert edger.hash != edger2.hash

    def test_edger_column_names(self):
        columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
        columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
        edger = EdgeR_Unpaired(columns_a, columns_b, self.condition_to_columns)
        assert edger.logFC_column == f"log2FC ({edger.name})"
        assert edger.p_column == f"p ({edger.name})"
        assert edger.fdr_column == f"FDR ({edger.name})"
        assert edger.logCPM_column == f"logCPM ({edger.name})"

    def test_edger_unpaired_call(self, test_frame):
        columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
        columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
        lib_sizes = test_frame.sum(axis=0).values
        edger = EdgeR_Unpaired(columns_a, columns_b, self.condition_to_columns, library_sizes=lib_sizes)
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
        print(result.columns)
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


class TestSuite_DESeq2UnpairedAB:

    condition_to_columns = {
        "A": ["sampleA_1", "sampleA_2", "sampleA_3"],
        "B": ["sampleB_1", "sampleB_2", "sampleB_3"],
    }

    def test_DESeq2UnpairedAB_init(self):
        columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
        columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
        deseq = DESeq2UnpairedAB(columns_a, columns_b, self.condition_to_columns)
        assert deseq.name == "DESeq2UnpairedAB"
        assert deseq.suffix == " (DESeq2UnpairedAB)"
        assert deseq.columns_a == columns_a
        assert deseq.columns_b == columns_b
        deseq = DESeq2UnpairedAB(columns_a, columns_b, self.condition_to_columns, "othername")
        assert deseq.suffix == " (othername)"

    def test_DESeq2Unpaired_call(self, test_frame):
        columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
        columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
        deseq = DESeq2UnpairedAB(columns_a, columns_b, self.condition_to_columns)
        assert callable(deseq)
        result = deseq(test_frame)
        assert isinstance(result, DataFrame)
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

    def test_deseq2_column_names(self):
        columns_a = ["sampleA_1", "sampleA_2", "sampleA_3"]
        columns_b = ["sampleB_1", "sampleB_2", "sampleB_3"]
        deseq = DESeq2UnpairedAB(columns_a, columns_b, self.condition_to_columns)
        assert deseq.logFC_column == f"log2FC ({deseq.name})"
        assert deseq.p_column == f"p ({deseq.name})"
        assert deseq.fdr_column == f"FDR ({deseq.name})"
        assert deseq.baseMean_column == "baseMean" + deseq.suffix
        assert deseq.stat_column == "stat" + deseq.suffix
        assert deseq.lfcSE_column == "lfcSE" + deseq.suffix


class TestSuite_DESeq2Unpaired:
    @pytest.fixture
    def test_frame(self):
        return pd.DataFrame(
            {
                "sampleA_1": [23, 12, 9, 40],
                "sampleA_2": [2, 14, 6, 80],
                "sampleA_3": [21, 14, 6, 80],
                "sampleB_1": [23, 4, 21, 90],
                "sampleB_2": [3, 4, 22, 90],
                "sampleB_3": [4, 5, 23, 80],
                "sampleC_1": [20, 12, 12, 70],
                "sampleC_2": [4, 7, 22, 75],
                "sampleC_3": [6, 7, 13, 76],
            },
            index=["genA", "genB", "genC", "genD"],
        )

    condition_to_columns = {
        "A": ["sampleA_1", "sampleA_2", "sampleA_3"],
        "B": ["sampleB_1", "sampleB_2", "sampleB_3"],
        "C": ["sampleC_1", "sampleC_2", "sampleC_3"],
    }

    def test_DESeq2Unpaired_init(self):
        deseq = DESeq2Unpaired("A", "B", self.condition_to_columns)
        assert deseq.name == "DESeq2Unpaired"
        assert deseq.condition_to_columns == self.condition_to_columns
        assert deseq.suffix == " (DESeq2Unpaired)"
        assert deseq.columns_a == self.condition_to_columns["A"]
        assert deseq.columns_b == self.condition_to_columns["B"]
        assert not deseq.include_other_columns_for_variance
        deseq = DESeq2Unpaired(
            "A",
            "B",
            self.condition_to_columns,
            "othername",
            include_other_columns_for_variance=True,
        )
        assert deseq.suffix == " (othername)"
        assert deseq.include_other_columns_for_variance

    def test_DESeq2Unpaired_call(self, test_frame):
        deseq = DESeq2Unpaired(
            "A", "B", self.condition_to_columns, include_other_columns_for_variance=True
        )
        assert callable(deseq)
        result = deseq(test_frame)
        assert isinstance(result, DataFrame)
        r_result = DataFrame(
            {
                "genA": [10.302582, 0.6577772, 0.8259099, 0.7964272, 0.425783759, 0.42578376],
                "genB": [8.938091, 1.5981515, 0.7503795, 2.1297910, 0.033188868, 0.06637774],
                "genC": [15.465866, -1.7288386, 0.6314611, -2.7378387, 0.006184439, 0.02473776],
                "genD": [78.136713, -0.3961812, 0.4143803, -0.9560813, 0.339031148, 0.42578376],
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
            assert col in result.columns


class TestSuite_NOIseq:
    @pytest.fixture
    def test_frame(self):
        return pd.DataFrame(
            {
                "sampleA_1": [23, 12, 9, 40],
                "sampleB_1": [23, 4, 21, 90],
                "sampleC_1": [20, 12, 12, 70],
            },
            index=["genA", "genB", "genC", "genD"],
        )

    @pytest.fixture
    def df_genes(self):
        return pd.DataFrame(
            {
                "chr": ["1", "2", "17", "18"],
                "start": [1234, 2345, 3456, 4567],
                "stop": [1456, 3233, 3900, 6321],
                "biotype": ["protein_coding", "protein_coding", "lincRNA", "protein_coding"],
                "gene_stable_id": ["genA", "genB", "genC", "genD"],
            },
            index=["genA", "genB", "genC", "genD"],
        )

    condition_to_columns = {
        "A": ["sampleA_1"],
        "B": ["sampleB_1"],
        "C": ["sampleC_1"],
    }

    def test_NOIseq_init(self, df_genes):
        noiseq = NOIseq("A", "B", self.condition_to_columns, df_genes=df_genes)
        assert noiseq.name == "NOIseq"
        assert noiseq.condition_to_columns == self.condition_to_columns
        assert noiseq.suffix == " (NOIseq)"
        assert noiseq.columns_a == self.condition_to_columns["A"]
        assert noiseq.columns_b == self.condition_to_columns["B"]
        assert not noiseq.include_other_columns_for_variance
        noiseq = NOIseq(
            "A",
            "B",
            self.condition_to_columns,
            "othername",
            include_other_columns_for_variance=True,
            df_genes=df_genes
        )
        assert noiseq.suffix == " (othername)"
        assert noiseq.include_other_columns_for_variance

    def test_NOIseq_call(self, test_frame, df_genes):
        parameter = {
            "include_other_columns_for_variance": True,
            "k": 0.5,
            "norm": "tmm",
            "factor": "condition",
            "replicates": "no",
            "lc": 0,
            "pnr": 0.2,
            "nss": 5,
            "v": 0.02,
            "df_genes": df_genes
        }
        noiseq = NOIseq(
            "A", "B", self.condition_to_columns, "testcomparison", **parameter
        )
        assert callable(noiseq)
        result = noiseq(test_frame)
        assert isinstance(result, DataFrame)
        r_result = DataFrame(
            {
                "genA": [ 0.9358447, 15.65268, 0.5500,  15.680631],
                "genB": [ 2.5208073, 14.12976, 0.9125,  14.352857],
                "genC": [-0.2865477,  2.81975, 0.3125,  -2.834273],
                "genD": [-0.2340803, 10.04758, 0.2625, -10.050308],
            },
            index=["M", "D", "prob", "ranking"],
        ).transpose()
        np.testing.assert_almost_equal(
            result[noiseq.logFC].values, r_result["M"].values, decimal=5
        )
        np.testing.assert_almost_equal(
            result[noiseq.prob].values, r_result["prob"].values, decimal=5
        )
        np.testing.assert_almost_equal(
            result[noiseq.D].values, r_result["D"].values, decimal=5
        )
        np.testing.assert_almost_equal(
            result[noiseq.rank].values, r_result["ranking"].values, decimal=5
        )
        for col in noiseq.columns:
            assert col in result.columns
