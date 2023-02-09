#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from mdataframe.transformations import _Transformer, TMM, VST
from pandas import DataFrame


def test_transformer_init():
    transformer = _Transformer(
        "No Transformer", "some argument", kw_argument="some_keyword argument"
    )
    assert transformer.name == "No Transformer"
    assert transformer._parameter_as_string == "some argument,(kw_argument=some_keyword argument)"
    assert hasattr(transformer, "hash")


def test_transformer_hash():
    trans1 = _Transformer("No Transformer", "some argument", kw_argument="some_keyword argument")
    trans2 = _Transformer("No Transformer 2", "some argument", kw_argument="some_keyword argument")
    trans3 = _Transformer("No Transformer 3")
    assert trans1.hash == trans2.hash
    assert trans1.hash != trans3.hash


def test_transformer_call():
    trans = _Transformer("No Transformer")
    with pytest.raises(NotImplementedError):
        trans("mock")


def test_tmm_init(test_frame, test_samples_to_group):
    tmm = TMM()
    assert tmm.name == "TMM"
    assert tmm.samples_to_group is None
    assert tmm.batch_effects is None
    assert not tmm.suffix
    tmm = TMM(test_samples_to_group, suffix=True)
    assert tmm.samples_to_group == test_samples_to_group
    assert tmm.batch_effects is None
    assert tmm.suffix == " (TMM)"
    batches = dict(zip(test_frame.columns, ["x"] * 2 + ["y"] * 4))
    tmm = TMM(test_samples_to_group, batches, suffix=True)
    assert tmm.samples_to_group == test_samples_to_group
    assert tmm.batch_effects == batches
    assert tmm.suffix == " (TMM batch-corrected)"


def test_tmm_call(test_frame, test_samples_to_group):
    tmm = TMM(test_samples_to_group)
    assert callable(tmm)
    df = tmm(test_frame)
    assert isinstance(df, DataFrame)
    assert (df.columns == test_frame.columns).all()
    with pytest.raises(ValueError, match="Transformer calls need a DataFrame as first parameter"):
        tmm("this is not a dataframe")


def test_tmm_call_numeric_labels(test_frame, test_samples_to_group):
    replace_columns = {col: "2" + col for col in test_frame.columns}
    numeric_column_frame = test_frame.rename(columns=replace_columns)
    numeric_column_samples_groups = {
        "2" + col: test_samples_to_group[col] for col in test_samples_to_group
    }
    tmm = TMM(numeric_column_samples_groups)
    assert callable(tmm)
    df = tmm(numeric_column_frame)
    assert (numeric_column_frame.columns == df.columns).all()


def test_tmm_no_batches(test_frame, test_samples_to_group):
    tmm = TMM(test_samples_to_group, suffix=True)
    result = tmm(test_frame)
    r_result = DataFrame(
        {
            "genA": [17.83412, 15.83628, 17.77050, 17.69636, 15.97150, 16.20760],
            "genB": [17.11131, 17.31247, 17.31247, 16.11376, 16.13496, 16.36352],
            "genC": [16.82961, 16.50897, 16.50897, 17.59148, 17.68496, 17.87144],
            "genD": [18.52032, 19.48979, 19.48979, 19.44007, 19.48721, 19.48176],
        },
        index=["sampleA_1", "sampleA_2", "sampleA_3", "sampleB_1", "sampleB_2", "sampleB_3"],
    ).transpose()
    for col in r_result:
        np.testing.assert_almost_equal(
            result[col + tmm.suffix].values, r_result[col].values, decimal=5
        )


def test_tmm_batches(test_frame, test_samples_to_group):
    batches = dict(zip(test_frame.columns, ["x"] * 2 + ["y"] * 4))
    tmm = TMM(test_samples_to_group, batches, suffix=True)
    result = tmm(test_frame)
    r_result = DataFrame(
        {
            "genA": [17.87226, 15.87443, 17.73236, 17.65821, 15.93335, 16.16945],
            "genB": [16.74595, 16.94712, 17.67783, 16.47911, 16.50032, 16.72888],
            "genC": [17.20207, 16.88143, 16.13651, 17.21902, 17.31250, 17.49898],
            "genD": [18.75514, 19.72461, 19.25496, 19.20524, 19.25238, 19.24694],
        },
        index=["sampleA_1", "sampleA_2", "sampleA_3", "sampleB_1", "sampleB_2", "sampleB_3"],
    ).transpose()
    for col in r_result:
        np.testing.assert_almost_equal(
            result[col + tmm.suffix].values, r_result[col].values, decimal=5
        )


def test_VST_init(test_frame, test_samples_to_group):
    vst = VST()
    assert vst.name == "VST"
    assert vst.samples_to_group is None
    assert not vst.suffix
    vst = VST(test_samples_to_group, suffix=True)
    assert vst.samples_to_group == test_samples_to_group
    assert vst.suffix == " (VST)"


def test_vst_call(test_frame, test_samples_to_group):
    vst = VST(test_samples_to_group)
    assert callable(vst)
    df = vst(test_frame)
    assert (df.columns == test_frame.columns).all()
    assert isinstance(df, DataFrame)
    with pytest.raises(ValueError, match="Transformer calls need a DataFrame as first parameter"):
        vst("this is not a dataframe")


def test_vst_results(test_frame, test_samples_to_group):
    vst = VST(test_samples_to_group, suffix=True)
    result = vst(test_frame)
    r_result = DataFrame(
        {
            "genA": [5.130107, 3.792689, 4.813833, 4.873514, 3.929090, 4.046865],
            "genB": [4.587514, 5.037780, 4.502475, 3.795954, 4.074965, 4.170356],
            "genC": [4.384594, 4.377605, 3.990550, 4.797907, 5.371575, 5.363428],
            "genD": [5.681789, 6.966835, 6.152893, 6.268157, 6.991604, 6.775741],
        },
        index=["sampleA_1", "sampleA_2", "sampleA_3", "sampleB_1", "sampleB_2", "sampleB_3"],
    ).transpose()
    for col in r_result:
        np.testing.assert_almost_equal(
            result[col + vst.suffix].values, r_result[col].values, decimal=5
        )
