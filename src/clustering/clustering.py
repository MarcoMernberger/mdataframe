"""
Created on Aug 28, 2015

@author: mernberger
"""
import matplotlib

matplotlib.use("agg")
import pypipegraph as ppg
import mbf_genomics
import copy
import os
import itertools
import numpy as np
import sklearn
import sklearn.cluster
import scipy
import scipy.cluster
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import math
import matplotlib.gridspec as grid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from . import plots
from . import strategies
import functools
import collections


def equality_of_function(func1, func2):
    eq_bytecode = func1.__code__.co_code == func1.__code__.co_code
    eq_closure = func1.__closure__ == func1.__closure__
    eq_constants = func1.__code__.co_consts == func1.__code__.co_consts
    eq_conames = func1.__code__.co_names == func1.__code__.co_names
    eq_varnames = func1.__code__.co_varnames == func1.__code__.co_varnames
    return eq_bytecode & eq_closure & eq_conames & eq_constants & eq_varnames


class ClusterAnnotator(mbf_genomics.annotator.Annotator):
    def __init__(self, clustering):
        self.clustering = clustering
        self.column_name = "clustered_by_%s" % self.clustering.name
        self.column_names = [self.column_name]
        self.column_properties = {
            self.column_names[0]: {
                "description": "The flat cluster obtained for the gene based on a given clustering."
            }
        }
        mbf_genomics.annotator.Annotator.__init__(self)

    def get_dependencies(self, genomic_regions):
        return [genomic_regions.load(), self.clustering.cluster()]

    def annotate(self, genes):
        if self.clustering.genes.name != genes.name:
            raise ValueError(
                "The genes you are trying to annotate are not the genes used in the clustering."
            )
        else:
            return pd.DataFrame(
                {
                    self.column_name: [
                        self.clustering.clusters[stable_id]
                        for stable_id in genes.df["stable_id"]
                    ]
                }
            )


class Transformer:
    def __init__(self, name):
        self.name = name
        self.invariants = [self.name]

    def transform(self):
        raise NotImplementedError()

    def get_invariant_parameters(self):
        return self.invariants


class ImputeFixed(Transformer):
    def __init__(self, missing_value=np.NaN, replacement_value=0):
        """
        replaces missing values with  with zeros
        """
        name = f"Im({missing_value}{replacement_value})"
        super().__init__(name)
        self.invariants.extend([str(missing_value), replacement_value])
        self.missing_value = missing_value
        self.replacement_value = replacement_value

    def transform(self, df):
        df = df.replace(self.missing_value, self.replacement_value)
        return df


class ImputeMeanMedian(Transformer):
    def __init__(self, missing_value=np.NaN, axis=0, strategy="mean"):
        name = f"Im({missing_value}{axis}{strategy})"
        if not strategy in ["mean", "median", "most_frequent"]:
            raise ValueError(
                "Wrong strategy, allowed is mean, median and most_frequent, was {}.".format(
                    strategy
                )
            )
        super().__init__(name)
        self.missing_value = missing_value
        self.strategy = strategy
        self.invariants.extend([strategy, missing_value, axis])
        self.imputer = sklearn.preprocessing.Imputer(
            missing_values=self.missing_value, strategy=self.strategy, axis=axis
        )

    def transform(self, matrix):
        return self.imputer.fit_transform(matrix)


class TransformScaler(Transformer):
    def __init__(self, name, transformation_function):
        super().__init__(name)
        if not hasattr(transformation_function, "__call__"):
            raise ValueError(
                "Transformation function was not a callable for {}.".format(self.name)
            )
        self.transformation_function = transformation_function

    def transform(self, data):
        return self.transformation_function(data)


# put me in the dependencies in transform
#    def get_dependencies(self):
#        return [
#            ppg.ParameterInvariant(f"{self.name}_PI", [self.name]),
#            ppg.FunctionInvariant(f"{self.name}_PI", self.transformation_function),
#        ]


class ZScaler(Transformer):
    def __init__(self, name="Z", transformation=None):
        super().__init__(name)
        self.transformation_function = transformation

    def transform(self, df):
        if df.max() == df.min():
            return pd.Series(0, index=df.index, dtype=df.dtype)
        if self.transformation_function is not None:
            df = df.transform(self.transformation_function)
        ret = ((df.transpose() - df.mean()) / df.std(ddof=1)).transpose()
        return ret


class ML(object):
    def __init__(
        self,
        name,
        genes_or_df_or_loading_function,
        columns=None,
        rows=None,
        index_column=None,
        dependencies=[],
        annotators=[],
        predecessor_objects={},
        **kwargs,
    ):
        """
        This is a wrapper class for a machine learning approach, that takes a dataframe or a
        genomic region alternatively and does some ML with it.
        @genes_or_df_or_loading_function Dataframe or GenomicRegion containing 2-dimensional data
        @dependencies dependencies
        @annotators annotators to be added to the genomics.genes.Genes object, if given.
        
        imputation, scaling, transformation and clustering is modeled as functions on ML that return a transformed ml and work on the 
        relevant part of the dataframe
        """
        self.name = name
        self.df_or_gene_or_loading_function = genes_or_df_or_loading_function
        self.columns = columns
        self.rows = rows
        self.index_column = index_column
        self.dependencies = dependencies
        self.annotators = annotators
        self.cache_dir = Path("cache") / "ML" / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.predecessor_objects = predecessor_objects
        self.result_dir = kwargs.get("result_dir", None)
        if self.result_dir is None:
            if hasattr(genes_or_df_or_loading_function, "result_dir"):
                self.result_dir = genes_or_df_or_loading_function.result_dir / self.name
                if "row_labels" not in kwargs:
                    kwargs["row_labels"] = "name"
            else:
                self.result_dir = Path("results") / "ML" / self.name
        self.result_dir.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(
            self.df_or_gene_or_loading_function, mbf_genomics.regions.GenomicRegions
        ):
            self.dependencies.append(self.df_or_gene_or_loading_function.load())
            self.annotators = annotators
            anno_names = []
            for anno in self.annotators:
                anno_names.extend(list(anno.columns))
                self.dependencies.append(
                    self.df_or_gene_or_loading_function.add_annotator(anno)
                )  #
            self.dependencies.append(
                ppg.ParameterInvariant(
                    self.name + "_anno_parameter_invariant", [anno_names]
                )
            )
        elif isinstance(self.df_or_gene_or_loading_function, pd.DataFrame):
            self.dependencies.append(
                ppg.ParameterInvariant(
                    self.name + "_df_or_gene_or_loading_function",
                    [
                        self.df_or_gene_or_loading_function.columns,
                        self.df_or_gene_or_loading_function.as_matrix(),
                        self.df_or_gene_or_loading_function.index,
                    ],
                )
            )
        self.dependencies.append(
            ppg.ParameterInvariant(self.name + "_columns", self.columns)
        )
        self.dependencies.append(ppg.ParameterInvariant(self.name + "_rows", self.rows))
        self.dependencies.append(
            ppg.ParameterInvariant(self.name + "_rowid", self.index_column)
        )
        self.row_labels = kwargs.get("row_labels", None)
        self.column_labels = kwargs.get("column_labels", None)
        self.dependencies.append(
            ppg.ParameterInvariant(
                self.name + "_kwargs",
                    list(kwargs),
                )
        )
        
    def __register_attribute():
        pass

    def register_synonym():
        pass

    def __getattr__(self, name):
        def method(*args):
            return self.transform(name, *args)
        return method

    def load(self):
        """Resolves all load dependencies."""

        def __calc():
            dictionary_with_attributes = {}
            if isinstance(
                self.df_or_gene_or_loading_function, mbf_genomics.genes.Genes
            ):
                df = self.df_or_gene_or_loading_function.df
                df.index = df["gene_stable_id"]
            elif isinstance(self.df_or_gene_or_loading_function, pd.DataFrame):
                df = self.df_or_gene_or_loading_function
            elif callable(self.df_or_gene_or_loading_function):
                dict_or_df = self.df_or_gene_or_loading_function()
                if isinstance(dict_or_df, pd.DataFrame):
                    df = dict_or_df
                elif isinstance(dict_or_df, dict):
                    try:
                        df = dict_or_df["df"]
                    except KeyError:
                        raise ValueError(
                            "Dictionary returned by df_or_gene_or_loading_function must contain the key df and the dataframe."
                        )
                    dictionary_with_attributes.update(dict_or_df)
                else:
                    raise ValueError(
                        "Return value of callable df_or_gene_or_loading_function must be a pandas.DataFrame or a dictionary, was {}.".format(
                            type(dict_or_df)
                        )
                    )
            else:
                raise ValueError(
                    "Don't know how to interpret type %s."
                    % type(self.df_or_gene_or_loading_function)
                )
            # trim the full dataframe according to the columns/rows given
            if self.columns is None:
                self.columns = df.columns.values
            if self.rows is None:
                self.rows = df.index.values
            additional_columns = [column for column in df.columns.values if column not in self.columns]
            additional_rows = [row for row in df.index.values if row not in self.rows]
            df_meta_rows = df.copy()
            df_meta_rows = df_meta_rows[additional_columns]
            if "df_meta_rows" in dictionary_with_attributes:
                df_meta_rows = df_meta_rows.join(dictionary_with_attributes["df_meta_rows"])
            df_meta_rows = df_meta_rows.loc[self.rows]
            dictionary_with_attributes["df_meta_rows"] = df_meta_rows
            df_meta_columns = df.copy().transpose()
            df_meta_columns = df_meta_columns[additional_rows]
            if "df_meta_columns" in dictionary_with_attributes:
                df_meta_columns = df_meta_columns.join(dictionary_with_attributes["df_meta_columns"])    
            df_meta_columns = df_meta_columns.loc[self.columns]
            dictionary_with_attributes["df_meta_columns"] = df_meta_columns
            df = df[self.columns]
            df = df.loc[self.rows]
            if not isinstance(df, pd.DataFrame):
                print(df)
                print(self.name)
                raise ValueError("Load function did not set a Dataframe.")
            dictionary_with_attributes["df"] = df
            assert df_meta_rows.index.equals(df.index)
            assert df_meta_columns.index.equals(df.columns)
            return dictionary_with_attributes

        def __load(dictionary_with_attributes):
            for attr_name in dictionary_with_attributes:
                setattr(self, attr_name, dictionary_with_attributes[attr_name])
        return (
            ppg.CachedDataLoadingJob(
                os.path.join(self.cache_dir, self.name + "_load"), __calc, __load
            )
            .depends_on(self.dependencies)
            .depends_on(ppg.FunctionInvariant(self.name + "_load_calc", __calc))
        )

    def sort(self, *sorting_transformations, axis=0):
        """
        We want to be able to sort by row or column --> need by, axis and ascending
        to supply a function that can be applied on the dataframe
        to apply mulitple sortings in order given --> 
        sort(column1, axis1, ascending1, column2, column3) --> this will apply the sortings separately after one another
        sort([column1, column2], [ascending1, ascending2], axis) --> this will apply the sorting simultaneously
        each string is a column/row to sort "by"
        each int is an "axis" referring to the previous "by"
        optionally bool is an "ascending" for the previous "by"
        multiple lists can be lists of columns/rows, optionally followed by int and bool lists and are applied simultanouesly, so you get a prioritized ordering 
        multiple lists can also contain [by, axis, ascending] and are applied consecutively.
        """
        new_name = self.name
        sorts = []
        by = None
        ax = axis
        ac = True
        deps = []
        for sorting in sorting_transformations:
            if callable(sorting):
                # supply a sorting function that takes a dataframe
                sorts.append([sorting, by, axis, ac])
                new_name = f"{new_name}_sort({sorting.__name__.replace('>', '').replace('<', '')})"
            elif isinstance(sorting, str):
                # row name or column name to sort by
                if by is not None:
                    sorts.append(["sort_values", by, ax, ac])
                    new_name = f"{new_name}_sort({by}_{ax}_{ac})"
                    ax = axis
                    ac = True
                by = sorting
            elif isinstance(sorting, bool):
                # then it is the ascending parameter
                ac = sorting
            elif isinstance(sorting, int):
                # then it is another axis
                ax = sorting
            elif isinstance(sorting, list) or isinstance(sorting, tuple):
                if all([isinstance(x, str) for x in sorting]):
                    if by is not None:
                        sorts.append(["sort_values", by, ax, ac])
                        new_name = f"{new_name}_sort({by}_{ax}_{ac})"
                        ax = axis
                        ac = True
                    by = sorting
                elif all([isinstance(x, bool) for x in sorting]):
                    ac = sorting
                    if len(by) != len(ac):
                        raise ValueError(
                            "If you supply a list of columns/row to sort by, you must supply the same number of ascending parameters or a single ascending value. Offending was {}.".format(
                                ac
                            )
                        )
                elif all([isinstance(x, int) for x in sorting]):
                    ax = sorting
                    if len(by) != len(ax):
                        raise ValueError(
                            "If you supply a list of columns/row to sort by, you must supply the same number of axis parameters or a single axis. Offending was {}.".format(
                                ac
                            )
                        )
                elif len(sorting) <= 3:
                    by = None
                    for parameter in sorting:
                        if isinstance(parameter, str):
                            by = parameter
                        if isinstance(parameter, bool):
                            ac = parameter
                        if isinstance(parameter, int):
                            assert parameter in [0, 1]
                            ax = parameter
                    sorts.append(["sort_values", by, ax, ac])
                    new_name = f"{new_name}_sort({by}_{ax}_{ac})"
            else:
                raise ValueError("Don't know how to sort by this: {}.".format(sorting))
        if by is not None:
            sorts.append(["sort_values", by, ax, ac])
            new_name = f"{new_name}_sort({by}_{ax}_{ac})"

        for sort in sorts:
            if callable(sort):
                deps.append(
                    ppg.FunctionInvariant(
                        f"{new_name}_{sort.__name__.replace('>', '').replace('<', '')}",
                        sort,
                    )
                )
            elif isinstance(sort, list):
                deps.append(
                    ppg.ParameterInvariant(f"{new_name}_PI{str(sort[1])}", sort[1:])
                )
        deps.extend(self.get_dependencies() + [self.load()])

        def __scale():
            df_scaled = self.df.copy()
            df_meta_columns = self.df_meta_columns.copy()
            df_meta_rows = self.df_meta_rows.copy()
            for sorting, by, ax, ac in sorts:
                if callable(sorting):
                    df_scaled = sorting(df_scaled)
                elif by in df_scaled.columns:
                    df_scaled = df_scaled.sort_values(by=by, axis=ax, ascending=ac)
                    if ax == 0:
                        df_meta_rows = df_meta_rows.loc[df_scaled.index]
                    else:
                        df_meta_columns = df_meta_columns[df_scaled.columns]
                else:
                    if ax == 0: 
                        print(by)
                        print(df_meta_rows)
                        print(df_scaled)
                        df_meta_rows = df_meta_rows.sort_values(by=by, axis=ax, ascending=ac)
                        df_scaled = df_scaled.loc[df_meta_rows.index]
                    else:
                        print(by)
                        print(df_meta_columns)
                        print(df_scaled)
                        df_meta_columns = df_meta_columns.sort_values(by=by, axis=0, ascending=ac)
                        df_scaled = df_scaled[df_meta_columns.index]

            return {
                "df": df_scaled,
                "rows": df_scaled.index.values,
                "columns": df_scaled.columns.values,
                "df_meta_columns": df_meta_columns,
                "df_meta_rows": df_meta_rows,
            }

        deps.append(ppg.FunctionInvariant(new_name + "_sort", __scale))
        return ML(
            new_name,
            __scale,
            index_column = self.index_column,
            dependencies = deps,
            annotators = self.annotators,
            predecessor_objects = self.predecessor_objects,
        )

    def __interpret_transformations_and_dependencies(self, *transformations, **kwargs):
        new_name = self.name
        deps = []
        transforms = []
        for transformation in transformations:
            if callable(transformation):
                transformation_name = transformation.__name__.replace(">", "").replace(
                    "<", ""
                )
                transforms.append((0, transformation))
                deps.append(
                    ppg.FunctionInvariant(
                        new_name + "_FI{}".format(transformation_name), transformation
                    )
                )
            elif (
                hasattr(transformation, "name")
                and hasattr(transformation, "transform")
                and callable(transformation.transform)
            ):

                transforms.append((0, transformation.transform))
                transformation_name = transformation.name
                deps.append(
                    ppg.FunctionInvariant(
                        new_name + "_{}".format(transformation_name),
                        transformation.transform,
                    )
                )
                if hasattr(transformation, "get_invariant_parameters"):
                    deps.append(
                        ppg.ParameterInvariant(
                            f"{new_name}_PI_{transformation.name}",
                            transformation.get_invariant_parameters(),
                        )
                    )
            elif isinstance(transformation, str):
                if hasattr(pd.DataFrame, transformation):
                    transforms.append((1, transformation))
                    transformation_name = transformation
                    deps.append(
                        ppg.ParameterInvariant(
                            f"{new_name}_PI_{transformation}", [transformation]
                        )
                    )
                else:
                    raise ValueError(
                        "Don't know how to apply this transformation: {}.".format(
                            transformation
                        )
                    )
            elif isinstance(transformation, list):
                if hasattr(pd.DataFrame, transformation[0]):
                    transformation_name = transformation[0]
                    positional = []
                    keyargs = {}
                    for item in transformation[1:]:
                        if isinstance(item, list):
                            positional.extend(item)
                        elif isinstance(item, dict):
                            keyargs.update(item)
                        else:
                            positional.append(item)
                    transforms.append((2, [transformation[0], positional, keyargs]))
                    deps.append(
                        ppg.ParameterInvariant(
                            f"{new_name}_PI_{transformation[0]}", transformation
                        )
                    )
                else:
                    raise ValueError(
                        "First parameter in list must be the name of a pandas function to apply. Don't know how to apply this transformation: {}.".format(
                            transformation
                        )
                    )
            elif isinstance(transformation, dict):
                if not "func" in transformation:
                    raise ValueError(
                        "If a dictionary is passed, it must contain the key 'func' with the pandas function name to be applied."
                    )
                else:
                    if not hasattr(pd.DataFrame, transformation["func"]):
                        raise ValueError(
                            "Don't know how to apply this transformation: {}.".format(
                                transformation["func"]
                            )
                        )
                transformation_name = transformation["func"]
                func = transformation["func"]
                del transformation["func"]
                transforms.append((2, [func, [], transformation]))
                deps.append(
                    ppg.ParameterInvariant(
                        f"{new_name}_PI_{func}", list(transformation)
                    )
                )
            else:
                raise ValueError(
                    "{} did not have a name or callable transformation function named 'transform' and is not a valid method on pandas.DataFrame.".format(
                        transformation
                    )
                )
            new_name = f"{new_name}_{transformation_name}"
        deps.extend(
            self.get_dependencies()
            + [self.load()]
            + [ppg.ParameterInvariant(f"{new_name}_PI_kwargs", list(kwargs))]
        )
        return new_name, transforms, deps

    def transform(self, *transformations, **kwargs):
        """Transforms the dataframe"""
        axis=kwargs.get("axis", 1)
        meta_rows = kwargs.get("meta_rows", False)
        meta_columns = kwargs.get("meta_rows", False)
        new_name = self.name
        new_name, transforms, deps = self.__interpret_transformations_and_dependencies(*transformations, **kwargs)
        def get_transformation_callable(ttype, axis, transformation):
            if ttype == 0: #callable
                if axis == 0:
                    def transformation_call(df):
                        return df.apply(transformation, axis=0)
                    return transformation_call
                elif axis == 1:
                    def transformation_call(df):
                        return df.transpose().apply(transformation, axis=0).transpose()
                else:
                    def transformation_call(df):
                        return transformation(df)
            if ttype == 1: #pandas method
                def transformation_call(df):
                    func = getattr(df, transformation)
                    return func()
            if ttype == 2: #list of pandas methods
                def transformation_call(df):
                    func = getattr(df, transformation[0])
                    return func(*transformation[1], **transformation[2])
            return transformation_call

        def __scale():
            df_scaled = self.df.copy()
            df_meta_rows = self.df_meta_rows.copy()
            df_meta_columns = self.df_meta_columns.copy()
            for ttype, transformation in transforms:
                transformation_callable = get_transformation_callable(ttype, axis, transformation)
                df_scaled = transformation_callable(df_scaled)
                #sometimes you might want to apply the transformation to the meta dfs as well .. rename comes to mind
                if meta_columns: 
                    df_meta_columns = transformation_callable(df_meta_columns)
                if meta_rows:
                    df_meta_rows = transformation_callable(df_meta_rows)
            #if the index has changed, the meta dfs loose there 1 to 1 mapping, so discard them
            if not df_scaled.index.equals(df_meta_rows.index):
                df_meta_rows = pd.DataFrame({}, index = df_scaled.index)
            if not df_scaled.columns.equals(df_meta_columns.index):
                df_meta_columns = pd.DataFrame({}, index = df_scaled.columns)
            assert isinstance(df_scaled, pd.DataFrame)
            assert isinstance(df_meta_rows, pd.DataFrame)
            assert isinstance(df_meta_columns, pd.DataFrame)
            return {
                "df": df_scaled,
                "rows": df_scaled.index.values,
                "columns": df_scaled.columns.values,
                "df_meta_rows": df_meta_rows,
                "df_meta_columns": df_meta_columns,
            }

        deps.append(ppg.FunctionInvariant(new_name + "_func", __scale))
        return ML(
            new_name,
            __scale,
            index_column = self.index_column,
            dependencies = deps,
            annotators = self.annotators,
            predecessor_objects = self.predecessor_objects,
        )

    def impute(self, *transformations, axis=1):

        if len(transformations) == 0:
            return self.transform(
                ImputeFixed(missing_value=np.NaN, replacement_value=0)
            )

        return self.transform(*transformations, axis=1)

    def scale(self, *transformations, axis=1):
        if len(transformations) == 0:
            return self.transform(sklearn.preprocessing.scale, axis=axis)
        return self.transform(*transformations, axis=axis)

    def get_dependencies(self):
        return self.dependencies

    def cluster(self, clustering_strategy=None, axis=1, **fit_parameter):
        """
        This defines the clustering model, fits the data and adds the trained model as attribute to self.
        """
        strategy = clustering_strategy
        if clustering_strategy is None:
            strategy = strategies.NewKMeans(f"KNN_{axis}", 2)
        new_name = "{}_Cl({})_axis_{}".format(self.name, strategy.name, axis)
        deps = self.dependencies + [
            self.load(),
            ppg.ParameterInvariant(
                new_name + "_params", [axis, fit_parameter] + strategy.invariants
            ),
            ppg.FunctionInvariant(new_name + "_func", strategy.fit),
        ]

        def __do_cluster():
            if not isinstance(strategy, strategies.ClusteringMethod):
                raise ValueError(
                    "Please supply a valid clustering strategy. Needs to be an instance of {}"
                )
            df = self.df.copy()
            df_meta_columns = self.df_meta_columns.copy()
            df_meta_rows = self.df_meta_rows.copy()
            if axis == 0:
                df = df.transpose()
            strategy.fit(df, **fit_parameter)
            if axis == 0:
                #cluster columns
                df_meta_columns = df_meta_columns.join(strategy.clusters)
                df_meta_columns[strategy.name] = df_meta_columns[strategy.name].fillna(-1)
                df = df.transpose()
            else:
                #cluster rows
                df_meta_rows = df_meta_rows.join(strategy.clusters)
                df_meta_rows[strategy.name] = df_meta_rows[strategy.name].fillna(-1)
            self.predecessor_objects[strategy.name] = strategy
            return {
                "df": self.df,
                "columns": df.columns.values,
                "rows": df.index.values,
                "df_meta_rows": df_meta_rows,
                "df_meta_columns": df_meta_columns,
            }

        deps.append(ppg.FunctionInvariant(new_name + "_do_Cluster", __do_cluster))
        return ML(
            new_name,
            __do_cluster,
            index_column = self.index_column,
            dependencies = deps,
            annotators = self.annotators,
            predecessor_objects = self.predecessor_objects
        )


    def reduce(self, dimensionality_reduction=None, axis=1, name = None, **fit_parameter):
        """
        This performas a dimensionality reduction on df and returns a new ML wich now contains the reduced matrix.
        """
        strategy = dimensionality_reduction
        if dimensionality_reduction is None:
            strategy = strategies.SklearnPCA("PCA")
        if name is None:
            new_name = "{}_Cl({})_axis_{}".format(self.name, strategy.name, axis)
        else:
            new_name = name
        deps = self.dependencies + [
            self.load(),
            ppg.ParameterInvariant(
                new_name + "_params", [axis, fit_parameter] + strategy.invariants
            ),
            ppg.FunctionInvariant(new_name + "_func", strategy.fit),
        ]
        new_index = [f"Dim {i+1}" for i in range(strategy.dimensions)]
        new_resdir = self.result_dir.parent / new_name
        def __do_reduce():
            if not isinstance(strategy, strategies.DimensionalityReduction):
                raise ValueError(
                    "Please supply a valid clustering strategy. Needs to be an instance of {}"
                )
            df = self.df.copy()
            df_meta_rows = self.df_meta_rows.copy()
            df_meta_columns = self.df_meta_columns.copy()
            if axis == 1:
                df = df.transpose()
            strategy.fit(df, **fit_parameter)
            columns = self.columns
            rows = self.rows
            ret = {}
            if axis == 1:
                #columns are data points ...
                new_df = pd.DataFrame(strategy.reduced_matrix.transpose(), index = new_index, columns = df.index)
                df = new_df.transpose()
                df_meta_rows = pd.DataFrame({}, index = df.index)
            else:
                #rows are data points
                df = pd.DataFrame(strategy.reduced_matrix, index = df.index, columns = new_index)
                df_meta_columns = pd.DataFrame({}, index = df.index.columns)
            rows = df.index.values
            columns = df.columns.values
            self.predecessor_objects[strategy.name] = strategy
            return {
                "df": df,
                "columns": columns,
                "rows": rows,
                "df_meta_rows": df_meta_rows,
                "df_meta_columns": df_meta_columns,
            }
    
        deps.append(ppg.FunctionInvariant(new_name + "__do_reducefunc", __do_reduce))
        return ML(
            new_name,
            __do_reduce,
            index_column = self.index_column,
            dependencies = deps,
            annotators = self.annotators,
            predecessor_objects = self.predecessor_objects,
            result_dir = new_resdir
            )
        
    
    def write(self, filename = None, index=False):
        if filename is None:
            outdir = self.result_dir
            outfile = self.result_dir / (self.name + ".tsv")
            self.result_dir.mkdir(parents=True, exist_ok=True)
        else:
            outfile = filename
            outdir = filename.parent
            outdir.mkdir(parents=True, exist_ok=True)
        def __write():
            self.df.to_csv(str(outfile), sep="\t", index=True)

        return ppg.FileGeneratingJob(outfile, __write).depends_on(self.load())

    def write_full(self):
        outfile = self.result_dir / (self.name + "_full.tsv")
        self.result_dir.mkdir(parents=True, exist_ok=True)

        def __write():
            df = self.df.transpose().join(self.df_meta_columns).transpose().join(self.df_meta_rows)
            print(df)
            raise ValueError()
            df.to_csv(str(outfile), sep="\t", index=True)

        return ppg.FileGeneratingJob(outfile, __write).depends_on(self.load())

    def plot_simple(
        self,
        outfile=None,
        show_column_label=True,
        title=None,
        dependencies=[],
        **params,
    ):
        dependencies.append(self.load())
        if outfile is None:
            outfile = Path(self.result_dir) / f"{self.name}_simple_hm.png"
        elif isinstance(outfile, str):
            outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        
        def __plot():
            df = self.df
            figure = plots.generate_heatmap_simple_figure(
                df, title, show_column_label=show_column_label, **params
            )
            figure.savefig(outfile)
            df.to_csv(str(outfile) + ".tsv", index=False, sep="\t")

        params_job = ppg.ParameterInvariant(
            outfile.name + "_label_nice", list(params) + [title, show_column_label]
        )

        return (
            ppg.FileGeneratingJob(outfile, __plot)
            .depends_on(dependencies)
            .depends_on(params_job)
        )

    def plot_singlepage(
        self,
        outfile=None,
        title=None,
        display_linkage_column=None,
        display_linkage_row=None,
        legend_location=None,
        show_column_label=True,
        show_row_label=True,
        dependencies=[],
        **params,
    ):
        dependencies.append(self.load())
        if outfile is None:
            outfile = Path(self.result_dir) / f"{self.name}_hm.png"
        elif isinstance(outfile, str):
            outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        def __plot():
            df = self.df
            figure = plots.generate_heatmap_figure(
                df,
                title,
                display_linkage_column=display_linkage_column,
                display_linkage_row=display_linkage_row,
                legend_location=legend_location,
                show_column_label=show_column_label,
                show_row_label=show_row_label,
                **params,
            )
            #    max_inches_to_plot = 60000 / (2*dpi)
            #    if inches_top+inches_bottom >= max_inches_to_plot:
            #    raise ValueError("The figure you are trying to plot is too large.")
            figure.savefig(outfile)
            df.to_csv(str(outfile) + ".tsv", index=False, sep="\t")

        params_job = ppg.ParameterInvariant(
            outfile.name + "_label_nice",
            list(params)
            + [
                title,
                display_linkage_column,
                display_linkage_row,
                legend_location,
                show_column_label,
                show_row_label,
            ],
        )

        return (
            ppg.FileGeneratingJob(outfile, __plot)
            .depends_on(dependencies)
            .depends_on(params_job)
        )

    def plot_multipage(
        self,
        outfile=None,
        title=None,
        display_linkage_column=None,
        display_linkage_row=None,
        legend_location=None,
        show_column_label=True,
        show_row_label=True,
        dependencies=[],
        **params,
    ):
        # figure out if we need to separate the plot
        dependencies.append(self.load())
        outf = outfile
        if outfile is None:
            outf = Path(self.result_dir) / f"{self.name}_hmmp.pdf"
        elif isinstance(outf, str):
            outf = Path(outf)
        outf.parent.mkdir(parents=True, exist_ok=True)

        def __plot():
            dpi = params.get("dpi", 300)
            rows_per_inch = params.get("rows_per_inch", 6)
            df = self.df
            len_y = len(df)
            max_inches_to_plot = 60000 / (2 * dpi)

            max_rows_per_page = max_inches_to_plot * rows_per_inch
            split_indices = np.arange(0, len(df), max_rows_per_page)
            pdf = PdfPages(outf)
            for index in split_indices:
                fro = df.index[index]
                to = df.index[min(index + max_rows_per_page, len_y - 1)]
                df_sub = df.loc[fro:to]
                fig = plots.generate_heatmap_figure(
                    df_sub,
                    title,
                    display_linkage_column=display_linkage_column,
                    display_linkage_row=display_linkage_row,
                    legend_location=legend_location,
                    show_column_label=show_column_label,
                    show_row_label=show_row_label,
                    **params,
                )
                pdf.savefig(fig)
            pdf.close()
            df.to_csv(str(outf) + ".tsv", index=False, sep="\t")

        params_job = ppg.ParameterInvariant(
            outf.name + "_label_nice",
            list(params)
            + [
                title,
                display_linkage_column,
                display_linkage_row,
                legend_location,
                show_column_label,
                show_row_label,
            ],
        )
        return (
            ppg.FileGeneratingJob(outf, __plot)
            .depends_on(dependencies)
            .depends_on(params_job)
        )

    def plot_2d(
        self,
        outfile=None,
        title=None,
        class_label_column = None,
        show_names = False,
        dependencies=[],
        **params,
    ):
        dependencies.append(self.load())
        if outfile is None:
            outfile = Path(self.result_dir) / f"{self.name}_2d.png"
        elif isinstance(outfile, str):
            outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        row_labels = params.get('label_dict', None)
        def __plot():
            df = self.df
            if class_label_column is not None:
                if not class_label_column in self.df_meta_rows.columns:
                    raise ValueError(f"No class label column {class_label_column} in df_meta_rows.")
                else:
                    df[class_label_column] = self.df_meta_rows[class_label_column].values
            if row_labels is not None:
                tmp = []
                for i in df.index:
                    tmp.append(row_labels[i])
                df['labels'] = tmp
            xlabel = ""
            ylabel = ""
            figure = plots.generate_dr_plot(
                df,
                title,
                class_label_column,
                xlabel = xlabel,
                ylabel = ylabel,
                show_names = show_names,
                **params                
                ) 
            figure.savefig(outfile)
            df.to_csv(str(outfile) + ".tsv", index=False, sep="\t")

        params_job = ppg.ParameterInvariant(
            outfile.name + "_2d",
            list(params)
            + [
                title,
            ],
        )
        return ppg.FileGeneratingJob(outfile, __plot).depends_on(dependencies).depends_on(params_job)
        
    
