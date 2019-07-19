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
from .plots import generate_heatmap_figure, generate_heatmap_simple_figure
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
        self.invariants.extend([missing_value, replacement_value])
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

#put me in the dependencies in transform
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
        predecessor=None,
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
        self.predecessor = predecessor
        if self.predecessor is not None:
            self.result_dir = self.predecessor.result_dir.parent / self.name
        elif hasattr(genes_or_df_or_loading_function, "result_dir"):
            self.result_dir = genes_or_df_or_loading_function.result_dir / self.name
            if "row_labels" not in kwargs:
                kwargs["row_labels"] = "name"
        else:
            self.result_dir = Path("results") / self.name
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
        self.row_f = kwargs.get("row_labels", None)
        self.col_f = kwargs.get("column_labels", None)
        if self.row_f is None and self.col_f is None and self.predecessor is not None:
            self.row_f = self.predecessor.row_f
            self.col_f = self.predecessor.col_f
        self.set_labels(self.row_f, self.col_f)

    def load(self):
        """Resolves all load dependencies."""

        def __calc():
            dictionary_with_attributes = {}
            if isinstance(
                self.df_or_gene_or_loading_function, mbf_genomics.genes.Genes
            ):
                df_full = self.df_or_gene_or_loading_function.df
                df_full.index = df_full["gene_stable_id"]
            elif isinstance(self.df_or_gene_or_loading_function, pd.DataFrame):
                df_full = self.df_or_gene_or_loading_function
            elif callable(self.df_or_gene_or_loading_function):
                dict_or_df = self.df_or_gene_or_loading_function()
                if isinstance(dict_or_df, pd.DataFrame):
                    df_full = dict_or_df
                elif isinstance(dict_or_df, dict):
                    try:
                        df_full = dict_or_df["df_full"]
                    except KeyError:
                        raise ValueError(
                            "Dictionary returned by df_or_gene_or_loading_function must contain the key df_full and the full dataframe."
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
            if "df" not in dictionary_with_attributes:
                df = df_full
                if self.columns is None:
                    self.columns = df_full.columns.values
                if self.rows is None:
                    self.rows = df_full.index.values
                df = df[self.columns]
                df = df.loc[self.rows]
                dictionary_with_attributes["df"] = df
            if "df_full" not in dictionary_with_attributes:
                dictionary_with_attributes["df_full"] = df_full
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

    def set_labels(self, row_labels=None, column_labels=None):
        self.row_f = row_labels
        self.col_f = column_labels

        def verify_label_arguments(row_labels, col_labels):
            def decorator_label_nice(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    if row_labels is not None:
                        if isinstance(row_labels, str):
                            if row_labels not in self.df_full.columns:
                                raise ValueError(
                                    f"Column given by row_labels = {row_labels} not in {self.df_full.columns}."
                                )
                    if col_labels is not None:
                        if isinstance(col_labels, str):
                            if col_labels not in self.df.index.values:
                                raise ValueError(
                                    f"Row index given by column_labels={col_labels} not in {self.df.index}."
                                )
                    return func(*args, **kwargs)

                return wrapper

            return decorator_label_nice

        def modify_df(modifier_func):
            def decorator_func(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    df = func(*args, **kwargs)
                    df = modifier_func(df)
                    return df

                return wrapper

            return decorator_func

        @verify_label_arguments(row_labels=self.row_f, col_labels=self.col_f)
        def __get_nice_df():
            return self.df.copy()

        if self.row_f is not None:
            if isinstance(self.row_f, str):

                def replace_index_column(df):
                    new_index = self.df_full[self.row_f].values
                    seen = collections.Counter()
                    for i in range(len(new_index)):
                        if new_index[i] in seen:
                            new_index[i] = f"{new_index[i]}_({seen[new_index[i]]})"
                        seen[new_index[i]] += 1
                    df.index = new_index
                    return df

                modfunc = replace_index_column
            elif isinstance(self.row_f, dict):

                def replace_index_values(df, repl_dict=self.row_f):
                    df.index = [
                        repl_dict[rname] if rname in repl_dict else rname
                        for rname in self.df.index.values
                    ]
                    return df

                modfunc = replace_index_values
            elif callable(self.row_f):
                modfunc = self.row_f
            else:
                raise ValueError(f"Don't know what to do with that row_f {self.row_f}.")
            __get_nice_df = modify_df(modfunc)(__get_nice_df)

        if self.col_f is not None:
            if isinstance(self.col_f, str):

                def replace_index(df):
                    df.columns = self.df.loc[self.col_f].values
                    return df

                modfunc = replace_index  # this is pretty useless
            elif isinstance(self.col_f, dict):

                def replace_index_values(df, repl_dict=self.col_f):
                    df.columns = [
                        repl_dict[rname] if rname in repl_dict else rname
                        for rname in df.columns.values
                    ]
                    return df

                modfunc = replace_index_values
            elif callable(self.col_f):
                modfunc = self.col_f
            else:
                raise ValueError(
                    f"Don't know what to do with that column_f {self.col_f}."
                )
            __get_nice_df = modify_df(modfunc)(__get_nice_df)
        self.__labels_nice = __get_nice_df

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
                #supply a sorting function that takes a dataframe
                sorts.append(sorting)
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
                        f"{new_name}_{sort.__name__.replace('>', '').replace('<', '')}", sort
                    )
                )
            elif isinstance(sort, list):
                deps.append(
                    ppg.ParameterInvariant(
                        f"{new_name}_PI{str(sort[1])}", sort[1:]
                    )
                )
        deps.extend(self.get_dependencies() + [self.load()])

        def __scale():
            df_scaled = self.df
            df_full = self.df_full.copy()
            for x in df_full.columns:
                print(x)
            for sorting, by, ax, ac in sorts:
                print(sorting, by, ax, ac)
                df_full = df_full.sort_values(by=by, axis=ax, ascending=ac)
                if ax == 0:
                    new_index = df_full.index
                    df_scaled = df_scaled.loc[new_index]
                else:
                    new_index = df_full.columns
                    df_scaled = df_scaled[new_index]

            return {
                "df": df_scaled,
                "rows": df_scaled.index.values,
                "columns": df_scaled.columns.values,
                "df_full": df_full,
            }

        deps.append(ppg.FunctionInvariant(new_name + "_sort", __scale))
        return ML(
            new_name,
            __scale,
            self.columns,
            self.rows,
            self.index_column,
            deps,
            self.annotators,
            self,
        )

    def transform(self, *transformations, axis=1):
        """Transforms the dataframe"""
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
                        new_name + "_{}".format(transformation_name), transformation.transform
                    )
                )                
                if hasattr(transformation, "get_invariant_parameters"):
                    deps.append(ppg.ParameterInvariant(f"{new_name}_PI_{transformation.name}", transformation.get_invariant_parameters()))
            elif isinstance(transformation, str):
                if hasattr(pd.DataFrame, transformation):
                    transforms.append((1, transformation))
                    transformation_name = transformation
                    deps.append(ppg.ParameterInvariant(f"{new_name}_PI_{transformation}", [transformation]))
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
                        ppg.ParameterInvariant(f"{new_name}_PI_{transformation[0]}", transformation)
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
                deps.append(ppg.ParameterInvariant(f"{new_name}_PI_{func}", list(transformation)))
            else:
                raise ValueError(
                    "{} did not have a name or callable transformation function named 'transform' and is not a valid method on pandas.DataFrame.".format(
                        transformation
                    )
                )
            new_name = f"{new_name}_{transformation_name}"
        deps.extend(self.get_dependencies() + [self.load()])

        def __scale():
            df_scaled = self.df
            for ttype, transformation in transforms:
                if ttype == 0:
                    df_scaled = df_scaled.apply(transformation, axis)
                if ttype == 1:
                    func = getattr(df_scaled, transformation)
                    df_scaled = func()
                if ttype == 2:
                    func = getattr(df_scaled, transformation[0])
                    df_scaled = func(*transformation[1], **transformation[2])
            df_full = self.df_full.copy()
            df_full.update(df_scaled)
            return {
                "df": df_scaled,
                "rows": df_scaled.index.values,
                "columns": df_scaled.columns.values,
                "df_full": df_full,
            }

        deps.append(ppg.FunctionInvariant(new_name + "_func", __scale))
        return ML(
            new_name,
            __scale,
            self.columns,
            self.rows,
            self.index_column,
            deps,
            self.annotators,
            self,
        )

    def impute(self, *transformations, axis=1):

        if len(transformations) == 0:
            return self.transform(
                ImputeFixed(missing_value=np.NaN, replacement_value=0)
            )

        return self.transform(*transformations, axis=1)

    def scale(self, *transformations, axis=1):
        if len(transformations) == 0:
            return self.transform(sklearn.preprocessing.scale)
        return self.transform(*transformations, axis=1)

    def get_dependencies(self):
        return self.dependencies

    def cluster(self, clustering_strategy=None, axis=1, **fit_parameter):
        """
        This defines the clustering model, fits the data and adds the trained model as attribute to self.
        """
        strategy = clustering_strategy
        if clustering_strategy is None:
            strategy = NewKMeans("KNN", 2)
        new_name = "{}_Cl({})_axis_{}".format(self.name, strategy.name, axis)
        deps = (
            self.dependencies
            + [
                self.load(),
                ppg.ParameterInvariant(new_name + "_params", [axis, fit_parameter]),
            ]
            + strategy.get_dependencies()
        )

        def __do_cluster():
            if not isinstance(strategy, ClusteringMethod):
                raise ValueError(
                    "Please supply a valid clustering strategy. Needs to be an instance of {}"
                )
            df = self.df
            if axis == 0:
                df = df.transpose()
            strategy.fit(df, **fit_parameter)
            df_full = self.df_full
            columns = self.columns
            rows = self.rows
            if axis == 0:
                df_full = df_full.append(strategy.clusters.transpose())
                df_full.loc[strategy.name] = df_full.loc[strategy.name].fillna(-1)
                df = df.transpose()
            else:
                df_full = df_full.join(strategy.clusters)
                df_full[strategy.name] = df_full[strategy.name].fillna(-1)
            return {
                strategy.name: strategy,
                "df_full": df_full,
                "df": df,
                "columns": df.columns.values,
                "rows": df.index.values,
            }

        deps.append(ppg.FunctionInvariant(new_name + "_func", __do_cluster))
        return ML(
            new_name,
            __do_cluster,
            self.columns,
            self.rows,
            self.index_column,
            deps,
            self.annotators,
            self,
        )

    def write(self, index=False, row_nice=False, column_nice=False):
        outfile = self.result_dir / (self.name + ".tsv")
        self.result_dir.mkdir(parents=True, exist_ok=True)

        def __write():
            self.df_full.to_csv(str(outfile), sep="\t", index=True)

        return ppg.FileGeneratingJob(outfile, __write).depends_on(self.load())

    def write_df(self):
        outfile = self.result_dir / (self.name + "_data.tsv")
        self.result_dir.mkdir(parents=True, exist_ok=True)

        def __write():
            self.df.to_csv(str(outfile), sep="\t", index=True)

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
            df = self.__labels_nice()
            figure = generate_heatmap_simple_figure(
                df, title, show_column_label=show_column_label, **params
            )
            figure.savefig(outfile)

        row_s = self.row_f
        col_s = self.col_f
        if callable(row_s):
            row_s = row_s.__name__
        if callable(col_s):
            col_s = col_s.__name__

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
            df = self.__labels_nice()
            figure = generate_heatmap_figure(
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

        row_s = self.row_f
        col_s = self.col_f
        if callable(row_s):
            row_s = row_s.__name__
        if callable(col_s):
            col_s = col_s.__name__

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
                row_s,
                col_s,
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
            outf = Path(self.result_dir) / f"{self.name}_hmmp.png"
        elif isinstance(outf, str):
            outf = Path(outf)
        outf.parent.mkdir(parents=True, exist_ok=True)

        def __plot():
            dpi = params.get("dpi", 300)
            rows_per_inch = params.get("rows_per_inch", 6)
            df = self.__labels_nice()
            len_y = len(df)
            max_inches_to_plot = 60000 / (2 * dpi)

            max_rows_per_page = max_inches_to_plot * rows_per_inch
            split_indices = np.arange(0, len(df), max_rows_per_page)
            pdf = PdfPages(outf)
            for index in split_indices:
                fro = df.index[index]
                to = df.index[min(index + max_rows_per_page, len_y - 1)]
                df_sub = df.loc[fro:to]
                fig = generate_heatmap_figure(
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

        row_s = self.row_f
        col_s = self.col_f
        if callable(row_s):
            row_s = row_s.__name__
        if callable(col_s):
            col_s = col_s.__name__
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
                row_s,
                col_s,
            ],
        )
        return (
            ppg.FileGeneratingJob(outf, __plot)
            .depends_on(dependencies)
            .depends_on(params_job)
        )

class ClusteringMethod:
    def __init__(self, name):
        """
        This is a wrapper for any clustering approach. 
        @param name of the clustering strategy
        """
        self.name = name

    def get_dependencies(self):
        pass

    def transform(self, df):
        df_imputed = df.transform(self.imputer.transform)
        df_imputed = df_imputed[
            df_imputed.max(axis=1) > 0
        ]  # can't have features with only zeros in the matrix
        df_scaled = df_imputed.transform(self.scaler.transform)
        return df_scaled

    def fit(self, df, **fit_parameter):
        self.model = self.clustering.fit(df, fit_parameter)
        cluster_ids = self.model.labels_
        index = df.index.values
        self.clusters = pd.DataFrame({self.name: cluster_ids}, index=index)

    def predict(self, df_other, sample_weights):
        if sample_weights == None:
            sample_weights = self.sample_weights
        df_imputed_other = df_other.transform(self.imputer.transform)
        df_imputed_other = df_imputed_other[df_imputed_other.max(axis=1) > 0]
        df_scaled_other = df_imputed_other.transform(self.scaler.transform)
        return self.model.predict(df_scaled_other.values, sample_weights)


class NewKMeans(ClusteringMethod):
    def __init__(
        self,
        name,
        no_of_clusters=2,
        n_init=10,
        max_iter=300,
        random_state=None,
        n_jobs=1,
        init="k-means++",
    ):
        """
        This is a wrapper for Kmeans
        @param name
        @param no_of_clusters number of clusters
        @param init 'k-means++', 'random' or ndarray of centroids
        @param max_iter max iterations til convergence
        """
        self.name = name
        self.no_of_clusters = no_of_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.init = init
        ClusteringMethod.__init__(self, name)
        self.clustering = sklearn.cluster.KMeans(
            n_clusters=self.no_of_clusters,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            algorithm="auto",
            tol=0.0001,
            precompute_distances="auto",
            verbose=0,
            random_state=self.random_state,
            copy_x=True,
            n_jobs=self.n_jobs,
        )

    def get_dependencies(self):
        return [
            ppg.ParameterInvariant(
                self.name + "parameters",
                [
                    self.n_init,
                    self.max_iter,
                    self.no_of_clusters,
                    self.random_state,
                    self.n_init,
                    self.n_jobs,
                ],
            ),
            ppg.FunctionInvariant(self.name + "_fit", self.fit),
        ]


class DBSCAN(ClusteringMethod):
    def __init__(
        self,
        name,
        scaler=None,
        imputer=None,
        missing_value=np.NaN,
        cluster_columns=False,
        eps=0.5,
        min_samples=5,
        metric="euclidean",
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=1,
        dependencies=[],
    ):
        """
        This is a wrapper for DBSCAN
        @param eps maximum neighborhood distance
        @param min_samples = minimum number of neighbors for a point to be a valid core point
        @param metric distance metric, allowed is string (metrics.pairwise.calculate_distance) or callable
        @param algorithm nearest neighbor algorithm, allowed is 'auto', 'ball_tree', 'kd_tree' or 'brute'
        @param leaf_size leaf_size passed to ball_tree or kd_tree
        @param p power for minkowski metric to calculate distances
        """
        self.eps = eps
        self.name = name
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        dependencies += [
            ppg.ParameterInvariant(
                self.name + "_parameters",
                [eps, p, min_samples, metric, algorithm, leaf_size, p],
            ),
            ppg.FunctionInvariant(self.name + "_fit", self.fit),
        ]
        ClusteringMethod.__init__(self, name)
        self.clustering = sklearn.cluster.DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        self.clustering.predict = self.clustering.fit_predict

    def get_dependencies(self):
        return [
            ppg.ParameterInvariant(
                self.name + "parameters",
                [
                    self.eps,
                    self.min_samples,
                    self.metric,
                    self.algorithm,
                    self.leaf_size,
                    self.p,
                    self.n_jobs,
                ],
            )
        ]


class SKlearnAgglomerative(ClusteringMethod):
    def __init__(
        self,
        name,
        no_of_clusters=2,
        affinity="euclidean",
        linkage="ward",
        connectivity=None,
        compute_full_tree="auto",
        memory=None,
    ):
        """
        This is a wrapper for Agglomerativeclustering from SKlearn
        @param genes_or_df_or_loading_function can be either genes, df or a loading function
        @param init 'k-means++', 'random' or ndarray of centroids
        @param max_iter max iterations til convergence
        @param cluster columns should we cluster columns?
        @params cluster_rows should we cluster rows?
        """
        self.name = name
        self.no_of_clusters = no_of_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.memory = memory
        ClusteringMethod.__init__(self, name)
        self.clustering = sklearn.cluster.AgglomerativeClustering(
            n_clusters=self.no_of_clusters,
            affinity=self.affinity,
            memory=self.memory,
            connectivity=self.connectivity,
            compute_full_tree=self.compute_full_tree,
            linkage=self.linkage,
        )
        self.clustering.predict = self.clustering.fit_predict

    def get_dependencies(self):
        return [
            ppg.ParameterInvariant(
                self.name + "parameters",
                [
                    self.no_of_clusters,
                    self.affinity,
                    self.linkage,
                    self.connectivity,
                    self.compute_full_tree,
                    self.memory,
                ],
            )
        ]

    def new_fit_function(self):
        """
        This is intended to replace the original  sklearn fit function of the clustering,
        slightly altered as I want the distances between the clusters as well.
        """

        def my_fit(self, X, y=None):
            """Fit the hierarchical clustering on the data
            Parameters
            ----------
            X : array-like, shape = [n_samples, n_features]
                Training data. Shape [n_samples, n_features], or [n_samples,
                n_samples] if affinity=='precomputed'.
            y : Ignored
            Returns
            -------
            self
            """
            X = sklearn.cluster.hierarchical.check_array(
                X, ensure_min_samples=2, estimator=self
            )
            memory = sklearn.cluster.hierarchical.check_memory(self.memory)

            if self.n_clusters <= 0:
                raise ValueError(
                    "n_clusters should be an integer greater than 0."
                    " %s was provided." % str(self.n_clusters)
                )

            if self.linkage == "ward" and self.affinity != "euclidean":
                raise ValueError(
                    "%s was provided as affinity. Ward can only "
                    "work with euclidean distances." % (self.affinity,)
                )

            if self.linkage not in sklearn.cluster.hierarchical._TREE_BUILDERS:
                raise ValueError(
                    "Unknown linkage type %s. "
                    "Valid options are %s"
                    % (self.linkage, sklearn.cluster.hierarchical._TREE_BUILDERS.keys())
                )
            tree_builder = sklearn.cluster.hierarchical._TREE_BUILDERS[self.linkage]

            connectivity = self.connectivity
            if self.connectivity is not None:
                if callable(self.connectivity):
                    connectivity = self.connectivity(X)
                connectivity = check_array(
                    connectivity, accept_sparse=["csr", "coo", "lil"]
                )

            n_samples = len(X)
            compute_full_tree = self.compute_full_tree
            if self.connectivity is None:
                compute_full_tree = True
            if compute_full_tree == "auto":
                # Early stopping is likely to give a speed up only for
                # a large number of clusters. The actual threshold
                # implemented here is heuristic
                compute_full_tree = self.n_clusters < max(100, 0.02 * n_samples)
            n_clusters = self.n_clusters
            if compute_full_tree:
                n_clusters = None

            # Construct the tree
            kwargs = {"return_distance": True}
            if self.linkage != "ward":
                kwargs["linkage"] = self.linkage
                kwargs["affinity"] = self.affinity
            self.children_, self.n_components_, self.n_leaves_, parents, self.distances = memory.cache(
                tree_builder
            )(
                X, connectivity, n_clusters=n_clusters, **kwargs
            )
            # Cut the tree
            if compute_full_tree:
                self.labels_ = sklearn.cluster.hierarchical._hc_cut(
                    self.n_clusters, self.children_, self.n_leaves_
                )
            else:
                labels = sklearn.cluster.hierarchical._hierarchical.hc_get_heads(
                    parents, copy=False
                )
                # copy to avoid holding a reference on the original array
                labels = np.copy(labels[:n_samples])
                # Reassign cluster numbers
                self.labels_ = np.searchsorted(np.unique(labels), labels)
            return self

        return my_fit

    def fit(self, df, **fit_parameter):
        fitter = self.new_fit_function()
        self.model = fitter(self.clustering, df)
        cluster_ids = self.model.labels_
        index = df.index.values
        self.clusters = pd.DataFrame({self.name: cluster_ids}, index=index)

    def get_linkage(self):
        number_of_observations = len(df)
        observations = dict([(x, 1) for x in range(number_of_observations)])
        current = 0
        ret = []
        for ii, item in enumerate(zip(self.model.children_, self.model.distances)):
            x = item[0]
            distance = item[1]
            o = observations[x[0]] + observations[x[1]]
            observations[number_of_observations + ii] = o
            ret.append([x[0], x[1], distance, o])
        return np.array(ret, dtype=float)


class ScipyAgglomerative(ClusteringMethod):
    def __init__(
        self, name, no_of_clusters=2, threshold=2, affinity="euclidean", linkage="ward"
    ):
        """
        This is a wrapper for hierarchical clustering from scipy. Use this if you want dendrograms.
        """
        self.no_of_clusters = no_of_clusters
        self.threshold = threshold
        self.affinity = affinity
        self.linkage = linkage
        ClusteringMethod.__init__(self, name)

    def get_dependencies(self):
        return [
            ppg.ParameterInvariant(
                self.name + "parameters",
                [self.no_of_clusters, self.threshold, self.affinity, self.linkage],
            )
        ]

    def fit(self, df, **fit_params):
        """
        Now transformation and imputation is realised in ML, we assume that the dataframe we have is already fully transformaed and scaled
        """
        print(df.as_matrix())
        raise ValueError()
        self.model = scipy.cluster.hierarchy.linkage(
            df.as_matrix(), method=self.linkage, metric=self.affinity
        )
        cluster_ids = scipy.cluster.hierarchy.fcluster(self.model, self.threshold)
        index = df.index.values
        self.clusters = pd.DataFrame({self.name: cluster_ids}, index=index)

    def predict(self, df_other, sample_weights):
        return None

    def get_linkage(self):
        return self.model
