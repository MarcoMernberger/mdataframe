'''
Created on Aug 28, 2015

@author: mernberger
'''
import matplotlib
import copy
matplotlib.use('cairo')
import pypipegraph as ppg
import genomics

import os
import itertools
import numpy
import numpy as np
import pyggplot
import sklearn
import sklearn.cluster
import scipy
import scipy.cluster
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import exptools
import pandas as pd
import pickle

"""
class AgglomerativeClusteringDistance(sklearn.cluster.AgglomerativeClustering):

    def __init__(self, n_clusters, affinity, linkage):
        sklearn.cluster.AgglomerativeClustering().__init__(n_clusters, affinity, linkage)

    def fit(self):

"""
class HeatMap():

    def __init__(self, name):
        self.name = name


def sort_clustered(linkage = 'ward', metric = 'euclidean', clusters = None):
    def sort(df):
        matrix = df.as_matrix()
        link_mat = scipy.cluster.hierarchy.linkage(matrix, method=linkage, metric = metric)
        dend = scipy.cluster.hierarchy.dendrogram(link_mat, no_plot = True)
        dendro_index = numpy.array(dend['leaves'])
        df['dendro_index'] = dendro_index
        df['row_labels'] = df.index
        df.index = range(len(df))
        if clusters is not None:
            cluster_column = [clusters[label] for label in df['row_labels'].values]
            df['clusters'] = cluster_column
            df = df.sort_values(['clusters', 'dendro_index'], ascending = [True, True])
        else:
            df = df.sort_values(['dendro_index'], True)
        index = df.index.values
        return index
    return sort

class Imputer():

    def __init__(self, name, missing_value):
        self.name = name
        self.missing_value = missing_value

    def transform(self, data):
        return data

class ImputeFixed(Imputer):

    def __init__(self, missing_value = np.NaN, replacement_value = 0):
        """
        replaces missing values with  with zeros
        """
        self.replacement_value = replacement_value
        name = 'ImputeFixed_{}_{}'.format(missing_value, replacement_value)
        Imputer.__init__(self, name, missing_value)

    def transform(self, matrix):
        if np.isnan(self.missing_value):
            index = np.isnan(matrix)
        else:
            index = matrix == self.missing_value
        matrix[index] = self.replacement_value
        return matrix

class ImputeMeanMedian(Imputer):

    def __init__(self, missing_value = np.NaN, axis = 0, strategy = 'mean'):
        name = "ImputeMeanMedian_{}_{}_{}".format(missing_value, axis, strategy)
        if not strategy in ['mean', 'median', 'most_frequent']:
            raise ValueError('Wrong strategy, allowed is mean, median and most_frequent, was {}.'.format(strategy))
        Imputer.__init__(self, name, missing_value)
        self.strategy = strategy
        self.imputer = sklearn.preprocessing.Imputer(
                                                     missing_values=self.missing_value,
                                                     strategy=self.strategy,
                                                     axis=axis
                                                     )

    def transform(self, matrix):
        return self.imputer.fit_transform(matrix)



class ClusterAnnotator(genomics.genes.annotators.Annotator):

    def __init__(self, clustering):
        self.clustering = clustering
        self.column_name = "clustered_by_%s" % self.clustering.name
        self.column_names = [self.column_name]
        self.column_properties = {
                self.column_names[0]: {
                    'description': 'The flat cluster obtained for the gene based on a given clustering.',
                }
        }
        genomics.genes.annotators.Annotator.__init__(self)


    def get_dependencies(self, genomic_regions):
        return [genomic_regions.load(), self.clustering.cluster()]

    def annotate(self, genes):
        if self.clustering.genes.name != genes.name:
            raise ValueError("The genes you are trying to annotate are not the genes used in the clustering.")
        else:
            return pd.DataFrame({self.column_name: [self.clustering.clusters[stable_id] for stable_id in genes.df['stable_id']]})

class Scaler():

    def __init__(self, name = 'Identity'):
        self.name = name

    def fit(self, df):
        pass

    def transform(self, data):
        return data

class TransformScaler(Scaler):

    def __init__(self, name, transformation_function):
        Scaler.__init__(self, name)
        if not hasattr(transformation_function, '__call__'):
            raise ValueError("Transformation function was not a callable for {}.".format(self.name))
        self.transformation_function = transformation_function

    def transform(self, data):
        return self.transformation_function(data)


class ZScaler(Scaler):

    def __init__(self, name = 'zTransform', transformation = None):
        self.transformation_function = transformation
        Scaler.__init__(self, name)

    def fit(self, matrix):
        pass

    def transform(self, df):
        if self.transformation_function is not None:
            df = df.transform(self.transformation_function)
        ret = ((df.transpose() - df.mean())/df.std(ddof = 1)).transpose()
        #matrix = df.asÄ_matrix()
        #ret = (matrix.T - np.mean(matrix, axis = 1))/np.std(matrix, axis = 1, ddof = 1)).T
        return ret

class ML(object):

    def __init__(
        self, 
        name, 
        genes_or_df_or_loading_function, 
        columns = None, 
        rows = None, 
        index_column = None, 
        dependencies = [], 
        annotators  = [], 
        predecessor = None):
        """
        This is a wrapper class for a machine learning approach, that takes a dataframe or a
        genomic region alternatively and does some ML with it.
        @genes_or_df_or_loading_function Dataframe or GenomicRegion containing 2-dimensional data
        @columns Dataframe columns that contain the actual data, may be None then all columns are used
        @columns Dataframe rows that contain the actual data, may be None, then all rows are used
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
        self.cache_dir = os.path.join('cache', 'ML', self.name)
        self.predecessor = predecessor
        if hasattr(genes_or_df_or_loading_function, 'result_dir'):
            self.result_dir = genes_or_df_or_loading_function.result_dir 
        else:
            self.result_dir = os.path.join('results', self.name)
        exptools.common.ensure_path(self.result_dir)            
        exptools.common.ensure_path(self.cache_dir)
        if isinstance(self.df_or_gene_or_loading_function, genomics.regions.GenomicRegions):
            self.dependencies.append(self.df_or_gene_or_loading_function.load())
            self.annotators = annotators
            anno_names = []
            for anno in self.annotators:
                anno_names.append(anno.column_name)
                self.dependencies.append(self.df_or_gene_or_loading_function.add_annotator(anno))   #
            self.dependencies.append(ppg.ParameterInvariant(self.name+'_anno_parameter_invariant', [anno_names]))
        self.dependencies.append(ppg.ParameterInvariant(self.name+'_columns', self.columns))
        self.dependencies.append(ppg.ParameterInvariant(self.name+'_rows', self.rows))
        self.dependencies.append(ppg.ParameterInvariant(self.name+'_rowid', self.index_column))

    def get_dependencies(self):
        return self.dependencies

    def write(self):
        outfile = os.path.join(self.result_dir, self.name + '.tsv')
        def __write():
            self.df_full.to_csv(outfile, sep = '\t', index = False)
        return ppg.FileGeneratingJob(outfile, __write).depends_on(self.load())
        
    def load(self):
        """Resolves all load dependencies."""
        def __calc():
            dictionary_with_attributes = {}
            if isinstance(self.df_or_gene_or_loading_function, genomics.regions.GenomicRegions):
                df_full = self.df_or_gene_or_loading_function.df
                df_full.index = df_full['stable_id']
            elif isinstance(self.df_or_gene_or_loading_function, pd.DataFrame):
                df_full = self.df_or_gene_or_loading_function
            elif callable(self.df_or_gene_or_loading_function):
                dict_or_df = self.df_or_gene_or_loading_function()
                if isinstance(dict_or_df, pd.DataFrame):
                    df_full = dict_or_df
                elif isinstance(dict_or_df, dict):
                    try:
                        df_full = dict_or_df['df_full']
                    except KeyError:
                        raise ValueError("Dictionary returned by df_or_gene_or_loading_function must contain the key df_full and the full dataframe.")
                    dictionary_with_attributes.update(dict_or_df)
                else:
                    raise ValueError("Return value of callable df_or_gene_or_loading_function must be a pandas.DataFrame or a dictionary, was {}.".format(type(dict_or_df)))
            else:
                raise ValueError("Don\'t know how to interpret type %s." % type(self.df_or_gene_or_loading_function))
            #trim the full dataframe according to the columns/rows given
            df = df_full
            if self.columns is not None:
                df = df[self.columns]
            if self.rows is not None:
                df = df.loc[self.rows]
            dictionary_with_attributes['df'] = df
            dictionary_with_attributes['df_full'] = df_full
            return dictionary_with_attributes
        def __load(dictionary_with_attributes):
            for attr_name in dictionary_with_attributes:
                setattr(self, attr_name, dictionary_with_attributes[attr_name])
        return ppg.CachedDataLoadingJob(os.path.join(self.cache_dir, self.name+'_load'), __calc, __load).depends_on(self.dependencies)

    def impute(self, imputer = None):
        """"""
        deps = self.get_dependencies() + [self.load()]
        imputer = imputer
        if imputer is None:
            imputer = Imputer('default', missing_value = 0)
        if not (hasattr(imputer, 'name') and hasattr(imputer, 'transform') and callable(imputer.transform)):
            raise ValueError("Imputer must have a callable transform and name attribute.")
        new_name = "{}_{}".format(self.name, imputer.name)
        def __impute():
            df_imputed = self.df.apply(imputer.transform)
        return ML(new_name, __impute, self.columns, self.rows, self.index_column, deps, self.annotators, self)
        
    def transform(self, scaler, axis = 1):
        new_name = "{}_{}".format(self.name, scaler.name)
        deps = self.get_dependencies() + [self.load()]
        def __scale():
            df_scaled = self.df.apply(scaler.transform, axis)
            return df_scaled
        return ML(new_name, __scale, self.columns, self.rows, self.index_column, deps, self.annotators, self)

    def scale(self, scaler, axis = 1):
        """the same as transform"""
        new_name = "{}_{}".format(self.name, scaler.name)
        deps = self.get_dependencies() + [self.load()]
        def __scale():
            df_scaled = self.df.apply(scaler.transform, axis)
            return df_scaled
        return ML(new_name, __scale, self.columns, self.rows, self.index_column, deps, self.annotators, self)

    def cluster(self, name = 'default', clustering_strategy = None, cluster_columns = True, sample_weights = None):        
        """
        This defines the clustering model, fits the data and adds the trained model as attribute to self.
        """
        strategy = clustering_strategy
        if clustering_strategy is None:
            strategy = NewKMeans(name, 2)
        deps = self.dependencies + [self.load()]
        new_name = "{}_Cl({})".format(self.name, strategy.name)
        def __do_cluster():
            if not isinstance(strategy, ClusteringMethod):
                raise ValueError("Please supply a valid clustering strategy. Needs to be an instance of {}")
            strategy.fit(self.df, sample_weights)
            df_full = self.df_full
            df_full = df_full.join(strategy.clusters)
            return {strategy.name : strategy, 'df_full' : df_full}
        return ML(new_name, __do_cluster, self.columns, self.rows, self.index_column, deps, self.annotators, self)

    
    def __heatmap(self, outpath, name, df, column_name_mangler = lambda x : x, colmap = 'seismic',
                      scaler_or_index = None, vmin = None, vmax = None, sort_function_row = None,
                      sort_function_column = None, shape = (50, 50), size_x=20, size_y=10, dendro = False,
                      cluster_ascending = True, dependencies = [], df_plot_rows_or_callable = None,
                      df_plot_columns_or_callable = None, show_column_labels = True, show_row_labels = True, 
                      transformation = None, row_name_mangler = None):
        #sort functions should take a df and return an index to sort by
        """
        @df_plot_rows_or_callable contains additional rows to be plotted and must be in the same order as the input df
        @df_plot_columns_or_callable contains additional columns to be plotted and must be in the same order as the input df
        @sort_function_row sorting function, takes a dataframe with index corresponding to rows and returns numerical sort index
        @sort_function_column sorting function, takes a dataframe with index corresponding to columns and returns numerical sort index
        """
        exptools.common.ensure_path(outpath)
        outfile = os.path.join(outpath, name)
        dump_file = os.path.join(self.cache_dir, name+'_pickled')
        deps = dependencies
        if self.cluster_rows:
            deps.append(self.fit_model_rows())
        if self.cluster_columns:
            deps.append(self.fit_model_columns())
        dependencies = self.cluster() + deps + [self.init_data_matrix(), self.load(), self.preprocess_fit()]
        if dendro:
            dependencies.extend(self.calculate_linkage())
        def plot():
            if scaler_or_index is None:
                data = self.get_data_matrix(0)
                scaler = None
            elif scaler_or_index == 0:
                data = self.get_input_matrix(0)
                scaler = self.scalers[0]
            elif scaler_or_index == 1: #columns
                data = self.get_input_matrix(1).T
                scaler = self.scalers[1]
            elif isinstance(scaler_or_index, Scaler) or isinstance(scaler_or_index, sklearn.preprocessing.StandardScaler) or isinstance(scaler_or_index, sklearn.preprocessing.RobustScaler):
                scaler = scaler_or_index
                scaler.fit(self.get_data_matrix(0))
                data = scaler.transform(self.get_data_matrix(0))
            elif hasattr(scaler_or_index, '__call__'):
                scaler = TransformScaler(self.name+'transformation_function', scaler_or_index)
                data = scaler(self.get_data_matrix(0))
            else:
                raise ValueError("Don\'t know how to use this scaler:{}.".format(scaler))

            column_labels = [column_name_mangler(x) for x in self.columns]
            row_labels = self.df[self.row_id_column].values
            #first sort the data
            df_reduced = pd.DataFrame(data, columns = self.columns, index = row_labels) #get rid of any additional columns
            row_index = self.sort_clustered(df_reduced,
                                            self.cluster_rows,
                                            axis = 0,
                                            sort_function = sort_function_row,
                                            cluster_ascending = cluster_ascending
                                            )
            df_reduced = pd.DataFrame(data.T, columns = row_labels, index = self.columns) #get rid of any additional columns
            column_index = self.sort_clustered(df_reduced,
                                               self.cluster_columns,
                                               axis = 1,
                                               sort_function = sort_function_column,
                                               cluster_ascending = cluster_ascending
                                               )
#            print(column_index
#            raise ValueError()
            figure = plt.figure(figsize=shape, dpi = 300)
            if vmin is None or vmax is None:
                norm = matplotlib.colors.Normalize()
            else:
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            if df_plot_columns_or_callable is not None:
                # in case we want to add additional columns, we can sort the rows according to the row_order and plot the sorted columns
                if hasattr(df_plot_columns_or_callable, '__call__'):
                    df_add_columns = df_plot_columns_or_callable()
                else:
                    df_add_columns = df_plot_columns_or_callable
                column_sort_matrix = df_add_columns.as_matrix()[row_index,:]
                if scaler is not None:
                    column_sort_matrix = scaler.transform(column_sort_matrix.T).T
                if transformation is not None:
                    column_sort_matrix = transformation(column_sort_matrix)
                single_width = .6 / data.shape[1]
                axdendro = figure.add_axes([0.1,0.2, single_width, 0.6], frameon = False, label = 'dendro1')
                axdendro.imshow(column_sort_matrix, interpolation='nearest', aspect='auto', norm = norm, cmap=plt.get_cmap(colmap), origin='lower')
                axdendro.yaxis.tick_right()
                axdendro.set_yticks(np.arange(column_sort_matrix.shape[0]), minor = False)
                axdendro.set_xticks(np.arange(column_sort_matrix.shape[1]), minor = False)
                axdendro.set_yticklabels(df_add_columns.index, horizontalalignment='left', size = size_y)
                axdendro.set_xticklabels(df_add_columns.columns, rotation = 'vertical', size = size_x)
            elif hasattr(self, 'get_linkage') and self.cluster_rows and dendro:
                #if we have a linkage getter (scipy like) and actually clustered rows and want a dendrogram
                #plot the dendrogram and set the row index accordingly
                axdendro = figure.add_axes([0.1,0.2,0.2,0.6], frameon = False, label = 'dendro1')
                #Z1 = scipy.cluster.hierarchy.dendrogram(self.get_linkage(axis = 0), orientation='left')
                linkage_matrix = self.get_linkage(axis = 0)
                Z1 = scipy.cluster.hierarchy.dendrogram(linkage_matrix, orientation='left')
                row_index = Z1['leaves']
                axdendro.set_xticks([])
                axdendro.set_yticks([])
            if df_plot_rows_or_callable is not None:
                # in case we want to plot additional rows instead of a tree, we can sort the columns in this df according to our column order and plot the sorted columns
                if hasattr(df_plot_rows_or_callable, '__call__'):
                    df_add_rows = df_plot_rows_or_callable()
                else:
                    df_add_rows = df_plot_rows_or_callable
                single_height = 0.6 / data.shape[0]
                row_sort_matrix = df_add_rows.as_matrix()[:,column_index]
                if scaler is not None:
                    row_sort_matrix = scaler.transform(row_sort_matrix)
                if transformation is not None:
                    row_sort_matrix = transformation(row_sort_matrix)
                axdendro2 = figure.add_axes([0.3,0.82,0.6,single_height], frameon = False, label = 'dendro2')
                axdendro2.imshow(row_sort_matrix, interpolation='nearest', aspect='auto', norm = norm, cmap=plt.get_cmap(colmap), origin='lower')
                axdendro2.yaxis.tick_right()
                axdendro2.set_xticklabels([])
                axdendro2.set_yticks(np.arange(row_sort_matrix.shape[0]), minor = False)
                axdendro2.set_yticklabels(df_add_rows.index, horizontalalignment='left', size = size_y)
                axdendro2.set_xticks(np.arange(row_sort_matrix.shape[1]), minor = False)
            elif hasattr(self, 'get_linkage') and self.cluster_columns and dendro:
                #if we have a linkage getter (scipy like) and actually clustered rows and want a dendrogram
                #plot the dendrogram and set the column index accordingly
                axdendro2 = figure.add_axes([0.3,0.8,0.6,0.2], frameon = False, label = 'dendro2')
                Z2 = scipy.cluster.hierarchy.dendrogram(self.get_linkage(axis = 1), orientation='top')
                column_index = Z2['leaves'][::-1]
                axdendro2.set_xticks([])
                axdendro2.set_yticks([])
            if row_index is not None:
                data = data[row_index,:]
                row_labels = [row_labels[i] for i in row_index]
            if column_index is not None:
                data = data[:,column_index]
                column_labels = [column_labels[i] for i in column_index]
            axmatrix = figure.add_axes([0.3,0.2,0.6,0.6])
            if transformation is not None:
                data = transformation(data)
            im = axmatrix.imshow(data, interpolation='nearest', aspect='auto', norm = norm, cmap=plt.get_cmap(colmap), origin='lower')
            axmatrix.yaxis.tick_right()
            axmatrix.set_xticks(np.arange(data.shape[1]), minor = False)
            axmatrix.set_yticks(np.arange(data.shape[0]), minor = False)
            row_labels = row_labels[::-1]
            if show_row_labels:
                labels_to_plot = row_labels[::-1]
                if row_name_mangler is not None:
                    labels_to_plot = [row_name_mangler(name) for name in labels_to_plot]
                axmatrix.set_yticklabels(labels_to_plot, horizontalalignment='left', size = size_y)
            else:
                axmatrix.set_yticklabels([], horizontalalignment='left', size = size_y)
            if show_column_labels:
                axmatrix.set_xticklabels(column_labels, rotation = 'vertical', size = size_x)
            else:
                axmatrix.set_xticklabels([], rotation = 'vertical', size = size_x)
            if dendro:
                if self.cluster_rows:
                    axcolor = figure.add_axes([0.09,0.875,0.18,0.05])
                    plt.colorbar(im, cax=axcolor, orientation='horizontal')
                    axcolor.set_xticklabels([x.get_text() for x in axcolor.get_xticklabels()], size = size_x)

                elif self.cluster_columns:
                    axcolor = figure.add_axes([0.2,0.2,0.02,0.6])
                    plt.colorbar(im, cax=axcolor, orientation='vertical')
                    axcolor.set_yticklabels([x.get_text() for x in axcolor.get_yticklabels()], size = size_y)
            else:
                axcolor = figure.add_axes([0.09,0.875,0.18,0.05])
                plt.colorbar(im, cax=axcolor, orientation='horizontal')
                axcolor.set_yticklabels([x.get_text() for x in axcolor.get_yticklabels()], size = size_y)
            to_df = {'Ylabel' : row_labels}
            for i, col in enumerate(column_labels):
                to_df[col] = data[:,i]
            df = pd.DataFrame(to_df)
            df.to_csv(outfile+'.tsv', index = False, sep = '\t')
            figure.savefig(outfile)
            #pickle.dump(figure, file(dump_file, 'w'))
        deps = []
        if hasattr(self, 'calculate_linkage'):
            deps.append(self.calculate_linkage())
        #print(self.cluster()
        #raise ValueError()
        return ppg.FileGeneratingJob(outfile, plot).depends_on(dependencies)
                    


    def plot_heatmap_new(self, outpath, sort_strategy, sort_strategy_column = None, dependencies = []):
        """
        sort df according to clustering, an arbitrary column, or via a sort function on the dataframe
        display additional columns that were not used in the clustering
        specify row name column

        automatically set the dimensions
        """
        exptools.common.ensure_path(outpath)
        outfile = os.path.join(outpath, "{}_{}.png".format(self.name, clustering_name))
        def plot():
            df = self.df
            #second sort the dataframe
            for key in self.__dict__:
                print(key)
            print(self.default, type(self.default))
            print(self.default.name)
            print(self.default.clusters)
            print(self.default.model)
            print(self.clustering2, type(self.clustering2))
            print(self.clustering2.clusters)
            print(len(self.clustering2.df_scaled))
            print(self.clustering2.get_linkage())
            print(hasattr(self.clustering2, "get_linkage"))
            print(hasattr(self.default, "get_linkage"))
            raise ValueError()
        return ppg.FileGeneratingJob(outfile, plot).depends_on(dependencies)

                
    def get_custom_metric(self, axis = 0):
        if axis == 0:
            metric = self.metric
        else:
            metric = self.metric_columns
        def calc_distance_matrix(data):
            """
            With a given metric, generate a distance matrix from the scaled matrix.
            """
            distance = np.ones([data.shape[0], data.shape[0]])
            for ii, vec1 in enumerate(data):
                for jj, vec2 in enumerate(data):
                    if jj < ii:
                        continue
                    distance[ii, jj] = metric(vec1, vec2)
                    distance[jj, ii] = distance[ii, jj]
            return distance
        return calc_distance_matrix

    def plot_heatmap(self, outpath, name, column_name_mangler = lambda x : x, colmap = 'seismic',
                      scaler_or_index = None, vmin = None, vmax = None, sort_function_row = None,
                      sort_function_column = None, shape = (50, 50), size_x=20, size_y=10, dendro = False,
                      cluster_ascending = True, dependencies = [], df_plot_rows_or_callable = None,
                      df_plot_columns_or_callable = None, show_column_labels = True, show_row_labels = True, 
                      transformation = None, row_name_mangler = None):
        #sort functions should take a df and return an index to sort by
        """
        @df_plot_rows_or_callable contains additional rows to be plotted and must be in the same order as the input df
        @df_plot_columns_or_callable contains additional columns to be plotted and must be in the same order as the input df
        @sort_function_row sorting function, takes a dataframe with index corresponding to rows and returns numerical sort index
        @sort_function_column sorting function, takes a dataframe with index corresponding to columns and returns numerical sort index
        """
        exptools.common.ensure_path(outpath)
        outfile = os.path.join(outpath, name)
        print(self.default_clusters, type(self.default_clusters))
        print(self.nd_clustering_clusters, type(self.clustering2_clusters))
        raise ValueError()
        dump_file = os.path.join(self.cache_dir, name+'_pickled')
        deps = dependencies
        if self.cluster_rows:
            deps.append(self.fit_model_rows())
        if self.cluster_columns:
            deps.append(self.fit_model_columns())
        dependencies = self.cluster() + deps + [self.init_data_matrix(), self.load(), self.preprocess_fit()]
        if dendro:
            dependencies.extend(self.calculate_linkage())
        def plot():
            if scaler_or_index is None:
                data = self.get_data_matrix(0)
                scaler = None
            elif scaler_or_index == 0:
                data = self.get_input_matrix(0)
                scaler = self.scalers[0]
            elif scaler_or_index == 1: #columns
                data = self.get_input_matrix(1).T
                scaler = self.scalers[1]
            elif isinstance(scaler_or_index, Scaler) or isinstance(scaler_or_index, sklearn.preprocessing.StandardScaler) or isinstance(scaler_or_index, sklearn.preprocessing.RobustScaler):
                scaler = scaler_or_index
                scaler.fit(self.get_data_matrix(0))
                data = scaler.transform(self.get_data_matrix(0))
            elif hasattr(scaler_or_index, '__call__'):
                scaler = TransformScaler(self.name+'transformation_function', scaler_or_index)
                data = scaler(self.get_data_matrix(0))
            else:
                raise ValueError("Don\'t know how to use this scaler:{}.".format(scaler))

            column_labels = [column_name_mangler(x) for x in self.columns]
            row_labels = self.df[self.row_id_column].values
            #first sort the data
            df_reduced = pd.DataFrame(data, columns = self.columns, index = row_labels) #get rid of any additional columns
            row_index = self.sort_clustered(df_reduced,
                                            self.cluster_rows,
                                            axis = 0,
                                            sort_function = sort_function_row,
                                            cluster_ascending = cluster_ascending
                                            )
            df_reduced = pd.DataFrame(data.T, columns = row_labels, index = self.columns) #get rid of any additional columns
            column_index = self.sort_clustered(df_reduced,
                                               self.cluster_columns,
                                               axis = 1,
                                               sort_function = sort_function_column,
                                               cluster_ascending = cluster_ascending
                                               )
#            print(column_index
#            raise ValueError()
            figure = plt.figure(figsize=shape, dpi = 300)
            if vmin is None or vmax is None:
                norm = matplotlib.colors.Normalize()
            else:
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            if df_plot_columns_or_callable is not None:
                # in case we want to add additional columns, we can sort the rows according to the row_order and plot the sorted columns
                if hasattr(df_plot_columns_or_callable, '__call__'):
                    df_add_columns = df_plot_columns_or_callable()
                else:
                    df_add_columns = df_plot_columns_or_callable
                column_sort_matrix = df_add_columns.as_matrix()[row_index,:]
                if scaler is not None:
                    column_sort_matrix = scaler.transform(column_sort_matrix.T).T
                if transformation is not None:
                    column_sort_matrix = transformation(column_sort_matrix)
                single_width = .6 / data.shape[1]
                axdendro = figure.add_axes([0.1,0.2, single_width, 0.6], frameon = False, label = 'dendro1')
                axdendro.imshow(column_sort_matrix, interpolation='nearest', aspect='auto', norm = norm, cmap=plt.get_cmap(colmap), origin='lower')
                axdendro.yaxis.tick_right()
                axdendro.set_yticks(np.arange(column_sort_matrix.shape[0]), minor = False)
                axdendro.set_xticks(np.arange(column_sort_matrix.shape[1]), minor = False)
                axdendro.set_yticklabels(df_add_columns.index, horizontalalignment='left', size = size_y)
                axdendro.set_xticklabels(df_add_columns.columns, rotation = 'vertical', size = size_x)
            elif hasattr(self, 'get_linkage') and self.cluster_rows and dendro:
                #if we have a linkage getter (scipy like) and actually clustered rows and want a dendrogram
                #plot the dendrogram and set the row index accordingly
                axdendro = figure.add_axes([0.1,0.2,0.2,0.6], frameon = False, label = 'dendro1')
                #Z1 = scipy.cluster.hierarchy.dendrogram(self.get_linkage(axis = 0), orientation='left')
                linkage_matrix = self.get_linkage(axis = 0)
                Z1 = scipy.cluster.hierarchy.dendrogram(linkage_matrix, orientation='left')
                row_index = Z1['leaves']
                axdendro.set_xticks([])
                axdendro.set_yticks([])
            if df_plot_rows_or_callable is not None:
                # in case we want to plot additional rows instead of a tree, we can sort the columns in this df according to our column order and plot the sorted columns
                if hasattr(df_plot_rows_or_callable, '__call__'):
                    df_add_rows = df_plot_rows_or_callable()
                else:
                    df_add_rows = df_plot_rows_or_callable
                single_height = 0.6 / data.shape[0]
                row_sort_matrix = df_add_rows.as_matrix()[:,column_index]
                if scaler is not None:
                    row_sort_matrix = scaler.transform(row_sort_matrix)
                if transformation is not None:
                    row_sort_matrix = transformation(row_sort_matrix)
                axdendro2 = figure.add_axes([0.3,0.82,0.6,single_height], frameon = False, label = 'dendro2')
                axdendro2.imshow(row_sort_matrix, interpolation='nearest', aspect='auto', norm = norm, cmap=plt.get_cmap(colmap), origin='lower')
                axdendro2.yaxis.tick_right()
                axdendro2.set_xticklabels([])
                axdendro2.set_yticks(np.arange(row_sort_matrix.shape[0]), minor = False)
                axdendro2.set_yticklabels(df_add_rows.index, horizontalalignment='left', size = size_y)
                axdendro2.set_xticks(np.arange(row_sort_matrix.shape[1]), minor = False)
            elif hasattr(self, 'get_linkage') and self.cluster_columns and dendro:
                #if we have a linkage getter (scipy like) and actually clustered rows and want a dendrogram
                #plot the dendrogram and set the column index accordingly
                axdendro2 = figure.add_axes([0.3,0.8,0.6,0.2], frameon = False, label = 'dendro2')
                Z2 = scipy.cluster.hierarchy.dendrogram(self.get_linkage(axis = 1), orientation='top')
                column_index = Z2['leaves'][::-1]
                axdendro2.set_xticks([])
                axdendro2.set_yticks([])
            if row_index is not None:
                data = data[row_index,:]
                row_labels = [row_labels[i] for i in row_index]
            if column_index is not None:
                data = data[:,column_index]
                column_labels = [column_labels[i] for i in column_index]
            axmatrix = figure.add_axes([0.3,0.2,0.6,0.6])
            if transformation is not None:
                data = transformation(data)
            im = axmatrix.imshow(data, interpolation='nearest', aspect='auto', norm = norm, cmap=plt.get_cmap(colmap), origin='lower')
            axmatrix.yaxis.tick_right()
            axmatrix.set_xticks(np.arange(data.shape[1]), minor = False)
            axmatrix.set_yticks(np.arange(data.shape[0]), minor = False)
            row_labels = row_labels[::-1]
            if show_row_labels:
                labels_to_plot = row_labels[::-1]
                if row_name_mangler is not None:
                    labels_to_plot = [row_name_mangler(name) for name in labels_to_plot]
                axmatrix.set_yticklabels(labels_to_plot, horizontalalignment='left', size = size_y)
            else:
                axmatrix.set_yticklabels([], horizontalalignment='left', size = size_y)
            if show_column_labels:
                axmatrix.set_xticklabels(column_labels, rotation = 'vertical', size = size_x)
            else:
                axmatrix.set_xticklabels([], rotation = 'vertical', size = size_x)
            if dendro:
                if self.cluster_rows:
                    axcolor = figure.add_axes([0.09,0.875,0.18,0.05])
                    plt.colorbar(im, cax=axcolor, orientation='horizontal')
                    axcolor.set_xticklabels([x.get_text() for x in axcolor.get_xticklabels()], size = size_x)

                elif self.cluster_columns:
                    axcolor = figure.add_axes([0.2,0.2,0.02,0.6])
                    plt.colorbar(im, cax=axcolor, orientation='vertical')
                    axcolor.set_yticklabels([x.get_text() for x in axcolor.get_yticklabels()], size = size_y)
            else:
                axcolor = figure.add_axes([0.09,0.875,0.18,0.05])
                plt.colorbar(im, cax=axcolor, orientation='horizontal')
                axcolor.set_yticklabels([x.get_text() for x in axcolor.get_yticklabels()], size = size_y)
            to_df = {'Ylabel' : row_labels}
            for i, col in enumerate(column_labels):
                to_df[col] = data[:,i]
            df = pd.DataFrame(to_df)
            df.to_csv(outfile+'.tsv', index = False, sep = '\t')
            figure.savefig(outfile)
            #pickle.dump(figure, file(dump_file, 'w'))
        deps = []
        if hasattr(self, 'calculate_linkage'):
            deps.append(self.calculate_linkage())
        #print(self.cluster()
        #raise ValueError()
        return ppg.FileGeneratingJob(outfile, plot).depends_on(dependencies)


    def plot_heatmap_combined(self, filename, result_dir, heatmap_dump, dependencies = []):
        def plot():
            #figure_heatmap = pickle.load(file(heatmap_dump, 'r'))
            #print(figure_heatmap.axes)
            raise ValueError()
        return ppg.FileGeneratingJob(os.path.join(self.cache_dir, filename), plot).depends_on(dependencies)


class ScipyHierarchical(ML):

    def __init__(self, name, genes_or_df_or_loading_function, columns, threshold, cluster_axis = 'rows', metric = 'euclidean',
                 metric_columns = None, linkage = 'ward', linkage_columns = None,
                 no_of_clusters = 10, scaler = None,
                  no_of_clusters_columns = None, row_id_column = 'stable_id',
                 dependencies = [], annotators = [],
                 column_order_or_sorting_function = None,
                 row_order_or_sorting_function = None,
                 imputer = None, missing_value = np.NaN):
        """
        @metric can be a string refering to usable scipy linkage distance functions or a function that calculates a
        distance matrix from the observed data
        """
        self.metric = metric
        self.linkage = linkage
        self.threshold = threshold
        self.metric_columns = metric if metric_columns is None else metric_columns
        self.linkage_columns = linkage if linkage_columns is None else linkage_columns
        precomputed = metric == 'precomputed'
        if hasattr(metric, '__call__'):
            self.affinity = self.get_custom_metric(0)
        elif metric in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']:
            self.affinity = metric
        else:
            raise ValueError('Invalid metric: %s. Allowed is ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"] or a callable.'%self.metric)
        if linkage is 'ward' and metric is not 'euclidean':
                raise ValueError('If linkage is ward, metric must be euclidean, was %s.' % (self.metric))
        if metric_columns is None:
            self.affinity_columns = self.affinity
        elif hasattr(self.metric_columns, '__call__'):
            self.metric_columns = metric_columns
            self.affinity_columns = self.get_custom_metric(1)
        elif self.metric_columns in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']:
            self.affinity_columns = self.metric_columns
        else:
            raise ValueError('Invalid metric_columns: %s. Allowed is ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"] or a callable.'%self.metric_columns)
        if self.linkage_columns is 'ward' and self.metric_columns is not 'euclidean':
            raise ValueError('If linkage is ward, metric (metric_columns) must be euclidean, was %s.' % (self.metric_columns))
        ML.__init__(self, name, genes_or_df_or_loading_function, columns, cluster_axis, dependencies, annotators, row_id_column, scaler, precomputed, imputer, missing_value)
        self.row_order_or_sorting_function = row_order_or_sorting_function
        self.column_order_or_sorting_function = column_order_or_sorting_function
        dependencies.extend([
                         ppg.ParameterInvariant(self.name+'_params', [metric, linkage, no_of_clusters, metric_columns, linkage_columns, no_of_clusters_columns, cluster_axis] + self.columns),
                         ])
        self.no_of_clusters = no_of_clusters
        self.no_of_clusters_columns = no_of_clusters if no_of_clusters_columns is None else no_of_clusters_columns

    def fit_model_rows(self):
        def compute():
            """
            loads the cluster numbers for the rows
            """
            return scipy.cluster.hierarchy.fcluster(self.get_linkage(axis = 0), self.threshold, criterion='inconsistent', depth=2, R=None, monocrit=None)
        return ppg.CachedAttributeLoadingJob(os.path.join(self.cache_dir, self.name+"_model_row"), self, 'model_row', compute).depends_on(self.init_data_matrix()).depends_on(self.calculate_linkage_rows())

    def fit_model_columns(self):
        """
        loads the cluster numbers for the rows
        """
        def compute():
            return scipy.cluster.hierarchy.fcluster(self.get_linkage(axis = 1), self.threshold, criterion='inconsistent', depth=2, R=None, monocrit=None)
        return ppg.CachedAttributeLoadingJob(
            os.path.join(self.cache_dir, self.name+"_model_column"), self, 'model_column', compute).depends_on(self.init_data_matrix()).depends_on(self.calculate_linkage_columns())

    def calculate_linkage_rows(self):
        def __load():
            data = self.get_input_matrix(axis = 0)
            linkage = scipy.cluster.hierarchy.linkage(data, method=self.linkage, metric=self.metric)
            return linkage
        return ppg.CachedAttributeLoadingJob(os.path.join(self.cache_dir, self.name+"_linkage_row"), self, 'linkage_row', __load).depends_on(self.init_data_matrix()).depends_on(self.dependencies).depends_on(self.preprocess_fit())

    def calculate_linkage_columns(self):
        def __load():
            data = self.get_input_matrix(axis = 1)
            linkage = scipy.cluster.hierarchy.linkage(data, method=self.linkage_columns, metric=self.metric_columns)
            return linkage
        return ppg.CachedAttributeLoadingJob(os.path.join(self.cache_dir, self.name+"_linkage_column"), self, 'linkage_column', __load).depends_on(self.init_data_matrix()).depends_on(self.dependencies).depends_on(self.preprocess_fit())

    def calculate_linkage(self):
        jobs = []
        if self.cluster_rows:
            jobs.append(self.calculate_linkage_rows())
        if self.cluster_columns:
            jobs.append(self.calculate_linkage_columns())
        return jobs

    def get_linkage(self, axis):
        if axis == 0:
            return self.linkage_row
        return self.linkage_column

#class SomeClass():##
#
#   def __init__(self, name, method, cluster_columns):
#        self.name = name
#        self.cluster_columns = cluster_columns
#        self.method = method

#class Clustering():
#    
#    def __init__(self, name, method, cluster_columns, dependencies = []):
#        self.name = name
#        self.method = method
#        self.cluster_columns = cluster_columns
#        self.dependencies = dependencies
##        self.dependencies.extend()
#        #if method is None:
#        #    self.method = NewKMeans(name, 2)
##        self.dependencies.extend([
##            ppg.ParameterInvariant(self.name+'_basic_parameters', [self.method, self.method.imputer, self.method.scaler, self.missing_value])
##            ])
#        
#    def get_dependencies():
#        return self.dependencies + [self.method.get_dependencies()]
#    
#    def fit(self, df, sample_weights = None):
#        if not self.cluster_columns:
#            df = df.transpose()
#        df_imputed = df.transform(self.method.imputer.transform)
#        df_imputed = df_imputed[df_imputed.max(axis = 1) > 0] #can't have features with only zeros in the matrix
#        self.sample_weights = sample_weights
#        self.df_scaled = df_imputed.transform(self.method.scaler.transform)
#        self.model = self.method.fit(self.df_scaled)
#        print(type(self.model))
#        self.clusters = self.model.predict(self.df_scaled.values) # apparently, oredicts takes only 2 positional arguments instead of 3
#        
#    def predict(self, df_other, sample_weights):
#        if sample_weights == None:
#            sample_weights = self.sample_weights
#        df_imputed_other = df_other.transform(self.imputer.transform)
#        df_imputed_other = df_imputed_other[df_imputed_other.max(axis = 1) > 0] 
#        df_scaled_other = df_imputed_other.transform(self.scaler.transform)
#        return self.model.predict(df_scaled_other.values, sample_weights)        
    
class ClusteringMethod():
    
    def __init__(self, name):
        #, scaler, imputer, missing_value, cluster_columns):
        """
        This is a wrapper for any clustering approach. 
        @param scaler scaler class for data scaling
        @param imputer class for imputing missing values
        @param missing_value missing_value
        """
        self.name = name
        #self.scaler = scaler
        #self.imputer = imputer
        #self.missing_value = missing_value
        #if scaler is None:
        #    self.scaler = Scaler()
        #elif isinstance(scaler, Scaler):
        #    self.scaler = scaler
        #elif isinstance(scaler, sklearn.base.TransformerMixin):
        #    self.scaler = scaler
        #    self.scaler.name = scaler.__class__.__name__
        #elif scaler == 'default':
        #    self.scaler = sklearn.preprocessing.StandardScaler()
        #    self.scaler.name = 'default'
        #elif scaler == 'robust':
        #    self.scaler = sklearn.preprocessing.RobustScaler()
        #    self.scaler.name = 'robust'
        #else:
        #    raise ValueError('No valid scaler specified, was {}.'.format(scaler))
        #self.scalers = [self.scaler, copy.copy(self.scaler)]
        #if imputer == 'default' or imputer is None:
        #    self.imputer = ImputeFixed(missing_value, replacement_value = 0)
        #else:
        #    self.imputer = imputer
        #self.clustering = None
        #self.cluster_columns = cluster_columns
    
    def get_dependencies(self):
        pass

    def transform(self, df):    
        df_imputed = df.transform(self.imputer.transform)
        df_imputed = df_imputed[df_imputed.max(axis = 1) > 0] #can't have features with only zeros in the matrix
        df_scaled = df_imputed.transform(self.scaler.transform)
        return df_scaled

    def fit(self, df, sample_weights = None):
#        if not self.cluster_columns:
#              df = df.transpose()
        #df_imputed = df.transform(self.imputer.transform)
        #df_imputed = df_imputed[df_imputed.max(axis = 1) > 0] #can't have features with only zeros in the matrix
        #self.sample_weights = sample_weights
        #self.df_scaled = df_imputed.transform(self.scaler.transform)
        self.model = self.clustering.fit(df, sample_weights)
        cluster_ids = self.model.labels_
        index = df.index.values
        self.clusters = pd.DataFrame({self.name : cluster_ids}, index = index) 

    def predict(self, df_other, sample_weights):
        if sample_weights == None:
            sample_weights = self.sample_weights
        df_imputed_other = df_other.transform(self.imputer.transform)
        df_imputed_other = df_imputed_other[df_imputed_other.max(axis = 1) > 0] 
        df_scaled_other = df_imputed_other.transform(self.scaler.transform)
        return self.model.predict(df_scaled_other.values, sample_weights)        
    
class NewKMeans(ClusteringMethod):


    def __init__(self, name, no_of_clusters = 2, n_init = 10, max_iter = 300, 
                 random_state = None, n_jobs = 1, init = 'k-means++'):
        """
        This is a wrapper for Kmeans
        @param genes_or_df_or_loading_function can be either genes, df or a loading function
        @param init 'k-means++', 'random' or ndarray of centroids
        @param max_iter max iterations til convergence
        @param cluster columns should we cluster columns?
        @params cluster_rows should we cluster rows?
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
            n_clusters = self.no_of_clusters,
            init='k-means++',
            n_init=self.n_init,
            max_iter=self.max_iter,
            algorithm='auto',
            tol=0.0001,
            precompute_distances='auto',
            verbose=0,
            random_state=self.random_state,
            copy_x=True,
            n_jobs=self.n_jobs
            )

    def get_dependencies(self):
        return [ppg.ParameterInvariant(
                self.name+'parameters', [
                    self.n_init, 
                    self.max_iter, 
                    self.no_of_clusters, 
                    self.random_state, 
                    self.scaler, 
                    self.n_init, 
                    self.n_jobs]
                )
            ]

class DBSCAN(ClusteringMethod):

    def __init__(self, name, scaler = None, imputer = None, missing_value = np.NaN, cluster_columns = False, 
                 eps = .5, min_samples = 5, metric = 'euclidean', algorithm = 'auto', leaf_size = 30,
                 p = None, n_jobs = 1, dependencies = []):
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
                         ppg.ParameterInvariant(self.name+'_parameters', [eps, p, min_samples, metric,
                                                                         algorithm, leaf_size, p,
                                                                         scaler,
                                                                         ])
                         ]
        ClusteringMethod.__init__(self, name, scaler, imputer, missing_value, cluster_columns)
        self.clustering = sklearn.cluster.DBSCAN(
                                        eps=self.eps,
                                        min_samples=self.min_samples,
                                        metric=self.metric,
                                        algorithm=self.algorithm,
                                        leaf_size=self.leaf_size,
                                        p=self.p,
                                        n_jobs=self.n_jobs
                                        )
        self.clustering.predict = self.clustering.fit_predict

    def get_dependencies(self):
        return [ppg.ParameterInvariant(
                self.name+'parameters', [
                    self.eps, 
                    self.min_samples, 
                    self.metric, 
                    self.algorithm, 
                    self.leaf_size,
                    self.p, 
                    self.n_jobs, 
                    self.scaler, 
                    self.imputer, 
                    self.missing_value, 
                    self.cluster_columns]
                )
            ]

class SKlearnAgglomerative(ClusteringMethod):

    def __init__(self, name, no_of_clusters = 2, scaler = None, imputer = None, 
                 missing_value = np.NaN, cluster_columns = False, 
                 affinity='euclidean', linkage='ward', 
                 connectivity=None, compute_full_tree='auto', 
                 memory=None
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
        ClusteringMethod.__init__(self, name, scaler, imputer, missing_value, cluster_columns)
        self.clustering = sklearn.cluster.AgglomerativeClustering(
            n_clusters=self.no_of_clusters, affinity=self.affinity, memory=self.memory, 
            connectivity=self.connectivity, compute_full_tree=self.compute_full_tree, 
            linkage=self.linkage)
        self.clustering.predict = self.clustering.fit_predict
        
    def get_dependencies(self):
        return [ppg.ParameterInvariant(
                self.name+'parameters', [
                    self.no_of_clusters, 
                    self.affinity, 
                    self.linkage, 
                    self.connectivity, 
                    self.compute_full_tree,
                    self.memory, 
                    self.scaler, 
                    self.imputer, 
                    self.missing_value, 
                    self.cluster_columns]
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
            X = sklearn.cluster.hierarchical.check_array(X, ensure_min_samples=2, estimator=self)
            memory = sklearn.cluster.hierarchical.check_memory(self.memory)

            if self.n_clusters <= 0:
                raise ValueError("n_clusters should be an integer greater than 0."
                                " %s was provided." % str(self.n_clusters))

            if self.linkage == "ward" and self.affinity != "euclidean":
                raise ValueError("%s was provided as affinity. Ward can only "
                                "work with euclidean distances." %
                                (self.affinity, ))

            if self.linkage not in sklearn.cluster.hierarchical._TREE_BUILDERS:
                raise ValueError("Unknown linkage type %s. "
                                "Valid options are %s" % (self.linkage,
                                                        sklearn.cluster.hierarchical._TREE_BUILDERS.keys()))
            tree_builder = sklearn.cluster.hierarchical._TREE_BUILDERS[self.linkage]

            connectivity = self.connectivity
            if self.connectivity is not None:
                if callable(self.connectivity):
                    connectivity = self.connectivity(X)
                connectivity = check_array(
                    connectivity, accept_sparse=['csr', 'coo', 'lil'])

            n_samples = len(X)
            compute_full_tree = self.compute_full_tree
            if self.connectivity is None:
                compute_full_tree = True
            if compute_full_tree == 'auto':
                # Early stopping is likely to give a speed up only for
                # a large number of clusters. The actual threshold
                # implemented here is heuristic
                compute_full_tree = self.n_clusters < max(100, .02 * n_samples)
            n_clusters = self.n_clusters
            if compute_full_tree:
                n_clusters = None

            # Construct the tree
            kwargs = {'return_distance': True}
            if self.linkage != 'ward':
                kwargs['linkage'] = self.linkage
                kwargs['affinity'] = self.affinity
            self.children_, self.n_components_, self.n_leaves_, parents, self.distances = \
                memory.cache(tree_builder)(X, connectivity,
                                        n_clusters=n_clusters,
                                        **kwargs)
            # Cut the tree
            if compute_full_tree:
                self.labels_ = sklearn.cluster.hierarchical._hc_cut(self.n_clusters, self.children_,
                                    self.n_leaves_)
            else:
                labels = sklearn.cluster.hierarchical._hierarchical.hc_get_heads(parents, copy=False)
                # copy to avoid holding a reference on the original array
                labels = np.copy(labels[:n_samples])
                # Reassign cluster numbers
                self.labels_ = np.searchsorted(np.unique(labels), labels)
            return self
        return my_fit      
    
    def fit(self, df, sample_weights = None):
#        if not self.cluster_columns:
#              df = df.transpose()
        df_imputed = df.transform(self.imputer.transform)
        df_imputed = df_imputed[df_imputed.max(axis = 1) > 0] #can't have features with only zeros in the matrix
        self.sample_weights = sample_weights
        self.df_scaled = df_imputed.transform(self.scaler.transform)
        #self.model = self.clustering.fit(self.clustering, self.df_scaled)
        fitter = self.new_fit_function()
        self.model = fitter(self.clustering, self.df_scaled)
        cluster_ids = self.model.labels_
        self.clusters = dict(zip(self.df_scaled.index.values, cluster_ids))
        for index in df.index.values:
            if index not in self.clusters:
                self.clusters[index] = -1
    
    def get_linkage(self):
        transformed_data = self.df_scaled
        number_of_observations = len(self.df_scaled)
        observations = dict([(x, 1) for x in range(number_of_observations)])
        current = 0
        ret = []
        for ii, item in enumerate(zip(self.model.children_, self.model.distances)):
            x = item[0]
            distance = item[1]
            o = observations[x[0]] + observations[x[1]]
            observations[number_of_observations+ii] = o
            ret.append([x[0], x[1], distance, o])
        return np.array(ret, dtype=float)

class ScipyAgglomerative(ClusteringMethod):

    def __init__(self, name, no_of_clusters, threshold, scaler = None, imputer = None, 
                 missing_value = np.NaN, cluster_columns = False, 
                 affinity='euclidean', linkage='ward', 
    ):
        """
        This is a wrapper for hierarchical clustering from scipy. Use this if you want dendrograms.
        """
        self.no_of_clusters = no_of_clusters
        self.threshold = threshold
        self.affinity = affinity
        self.linkage = linkage
        ClusteringMethod.__init__(self, name, scaler, imputer, missing_value, cluster_columns)

    def get_dependencies(self):
        return [ppg.ParameterInvariant(
                self.name+'parameters', [
                    self.no_of_clusters, 
                    self.threshold,
                    self.affinity, 
                    self.linkage, 
                    self.connectivity, 
                    self.compute_full_tree,
                    self.memory, 
                    self.scaler, 
                    self.imputer, 
                    self.missing_value, 
                    self.cluster_columns]
                )
            ]

    def fit(self, df, sample_weights = None):
        """
        Now transformation and imputation is realised in ML, we assume that the dataframe we have is already fully transformaed and scaled
        """
        self.model = self.clustering.fit(self.df_scaled, sample_weights)
        self.linkage = scipy.cluster.hierarchy.linkage(self.df_scaled.as_matrix(), method=self.linkage, metric=self.affinity)
        cluster_ids = scipy.cluster.hierarchy.fcluster(linkage, self.threshold)
        self.clusters = dict(zip(self.df_scaled.index.values, cluster_ids))
        for index in df.index.values:
            if index not in self.clusters:
                self.clusters[index] = -1
        
    def predict(self, df_other, sample_weights):
        return None        

    def get_linkage(self):
        return self.linkage
            
"""
def cluster_genes_kmeans2(outfile, df_windows_filename, df_sorted_file, genome, 
                         n_clusters = 8, shape = (20, 30), 
                         vmin = -2, vmax = 2, colmap = 'seismic', size_y = 3,
                         size_x = 20, dependencies = [], gene_label_off = False,
                         cluster_order = None):
    def __dump():
        df_sorted = pd.read_csv(df_sorted_file, sep = '\t')
        column_names_ordered = df_sorted['Cell'].values
        clustering = sklearn.cluster.KMeans(
            n_clusters=n_clusters, 
            init = 'k-means++', 
            n_init = 10, 
            max_iter = 300, 
            tol = 0.0001, 
            precompute_distances = 'auto', 
            verbose = 0, 
            random_state = None, 
            copy_x = True, 
            n_jobs = None, 
            algorithm = 'auto' 
            )
        df_genes = pd.read_csv(df_windows_filename, sep = '\t')
        df_genes.index = df_genes.stable_id
        df_genes = df_genes[column_names_ordered]
        names = [genome.gene_id_to_name(stable_id)[0] for stable_id in df_genes.index.values]
        df_genes.dropna(inplace = True)
        df_genes = df_genes.subtract(df_genes.mean(axis=1), axis = "rows").divide(df_genes.std(axis = 1), axis = 'rows')
        data = df_genes.as_matrix(column_names_ordered)
        clusters = clustering.fit_predict(data)
        df_genes['cluster'] = clusters
        if cluster_order is not None:
            df_genes['cluster_ordered'] = [cluster_order[c] for c in clusters]
            df_genes = df_genes.sort_values('cluster_ordered', ascending = False)
        else:
            df_genes = df_genes.sort_values('cluster')
        data = df_genes.as_matrix(column_names_ordered)
        figure = plt.figure(figsize=shape, dpi = 300)
        if vmin is None or vmax is None:
            norm = matplotlib.colors.Normalize()
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            
        axe = figure.add_axes([.1, .2 ,.8 ,.7])
        if colmap is None:
            cmap=plt.get_cmap('seismic')
        else:
            cmap = colmap
            
        axe.set_xticks(np.arange(data.shape[1]), minor = False)
        axe.set_yticks(np.arange(data.shape[0]), minor = False)
        row_labels = names[::-1]
        axe.set_yticklabels(row_labels, horizontalalignment='right', size = size_y)
        axe.set_xticklabels([])
        im = axe.imshow(data[::-1], interpolation='nearest', aspect='auto', norm = norm, cmap=cmap, origin='lower')
        if gene_label_off:
            axe.set_yticklabels([])
            axe.set_yticks([])
        axcolor = figure.add_axes([.34, .15, .32, 0.02])
        cbar = plt.colorbar(im, cax=axcolor, orientation='horizontal')
        cbar.set_ticks(np.array([-2, 0, 2]))
        cbar.ax.tick_params(labelsize = size_x)
        figure.savefig(outfile)
        df_genes.to_csv(outfile+'.tsv', index = True, sep = '\t')
    return ppg.PlotJob(outfile, __dump).depends_on(dependencies)
"""
