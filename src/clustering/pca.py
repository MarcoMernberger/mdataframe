'''
Created on Mar 23, 2016

This module is used for multi-dimensional scaling.
@author: mernberger
'''
import matplotlib
from matplotlib.mlab import PCA as mlabPCA
from matplotlib import pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pypipegraph as ppg
import os
import numpy as np
import sklearn
from sklearn import manifold
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import euclidean_distances
import sklearn.discriminant_analysis
import sklearn.datasets
from .clustering import ML_Base
import scipy
import scipy.stats
from scipy.interpolate import interp1d
#import seaborn.apionly as sns
import itertools
from pathlib import Path

class PCA(ML_Base):
    
    def __init__(self, name, genes_or_df_or_loading_function, columns, k, class_label_dict_or_function, axis,
                 dependencies, annotators, row_id_column, label_mangler, scaler, imputer, missing_value):
        """
        This is a wrapper class for a principle component analysis, that takes a dataframe or a
        genomic region alternatively to calculate a PCA.
        @df_or_gene Dataframe or GenomicRegion containing d-dimensional data
        @columns Dataframe columns that contain the actual data
        @classlabel_column column that contains class labels
        @axis 1 if columns are instances, 0 if rows are instances
        @k number of dimensions
        """
        ML_Base.__init__(self, name, genes_or_df_or_loading_function, columns, dependencies, annotators, row_id_column, scaler, imputer, missing_value)
        self.result_dir = Path('results') / 'PCA' / self.name
        self.cache_dir =  Path('cache') / 'pca' / self.name
        self.result_dir.mkdir(parents = True, exist_ok = True)
        self.cache_dir.mkdir(parents = True, exist_ok = True)
        self.d = len(columns)
        self.k = k
        self.label_mangler = label_mangler
        self.class_label_dict_or_function = class_label_dict_or_function
        self.axis = axis
        self.axis_label = 'principal component'
            
    def fit_transform(self):
        raise NotImplementedError()
    
    def dump_transformed_matrix(self):
        def dunp():
            pass
        pass
 
    def plot_2d_projection(self, filename = None, color_callable = None, dependencies = [], class_order = None):
        """
        This assumes that self.transformed_matrix is an array-like object with shape (n_samples, n_components)
        """
        if filename is None:
            outfile = self.result_dir / self.name+"_2D_projection.pdf"
        else:
            outfile = filename
        if self.k < 2:
            raise ValueError("No 2D projection possible with only %s components, set k >= 2." % self.k)
        def plot():
            if isinstance(self.class_label_dict_or_function, dict) or self.class_label_dict_or_function is None:
                class_label_dict = self.class_label_dict_or_function
            elif hasattr(self.class_label_dict_or_function, '__call__'):
                class_label_dict = self.class_label_dict_or_function()
            else:
                raise ValueError("class_label_dict was of type {}".format(type(self.class_label_dict_or_function)))
            markers = itertools.cycle(('o', 'v', '^', '*', 's', '+'))
            figure = plt.figure(figsize=(8, 8))
            ax_data = figure.add_subplot(111)            
            matrix_columns = ["{}_{}".format(self.axis_label, i) for i in range(2)]
            df = pd.DataFrame(self.transformed_matrix[:,:2], columns = matrix_columns)
            print(df.shape)
            #1 if columns are instances, 0 if rows are instances
            if self.axis == 1:
                ids = self.columns
            else:
                ids = self.df[self.row_id_column].values
            if class_label_dict is not None:
                df['class_label'] = [class_label_dict[instance_id] for instance_id in ids]            
            if 'class_label' in df.columns:
                if class_order is None:
                    labels = list(set(class_label_dict.values()))
                else:
                    labels = class_order
                for i, label in enumerate(labels):
                    df_sub = df[df['class_label'] == label][matrix_columns]
                    matrix_sub = df_sub.as_matrix()
                    ax_data.plot(matrix_sub[:,0],matrix_sub[:,1], marker = next(markers), markersize=7, alpha=0.5, label=label, linestyle="None")
                plt.title('Transformed samples with class labels')
            else:
                color = 'blue'
                if color_callable is not None:
                    color = color_callable(ids)
                ax_data.scatter(self.transformed_matrix[:,0], self.transformed_matrix[:,1], marker = 'o', c=color, cmap = 'plasma', alpha=0.5)
                plt.title('Transformed samples without classes')
            xmin = np.floor(np.min(self.transformed_matrix[:,0]))
            ymin = np.floor(np.min(self.transformed_matrix[:,1]))
            xmax = np.ceil(np.max(self.transformed_matrix[:,0]))
            ymax = np.ceil(np.max(self.transformed_matrix[:,1]))
            ax_data.set_xlim([1.3*xmin, 1.3*xmax])
            ax_data.set_ylim([1.3*ymin, 1.3*ymax])
            ax_data.set_xlabel('1st %s'%self.axis_label)
            ax_data.set_ylabel('2nd %s'%self.axis_label)
            ax_data.legend()            
            for i, instance_id in enumerate(ids):
                    plt.annotate(
                        self.label_mangler(instance_id), 
                        xy = (self.transformed_matrix[i,0], self.transformed_matrix[i,1]), xytext = (-1, 1),
                        textcoords = 'offset points', ha = 'right', va = 'bottom', size = 3)
            figure.savefig(outfile)
        return ppg.FileGeneratingJob(outfile, plot).depends_on(self.fit_transform()).depends_on(self.init_data_matrix()).depends_on(self.load()).depends_on(dependencies)


    def plot_3d_projection(self, filename = None, color_callable = None, dependencies = [], class_order = None):
        """
        This assumes that self.transformed_matrix is an array-like object with shape (n_samples, n_components)
        """        
        if self.k < 3:
            raise ValueError("No 3D prjection possible with only %s components, set k >= 3." % self.k)
        if filename is None:
            outfile = os.path.join(self.result_dir, self.name+"_3D_projection.pdf")
        else:
            outfile = filename
        def plot():
            if isinstance(self.class_label_dict_or_function, dict) or self.class_label_dict_or_function is None:
                class_label_dict = self.class_label_dict_or_function
            elif hasattr(self.class_label_dict_or_function, '__call__'):
                class_label_dict = self.class_label_dict_or_function()
            else:
                raise ValueError("class_label_dict was of type {}".format(type(self.class_label_dict_or_function)))
            markers = itertools.cycle(('o', 'v', '^', '*', 's', '+'))
            figure = plt.figure(figsize=(8, 8))
            ax3d = figure.add_subplot(111, projection = '3d')
            matrix_columns = ["{}_{}".format(self.axis_label, i) for i in range(3)]
            df = pd.DataFrame(self.transformed_matrix[:,:3], columns = matrix_columns)
            #1 if columns are instances, 0 if rows are instances
            if self.axis == 1:
                ids = self.columns
            else:
                ids = self.df[self.row_id_column].values
            if class_label_dict is not None:
                df['class_label'] = [class_label_dict[instance_id] for instance_id in ids]            
            if 'class_label' in df.columns:
                if class_order is None:
                    labels = list(set(class_label_dict.values()))
                else:
                    labels = class_order
                for i, label in enumerate(labels):
                    df_sub = df[df['class_label'] == label][matrix_columns]
                    matrix_sub = df_sub.as_matrix()
                    ax3d.plot(
                              matrix_sub[:,0],
                              matrix_sub[:,1],
                              matrix_sub[:,2],
                              marker = next(markers), 
                              markersize=7, 
                              alpha=0.5, 
                              label=label,
                              linestyle="None"
                              )
                plt.title('Transformed samples with class labels')
            else:
                color = 'blue'
                if color_callable is not None:
                    color = color_callable(ids)
                ax3d.scatter(
                          self.transformed_matrix[:,0], 
                          self.transformed_matrix[:,1], 
                          self.transformed_matrix[:,1],
                          marker = 'o', 
                          c=color, 
                          cmap = 'plasma', 
                          alpha=0.5
                          )
                plt.title('Transformed samples without classes')
            xmin = np.floor(np.min(self.transformed_matrix[:,0]))
            ymin = np.floor(np.min(self.transformed_matrix[:,1]))
            zmin = np.floor(np.min(self.transformed_matrix[:,2]))
            xmax = np.ceil(np.max(self.transformed_matrix[:,0]))
            ymax = np.ceil(np.max(self.transformed_matrix[:,1]))
            zmax = np.ceil(np.max(self.transformed_matrix[:,2]))
            ax3d.set_xlim([1.3*xmin, 1.3*xmax])
            ax3d.set_ylim([1.3*ymin, 1.3*ymax])
            ax3d.set_zlim([1.3*zmin, 1.3*zmax])
            ax3d.set_xlabel('1st %s'%self.axis_label)
            ax3d.set_ylabel('2nd %s'%self.axis_label)
            ax3d.set_zlabel('3rd %s'%self.axis_label)
            ax3d.legend()
            for i, instance_id in enumerate(ids):
                    plt.annotate(
                        instance_id, 
                        xy = (self.transformed_matrix[i,0], self.transformed_matrix[i,1]), xytext = (-1, 1),
                        textcoords = 'offset points', ha = 'right', va = 'bottom', size = 3)
                    ax3d.text(self.transformed_matrix[i,0],self.transformed_matrix[i,1],self.transformed_matrix[i,2],  '%s' % (self.label_mangler(instance_id)), size=3, zorder=1, color='k') 
            figure.savefig(outfile)
        return ppg.FileGeneratingJob(outfile, plot).depends_on(self.fit_transform()).depends_on(self.init_data_matrix()).depends_on(self.load()).depends_on(dependencies)
 
    
    def plot_3d_data(self):
        """
        This is for plotting the data directly, assuming that it is 3D ... for testing purposes.
        """
        filename = os.path.join(self.result_dir, self.name + "3d_data.pdf")
        def plot():
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
            markers = ['o', 'v', '^', '*', 's', '+']
            figure = plt.figure(figsize=(10, 10))
            ax3d = figure.add_subplot(111, projection = '3d')
            if self.class_label_dict is not None:
                last = 0
                for i, class_label in enumerate(self.class_label_dict):
                    l = len(self.class_label_dict[class_label])
                    ax3d.plot(
                              self.matrix[last:last+l,0],
                              self.matrix[last:last+l,1], 
                              self.matrix[last:last+l,2], 
                              markers[i], 
                              markersize=7, 
                              color=colors[i], 
                              alpha=0.5, 
                              label=class_label
                              )
                    last = last+l
                plt.title('Transformed samples with class labels')
            else:
                ax3d.plot(
                          self.transformed_matrix[:,0], 
                          self.transformed_matrix[:,1], 
                          self.transformed_matrix[:,2], 
                          'o', 
                          markersize=7, 
                          color='blue', 
                          alpha=0.5
                          )
                plt.title('Transformed samples without classes')
            xmin = np.floor(np.min(self.transformed_matrix[:,0]))
            ymin = np.floor(np.min(self.transformed_matrix[:,1]))
            zmin = np.floor(np.min(self.transformed_matrix[:,2]))
            xmax = np.ceil(np.max(self.transformed_matrix[:,0]))
            ymax = np.ceil(np.max(self.transformed_matrix[:,1]))
            zmax = np.ceil(np.max(self.transformed_matrix[:,2]))
            ax3d.set_xlim([2*xmin, 2*xmax])
            ax3d.set_ylim([2*ymin, 2*ymax])
            ax3d.set_zlim([2*zmin, 2*zmax])
            ax3d.set_xlabel('%s %i'%(self.axis_label, 1))
            ax3d.set_ylabel('%s %i'%(self.axis_label, 2))
            ax3d.set_zlabel('%s %i'%(self.axis_label, 3))
            ax3d.legend()
            figure.savefig(filename)
        return ppg.FileGeneratingJob(filename, plot).depends_on(self.fit_transform()).depends_on(self.init_data_matrix())

    def plot_jitter_component(self, filename = None, df_colors = None, component = 1, bins = 30, axis_label = 'component', class_order = None, styles = ['histogram', 'swarm', 'distribution', 'splines']):
        if filename is None:
            filename = os.path.join(self.result_dir, self.name + "component_projection_%i.pdf" % component)
        def plot():
            import seaborn as sns
            if isinstance(self.class_label_dict_or_function, dict) or self.class_label_dict_or_function is None:
                class_label_dict = self.class_label_dict_or_function
            elif hasattr(self.class_label_dict_or_function, '__call__'):
                class_label_dict = self.class_label_dict_or_function()
            else:
                raise ValueError("class_label_dict was of type {}".format(type(self.class_label_dict_or_function)))
            #calc
            embedding = self.transformed_matrix
            df = pd.DataFrame({'1' : embedding[:,0], '2' : embedding[:,1], '3' : embedding[:,2]})            
            x = sorted(embedding[:,component])
            max_value = np.max(x)
            min_value = np.min(x)
            rangeX = [min_value, max_value]
            width = (rangeX[1] - rangeX[0]) / bins
            df.to_csv(filename+'.tsv', sep = '\t', index = False)
            if df_colors is not None:
                df['color'] = df_colors['color'].values
            elif class_label_dict is not None:
                if self.axis == 1:
                    ids = self.columns
                else:
                    ids = self.df[self.row_id_column].values
                labels = [class_label_dict[instance_id] for instance_id in ids]            
                df['color'] = labels
            else:
                df['color'] = ['b']*len(df)
            #plot
            df = df.sort_values(str(component))
            if class_order is None:
                colors = set(df['color'].values)
            else:
                colors = class_order 
            fig, axes = plt.subplots(len(styles), sharex = True, figsize = (8, len(styles)*3))
            for i, style in enumerate(styles):
                if style == 'histogram':
                    axes[i].hist(
                        [df[df['color'] == c][str(component)].values for c in colors], 
                        bins = bins, 
                        width = width, 
                        range = rangeX, 
                        density = False,
                        histtype = 'barstacked',
                        label = colors
                        )
                    axes[i].set_title('Histogram')
                    axes[i].legend()
                elif style == 'swarm':
                    sns.swarmplot(x = df[str(component)], y=[""]*len(df), hue = df['color'], hue_order = colors, ax=axes[i])
                    #axes[i].get_legend().remove()
                    axes[i].set_title('Projection of %s %s' % (axis_label, component))
                elif style == 'distribution':
                    for c in colors:
                        values = df[df['color'] == c][str(component)].values
                        histogram = np.histogram(sorted(values), bins = bins, range = rangeX, density = False)
                        distribution = scipy.stats.rv_histogram(histogram)
                        fitted_pdf = distribution.pdf(x)
                        axes[i].plot(x, fitted_pdf)
                    axes[i].set_title('Fitted distribution')
                elif style == 'splines':
                    for c in colors:
                        values = df[df['color'] == c][str(component)].values
                        histogram = np.histogram(sorted(values), bins = bins, range = rangeX, density = False)
                        spl = interp1d(histogram[1][:-1], histogram[0], kind='cubic', fill_value = 0, bounds_error = False)
                        axes[i].plot(x, spl(x), label = c)
                    plt.legend()
                    axes[i].set_title('Cubic spline interpolation')
                else:
                    raise ValueError('Unknown style {}. Use any or all of [histogram, swarm, distribution, splines].')
            plt.suptitle('Projection of %s %s.' % (axis_label, component), y = 1)
            plt.tight_layout()            
            fig.savefig(filename)
        return ppg.FileGeneratingJob(filename, plot).depends_on(self.fit_transform()).depends_on(self.init_data_matrix()).depends_on(self.load())

    
    def plot_variance(self):
        """
        Plots explained variance for each component.
        """
        filename = os.path.join(self.result_dir, self.name + "variance.pdf")
        def plot():
            fig = plt.figure(figsize = (5, 5))
            explained_variance = self.model.explained_variance_
            x = np.array(range(len(explained_variance)), dtype = int)
            plt.bar(x, explained_variance)
            plt.title('Explained variance of the components')
            plt.gca().set_xlabels(['%i %s' % (ii, self.axis_label) for ii in range(explained_variance)])
            plt.tight_layout()
            fig.savefig(filename)
        return ppg.FileGeneratingJob(filename, plot).depends_on(self.fit_transform())

    def dump_correlation(self, genes, columns, dependencies = [], annotators = []):
        outfile = genes.result_dir / f"{self.name}_correlation.tsv"
        def dump():
            to_df = {'stable_id' : [], 'name' : []}
            components = {}
            for i in range(self.transformed_matrix.shape[1]):
                cname = '{}. component'.format(i+1)
                components[cname] = self.transformed_matrix[:,i]
                to_df["r {}".format(cname)] = []
            df = genes.df
            for row in df.iterrows():
                genes_expr = np.array([row[1][column] for column in columns])
                to_df['stable_id'].append(row[1]['stable_id'])
                to_df['name'].append(row[1]['name'])
                for cname in components:
                    r, p = scipy.stats.pearsonr(genes_expr, components[cname])
                    to_df["r {}".format(cname)].append(r)
            df_ret = pd.DataFrame(to_df)
            
            for column in columns:
                df_ret[column] = df[column]
            df_ret.to_csv(outfile, sep = '\t', index = False)
        for anno in annotators:
            dependencies.append(genes.add_annotator(anno))
        dependencies.append(genes.load())
        return ppg.FileGeneratingJob(outfile, dump).depends_on(dependencies).depends_on(self.fit_transform()).depends_on(self.init_data_matrix()).depends_on(self.load())
    
class MyPCA(PCA):    
    def __init__(self, name, genes_or_df_or_loading_function, columns, k = 3, class_label_dict = None, 
                 axis = 1, dependencies = [], annotators = [], row_id_column = 'stable_id', 
                 label_mangler = lambda x : x, scaler = None, imputer = None, missing_value = np.NaN):
        """
        This is a wrapper class for a principle component analysis, that takes a dataframe or a
        genomic region alternatively to calculate a PCA.
        @df_or_gene Dataframe or GenomicRegion containing d-dimensional data
        @columns Dataframe columns that contain the actual data
        @classlabel_column column that contains class labels
        """
        PCA.__init__(self, name, genes_or_df_or_loading_function, columns, k, class_label_dict, 
                     axis, dependencies, annotators, row_id_column, label_mangler, label_mangler, 
                     scaler, imputer, missing_value)
        self.axis = axis#1 if axis == 0 else 0
    
    def __calculate(self):
        def calc():
            matrix = self.get_scaled_data_matrix(self.axis).T#self.get_data_matrix(0)#self.matrix
            mean_vector = np.mean(matrix, axis = 1)
            scatter = np.zeros((matrix.shape[0],matrix.shape[0]))
            for i in range(matrix.shape[1]):
                x_minus_mean = matrix[:,i].reshape(matrix.shape[0],1) - mean_vector
                scatter += (x_minus_mean).dot(x_minus_mean.T)
            print("Scatter matrix")
            print(scatter)
            covariance = np.cov(matrix)
            print("covariance matrix")
            print(covariance)
            eigenvalues_scatter, eigenvectors_scatter = np.linalg.eig(scatter.T)
            eigenvalues_cov, eigenvectors_cov = np.linalg.eig(covariance)
            for i in range(len(eigenvalues_scatter)):
                eigvec_sc = eigenvectors_scatter[:,i].reshape(1,matrix.shape[0]).T
                eigvec_cov = eigenvectors_cov[:,i].reshape(1,matrix.shape[0]).T
                assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'
                print('Eigenvector {}: \n{}'.format(i+1, eigvec_cov))
                print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
                print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eigenvalues_scatter[i]))
                print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eigenvalues_cov[i]))
                print('Scaling factor: ', eigenvalues_scatter[i]/eigenvalues_cov[i])
                print(40 * '-')
            print('eigenvectors are identical')
            for i in range(len(eigenvalues_scatter)):
                eigv = eigenvectors_scatter[:,i].reshape(1,matrix.shape[0]).T
                np.testing.assert_array_almost_equal(scatter.dot(eigv), eigenvalues_scatter[i] * eigv, decimal=2, err_msg='', verbose=True)
            #now sort the eigenvectors
            for ev in eigenvectors_scatter:
                np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
            # Make a list of (eigenvalue, eigenvector) tuples
            list_of_tuples = [(np.abs(eigenvalues_cov[i]), eigenvectors_cov[:,i]) for i in range(len(eigenvalues_cov))]
            list_of_tuples.sort()
            list_of_tuples.reverse()
            return list_of_tuples
        return ppg.CachedAttributeLoadingJob('cache/'+self.name+'_calculate', self, 'sorted_eigenvector_tuples', calc).depends_on(self.init_data_matrix())

    def fit_transform(self, k = 3):
        def __transform():
            matrix = self.get_scaled_data_matrix(self.axis).T#self.get_data_matrix(0)#self.matrix
            weight_matrix = np.hstack([self.sorted_eigenvector_tuples[i][1].reshape(matrix.shape[0], 1) for i in xrange(k)])
            transformed = weight_matrix.T.dot(matrix).T
            assert transformed.shape == (matrix.shape[1], k), "The transformed matrix is not %s-dimensional." % k
            return transformed
        return ppg.CachedAttributeLoadingJob('cache/'+self.name + "_transformed_matrix", self, 'transformed_matrix', __transform).depends_on(self.init_data_matrix()).depends_on(self.__calculate())


class SklearnPCA(PCA):
    
    def __init__(self, name, genes_or_df_or_loading_function, columns, k = 3, class_label_dict_or_function = None, 
                 axis = 1, dependencies = [], annotators = [], whiten = True, 
                 row_id_column = 'stable_id', label_mangler = lambda x : x, scaler = None, imputer = None, missing_value = np.NaN):
        """
        This is PCA wrapper using the sklearn package, that takes a dataframe or a
        genomic region alternatively to calculate a PCA.
        @df_or_gene Dataframe or GenomicRegion containing d-dimensional data
        @columns Dataframe columns that contain the actual data
        @classlabel_column column that contains class labels
        """
        self.name = name
        self.whiten = whiten
        PCA.__init__(self, name, genes_or_df_or_loading_function, columns, k, class_label_dict_or_function, axis, dependencies, annotators, row_id_column, label_mangler, scaler, imputer, missing_value)

    def fit_transform(self):
        def __calc():
            sklearn_pca = sklearnPCA(
                                     n_components = self.k, 
                                     copy=True, 
                                     whiten=self.whiten, 
                                     svd_solver='auto', 
                                     tol=0.0, 
                                     iterated_power='auto', 
                                     random_state=None
                                     )
            matrix = self.get_scaled_data_matrix(self.axis)
            sklearn_pca = sklearn_pca.fit(matrix)
            explained_variance = sklearn_pca.explained_variance_
            sklearn_transform = -1*sklearn_pca.transform(matrix)
            return sklearn_transform
        return ppg.CachedAttributeLoadingJob('cache/'+self.name + "_transformed_matrix", self, 'transformed_matrix', __calc).depends_on(self.init_data_matrix())


class LDA(PCA):
    
    def __init__(self, name, genes_or_df_or_loading_function, columns, training_data, class_label_dict = None, 
                 axis = 1, dependencies = [], annotators = [], row_id_column = 'stable_id', 
                 label_mangler = lambda x : x, scaler = None, imputer = None, missing_value = np.NaN):
        """
        This is LDA wrapper using the sklearn package, that takes a dataframe or a
        genomic region alternatively to calculate a PCA for dimensionality reduction. 
        @df_or_gene Dataframe or GenomicRegion containing d-dimensional data
        @columns Dataframe columns that contain the actual data
        @classlabel_column column that contains class labels
        """
        self.name = name
        self.training_data = training_data
        PCA.__init__(self, name, genes_or_df_or_loading_function, columns, class_label_dict, axis, 
                     dependencies, annotators, row_id_column, label_mangler, transformation_function)
                 
    def fit_transform(self, k = 3):
        def __calc():
            lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
                                                                           solver = 'svd',
                                                                           shrinkage=None, 
                                                                           priors=None, 
                                                                           n_components=k, 
                                                                           store_covariance=False, 
                                                                           tol=0.0001
                                                                           )
            matrix = self.get_scaled_data_matrix(self.axis)
            lda.fit_transform(matrix, self.training_data)
            embedding = lda.transform(matrix)
            return embedding
        return ppg.CachedAttributeLoadingJob('cache/'+self.name + "_transformed_matrix", self, 'transformed_matrix', __calc).depends_on(self.init_data_matrix())


class TSNE(PCA):
                 
    def __init__(self, name, genes_or_df_or_loading_function, columns, k = 3, class_label_dict_or_function = None, 
                 axis = 1, dependencies = [], annotators = [], row_id_column = 'stable_id', 
                 label_mangler = lambda x : x, scaler = None, imputer = None, missing_value = np.NaN, preplexity = 30,
                 learning_rate = 1000.0, iterations = 1000, metric = 'euclidean', init = 'random',
                 random = None, method = 'barnes_hut'):
        """
        Wrapper for tSNE
        @param metric can be a string or callable, must either be a scipy.spatial.distance.pdist metric, from pairwise or precomputed.
        @param init might be 'random', 'pca' or numpy array of shape [n_instances, n_features]
        @param method can be 'exact' which does not scale but is more accurate and 'barnes_hut', which scales.
        """
        self.name = name
        self.perplexity = preplexity
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.metric = metric
        self.axis = axis
        self.random = random
        self.method = method
        self.axis_label = 'dimension'
        PCA.__init__(self, name, genes_or_df_or_loading_function, columns, k, class_label_dict_or_function, axis, 
                     dependencies, annotators, row_id_column, label_mangler, scaler, imputer, missing_value)
    
    
    def fit_transform(self):
        def __calc():
            tsne = manifold.TSNE(
                                 n_components = self.k, 
                                 perplexity = self.perplexity, 
                                 early_exaggeration=4.0, 
                                 learning_rate=self.learning_rate, 
                                 n_iter=self.iterations, 
                                 n_iter_without_progress=30, 
                                 min_grad_norm=1e-07, 
                                 metric=self.metric, 
                                 init='random', 
                                 verbose=0, 
                                 random_state=self.random, 
                                 method=self.method, 
                                 angle=0.5
                                 )
            matrix = self.get_scaled_data_matrix(self.axis)
            model = tsne.fit_transform(matrix)
            return model
        return ppg.CachedAttributeLoadingJob('cache/'+self.name + "_transformed_matrix", self, 'transformed_matrix', __calc).depends_on(self.init_data_matrix())
        

class MDS(PCA):
    
    def __init__(self, name, genes_or_df_or_loading_function, columns, k = 3, class_label_dict = None, 
                 axis = 1, dependencies = [], annotators = [], row_id_column = 'stable_id', 
                 label_mangler = lambda x : x, scaler = None, imputer = None, missing_value = np.NaN, n_init = 4, 
                 max_iterations = 300, n_jobs = 1, random = None, dissimilarity = 'euclidean'):
        """
        This is a wrapper for MDS.
        @param max_iterations max number of iterations
        @param dissimilarity can be 'euclidean' or 'precomputed'
        """
        self.name = name
        self.n_init = n_init
        self.max_iter = max_iterations
        self.n_jobs = n_jobs
        self.random = random
        self.dissimilarity = dissimilarity
        self.axis_label = 'dimension'
        PCA.__init__(self, name, genes_or_df_or_loading_function, columns, k, class_label_dict, axis, 
                     dependencies, annotators, row_id_column, label_mangler, label_mangler, scaler, imputer, missing_value)
        
        

    def fit_transform(self):
        def __calc():
            mds = manifold.MDS(
                               n_components = self.k, 
                               metric = True, 
                               n_init = self.n_init, 
                               max_iter = self.max_iter, 
                               verbose = 0, 
                               eps = 0.001, 
                               n_jobs = self.n_jobs, 
                               random_state = self.random, 
                               dissimilarity = self.dissimilarity
                               )
            matrix = self.get_scaled_data_matrix(self.axis)
            model = mds.fit_transform(matrix)
            return model
        return ppg.CachedAttributeLoadingJob('cache/'+self.name + "_transformed_matrix", self, 'transformed_matrix', __calc).depends_on(self.init_data_matrix())

class MlabPCA(PCA):
    
    def __init__(self, name, df_or_gene, columns, k = 3, class_label_dict = None,
                 instance_columns = True, dependencies = [], annotators = [], 
                 row_id_column = 'stable_id', label_mangler = lambda x : x, scaler = None, imputer = None, missing_value = np.NaN):
        """
        This is PCA wrapper using the matplotlib.mlab package, that takes a dataframe or a
        genomic region alternatively to calculate a PCA.
        This method will produce a different result than the others, since matplotlib scales the variables to unit variance.
        @df_or_gene Dataframe or GenomicRegion containing d-dimensional data
        @columns Dataframe columns that contain the actual data
        @classlabel_column column that contains class labels
        """
        self.name = name
        PCA.__init__(self, name, df_or_gene, columns, k, class_label_dict, instance_columns, dependencies, annotators, row_id_column, label_mangler, scaler, imputer, missing_value)
        
    def fit_transform(self, *args):
        """
        This method will produce a different result, since matplotlib scales the variables to unit variance.
        """
        def __calc():
            matrix = self.get_scaled_data_matrix(self.axis)
            mlab_pca = mlabPCA(matrix)
            return -1*mlab_pca.Y
        return ppg.CachedAttributeLoadingJob('cache/'+self.name + "_transformed_matrix", self, 'transformed_matrix', __calc).depends_on(self.init_data_matrix())            
