'''
Created on Mar 23, 2016

This module is used for multi-dimensional scaling.
@author: mernberger
'''
import matplotlib
#from matplotlib.mlab import PCA as mlabPCA
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
import scipy
import scipy.stats
from scipy.interpolate import interp1d
#import seaborn.apionly as sns
import itertools
from pathlib import Path

class DimensionalityReduction:
    
    def __init__(self, name, dimensions, invariants):
        """
        This is a wrapper for any dimensionality reduction approach. 
        @param name of the clustering strategy
        """
        self.__name = name
        self.__invariants = invariants
        self.__dimensions = dimensions
        self.__transformer = None

    @property
    def dimensions(self):
        return self.__dimensions

    @property
    def invariants(self):
        return self.__invariants

    @property
    def name(self):
        return self.__name
    
    def fit(self, df, **fit_parameter):
        self.__transformer = self.__transformer.fit(df, fit_parameter)
        self.__transformed_matrix = self.transformer.transform(df)
        

class SklearnPCA(DimensionalityReduction):
    
    def __init__(self, name, dimensions = 3, whiten = True):
        self.__whiten = whiten
        invariants = [
            dimensions,
            name,
            whiten
        ]
        super().__init__(name, dimensions, invariants)
        self.__transformer = sklearnPCA(
            n_components = self.dimensions, 
            copy=True, 
            whiten=self.__whiten, 
            svd_solver='auto', 
            tol=0.0, 
            iterated_power='auto', 
            random_state=None
            )
    
    def fit(self, df, **fit_parameter):
        self.__transformer = self.__transformer.fit(df, **fit_parameter)
        self.__transformed_matrix = self.__transformer.transform(df)
        self.__explained_variance = self.__transformer.explained_variance_
        self.__explained_variance_ratio = self.transformer.explained_variance_ratio_

    @property
    def transformer(self):
        return self.__transformer

    @property
    def reduced_matrix(self):
        return self.__transformed_matrix

    @property
    def explained_variance(self):
        return self.__explained_variance
    @property
    def explained_variance_ratio(self):
        return self.__explained_variance_ratio

class ClusteringMethod:
    def __init__(self, name, invariants):
        """
        This is a wrapper for any clustering approach. 
        @param name of the clustering strategy
        """
        self.__name = name
        self.__invariants = invariants

    @property
    def invariants(self):
        return self.__invariants

    @property
    def name(self):
        return self.__name

    def fit(self, df, **fit_parameter):
        """
        This method might alter the model and sets cluster, wich needs to be an additonal dataframe containing a single column
        with cluster names.
        """
        self.model = self.clustering.fit(df, fit_parameter)
        cluster_ids = self.model.labels_
        index = df.index.values
        self.clusters = pd.DataFrame({self.name: cluster_ids}, index=index)

    def predict(self, df_other, imputer, scaler, **fit_parameter):
        """
        This is used to apply a learned model to an additional data set.
        """
        df_imputed_other = df_other.transform(imputer.transform)
        df_imputed_other = df_imputed_other[df_imputed_other.max(axis=1) > 0]
        df_scaled_other = df_imputed_other.transform(scaler.transform)
        return self.model.predict(df_scaled_other.values, **fit_parameter)


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
        self.__no_of_clusters = no_of_clusters
        self.__n_init = n_init
        self.__max_iter = max_iter
        self.__random_state = random_state
        self.__n_jobs = n_jobs
        self.__init = init
        invariants = [
            self.__no_of_clusters,
            self.__init,
            self.__n_jobs,
            self.__random_state,
            self.__max_iter,
            self.__n_init,
        ]
        super().__init__(name, invariants)
        self.__clustering = sklearn.cluster.KMeans(
            n_clusters=self.__no_of_clusters,
            init="k-means++",
            n_init=self.__n_init,
            max_iter=self.__max_iter,
            algorithm="auto",
            tol=0.0001,
            precompute_distances="auto",
            verbose=0,
            random_state=self.__random_state,
            copy_x=True,
            n_jobs=self.__n_jobs,
        )

    @property
    def clustering(self):
        return self.__clustering

class ClassLabel(ClusteringMethod):
    def __init__(
        self,
        name,
        class_label_dict,
    ):
        """
        This is a wrapper for Kmeans
        @param name
        @param name class label dictionary with key = instance, value = class label
        """
        self.__class_label_dict = class_label_dict
        super().__init__(name, [class_label_dict])

    def fit(self, df, **fit_parameter):
        try:
            cluster_ids = [self.__class_label_dict[key] for key in df.index.values]
        except KeyError:
            print("Keys : ", self.__class_label_dict.keys())
            raise
        index = df.index.values
        self.clusters = pd.DataFrame({self.name: cluster_ids}, index=index)

    def predict(self, df_other, imputer, scaler, **fit_parameter):
        return NotImplementedError



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
        self.__no_of_clusters = no_of_clusters
        self.__affinity = affinity
        self.__linkage = linkage
        self.__connectivity = connectivity
        self.__compute_full_tree = compute_full_tree
        self.__memory = memory
        invariants = [
            self.__no_of_clusters,
            self.__affinity,
            self.__linkage,
            self.__connectivity,
            self.__compute_full_tree,
            self.__memory,
        ]
        super().__init__(name, invariants)
        self.__clustering = sklearn.cluster.AgglomerativeClustering(
            n_clusters=self.__no_of_clusters,
            affinity=self.__affinity,
            memory=self.__memory,
            connectivity=self.__connectivity,
            compute_full_tree=self.__compute_full_tree,
            linkage=self.__linkage,
        )

    @property
    def clustering(self):
        return self.__clustering

'''
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

            if self.__n_clusters <= 0:
                raise ValueError(
                    "n_clusters should be an integer greater than 0."
                    " %s was provided." % str(self.__n_clusters)
                )

            if self.__linkage == "ward" and self.__affinity != "euclidean":
                raise ValueError(
                    "%s was provided as affinity. Ward can only "
                    "work with euclidean distances." % (self.affinity,)
                )

            if self.__linkage not in sklearn.cluster.hierarchical._TREE_BUILDERS:
                raise ValueError(
                    "Unknown linkage type %s. "
                    "Valid options are %s"
                    % (self.linkage, sklearn.cluster.hierarchical._TREE_BUILDERS.keys())
                )
            tree_builder = sklearn.cluster.hierarchical._TREE_BUILDERS[self.linkage]

            connectivity = self.__connectivity
            if self.__connectivity is not None:
                if callable(self.connectivity):
                    connectivity = self.__connectivity(X)
                connectivity = check_array(
                    connectivity, accept_sparse=["csr", "coo", "lil"]
                )

            n_samples = len(X)
            compute_full_tree = self.__compute_full_tree
            if self.__connectivity is None:
                compute_full_tree = True
            if compute_full_tree == "auto":
                # Early stopping is likely to give a speed up only for
                # a large number of clusters. The actual threshold
                # implemented here is heuristic
                compute_full_tree = self.__n_clusters < max(100, 0.02 * n_samples)
            n_clusters = self.__n_clusters
            if compute_full_tree:
                n_clusters = None

            # Construct the tree
            kwargs = {"return_distance": True}
            if self.__linkage != "ward":
                kwargs["linkage"] = self.__linkage
                kwargs["affinity"] = self.__affinity
            self.__children_, self.__n_components_, self.__n_leaves_, parents, self.__distances = memory.cache(
                tree_builder
            )(
                X, connectivity, n_clusters=n_clusters, **kwargs
            )
            # Cut the tree
            if compute_full_tree:
                self.__labels_ = sklearn.cluster.hierarchical._hc_cut(
                    self.__n_clusters, self.__children_, self.__n_leaves_
                )
            else:
                labels = sklearn.cluster.hierarchical._hierarchical.hc_get_heads(
                    parents, copy=False
                )
                # copy to avoid holding a reference on the original array
                labels = np.copy(labels[:n_samples])
                # Reassign cluster numbers
                self.__labels_ = np.searchsorted(np.unique(labels), labels)
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
'''

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
        
        '''
# old stuff below
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
'''