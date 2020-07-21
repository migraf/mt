import pandas as pd
import numpy as np
import sklearn

from timeit import default_timer as timer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from analysis import *
import umap
import hdbscan
import scipy.stats as ss
import itertools
from pprint import pprint


def k_means_cluster(data=None, n_clusters=3, random_state=0, three_dimensional=True, max_iter=1000):
    """
    Attempts k means clustering for the selected data
    :param data:
    :type data:
    :return:
    :rtype:
    """
    # Perform k means clustering
    numpy_data = data.values
    # TODO give more configuration options
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter).fit(numpy_data)

    labels = kmeans.predict(numpy_data)
    plot_clusters(numpy_data, labels, three_d=three_dimensional)
    return labels


def gmm_cluster(data=None, n_models=3, three_dimensional=True, max_iter=1000):
    """
    Attempt clustering using Gaussian Mixture Models and different variations of
    Expectation Maximization Algorithms
    :param max_iter:
    :type max_iter:
    :param three_dimensional:
    :type three_dimensional:
    :param n_models:
    :type n_models:
    :param data:
    :type data:
    :return:
    :rtype:
    """
    numpy_data = data.values
    # TODO give more configuration options
    gmm = GaussianMixture(n_components=n_models,
                          covariance_type="full",
                          max_iter=max_iter).fit(numpy_data)
    # Get the number/names of the cluster
    labels = gmm.predict(numpy_data)
    # Create a figure with the data grouped by associated cluster
    plot_clusters(numpy_data, labels, three_d=three_dimensional)
    return labels


def vbgmm_cluster(data=None, n_models=3, three_dimensional=True, max_iter=1000):
    """
    Perform clustering using a variational bayesian gaussian mixture
    :param max_iter:
    :type max_iter:
    :param n_models:
    :type n_models:
    :param data:
    :type data:
    :param three_dimensional:
    :type three_dimensional:
    :return:
    :rtype:
    """
    numpy_data = data.values
    # TODO give more configuration options
    bgmm = BayesianGaussianMixture(n_components=n_models,
                                   covariance_type="full",
                                   weight_concentration_prior=0.001,
                                   max_iter=max_iter).fit(numpy_data)
    # Get the number/names of the cluster
    labels = bgmm.predict(numpy_data)
    # Create a figure with the data grouped by associated cluster
    plot_clusters(numpy_data, labels, three_d=three_dimensional)
    return labels


def umap_clustering(data, min_cluster_size=30):
    """
    Perform clustering using hdb scan while at the same time reducing the dimensions using umap
    :param data: data to cluster
    :type data:
    :param min_cluster_size: min samples that should make up a cluster
    :type min_cluster_size:
    :return:
    :rtype:
    """
    # TODO split into dimensionality reduction and hdbscan clustering
    clusterable_embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=50,
        random_state=42,
    ).fit_transform(data)
    plot_data = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=3,
        random_state=42,
    ).fit_transform(data)
    labels = hdbscan.HDBSCAN(
        min_samples=5,
        min_cluster_size=min_cluster_size,
    ).fit_predict(clusterable_embedding)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=plot_data[:, 0],
                               y=plot_data[:, 1],
                               z=plot_data[:, 2],
                               name=f"Umap HDB Scan clustering",
                               mode="markers",
                               marker_color=labels
                               )
                  )
    fig.show()
    return clusterable_embedding, labels
    # plot_clusters(data, labels)


def plot_clusters(data, clusters, three_d=True, reduction_algorithm="isomap"):
    """
    Visualize the data from clustering in either a two dimensional or three dimensional scatter plot,
    based on the clusters. To display the data dimensionality reduction techniques are applied.
    :param reduction_algorithm:
    :type reduction_algorithm:
    :param three_d:
    :type three_d:
    :param data: data used for calculating the clusters as well as performing dimensionality reduction
    :type data:
    :param clusters: the clusters found by the clustering algorithm
    :type clusters:
    :return: visualization for the clustering algorithm
    :rtype:
    """
    fig = go.Figure()
    # Create three dimensional scatter plot
    if three_d:
        # perform dimensionality reduction via pca
        # if reduction_algorithm == "pca":
        #     pca = PCA(n_components=3)
        #     pca.fit(data)
        #     mapped_components = pca.components_.dot(data.T)
        # elif reduction_algorithm == "tsne":
        #     pass
        # Create traces based on clusters
        mapped_components, _ = reduce_dims(data, reduction_algorithm=reduction_algorithm)
        for cluster in np.unique(clusters):
            fig.add_trace(go.Scatter3d(x=mapped_components[:, 0][clusters == cluster],
                                       y=mapped_components[:, 1][clusters == cluster],
                                       z=mapped_components[:, 2][clusters == cluster],
                                       name=f"Cluster {int(cluster) + 1}",
                                       mode="markers",
                                       marker=dict(
                                           size=4),
                                       )
                          )
    else:
        # create 2d scatter plot
        if reduction_algorithm == "pca":
            pca = PCA(n_components=2)
            pca.fit(data)
            mapped_components = pca.components_.dot(data.T)
        elif reduction_algorithm == "tsne":
            pass
        for cluster in np.unique(clusters):
            fig.add_trace(go.Scatter(x=mapped_components[:, clusters == cluster][0],
                                     y=mapped_components[:, clusters == cluster][1],
                                     name=f"Cluster {int(cluster) + 1}",
                                     mode="markers"
                                     )
                          )

    fig.show()


def perform_test(data, values, variable, type="numeric"):
    """
    Perform a statistical test of difference between the values,
    chooses which test to perform based on the selected variable
    :param data: The underlying data
    :type data: DataFrame
    :param values: the values found for each cluster
    :type values: list of dataframes
    :param variable:
    :type variable: str
    :return:
    :rtype:
    """
    # TODO switch this to be based on underlying data -> removing dummy variables
    if type == "numeric":
        # Check if variable is normally distributed
        ks, shapiro_p = ss.shapiro(data[variable][data[variable].notnull()])
        # If normally distributed use anova
        if shapiro_p > 0.05:
            an_h, an_p = ss.f_oneway(*values)
            return an_h, an_p
        else:
            try:
                kruskal_h, kruskal_p = ss.kruskal(*values)
                return kruskal_h, kruskal_p
            except BaseException as error:
                print(error)
                return np.inf, np.inf
    if type == "categorical":
        # TODO calculate contingency matrix and perform chi2 test
        pass


def compare_clusters(data, cluster_dfs, n=20):
    """
    Compare the details of two patient clusters
    :param data: dataframes the cluster dfs are extracted from
    :param cluster_dfs: list of dataframes containing clustered samples
    :param n: number of most different results to extract
    :return:
    :rtype:
    """
    columns = list(cluster_dfs[0].columns)
    general_anova_results = []

    # TODO handle categorical/dependant variables
    # Get indices of all combination of clusters
    combination_indices = list((i, j) for ((i, _), (j, _)) in itertools.combinations(enumerate(cluster_dfs), 2))
    between_cluster_results = {x: {} for x in range(len(cluster_dfs))}
    # TODO implement between item comparisons more efficiently
    for combination in combination_indices:
        between_cluster_results[combination[0]][combination[1]] = []
        if combination[0] not in between_cluster_results[combination[1]].keys():
            between_cluster_results[combination[1]][combination[0]] = []

    print(between_cluster_results)
    for variable in columns:
        # Extract values for each variable
        samples = []
        for df in cluster_dfs:
            if len(list(df[variable])) > 1:
                samples.append(list(df[variable]))
        # General comparison for the current variable
        h, p = perform_test(data, samples, variable)
        try:
            general_anova_results.append({"variable": variable, "p-Value": p})
        except ValueError:
            print("Exception occured")
            general_anova_results.append({"variable": variable, "p-Value": np.inf})

        # In between comparison between all clusters for this variable
        for combination in combination_indices:
            df1 = cluster_dfs[combination[0]]
            df2 = cluster_dfs[combination[1]]
            bet_h, bet_p = perform_test(data,
                                        [cluster_dfs[combination[0]][variable].values,
                                         cluster_dfs[combination[1]][variable].values],
                                        variable)
            between_cluster_results[combination[0]][combination[1]].append({"variable": variable, "p-Value": bet_p})
            between_cluster_results[combination[1]][combination[0]].append({"variable": variable, "p-Value": bet_p})

    # Sort general results and take
    general_anova_results = sorted(general_anova_results, key=lambda x: x["p-Value"])
    results = {"overall_results": general_anova_results[:n]}
    for res in between_cluster_results.keys():
        for cl in between_cluster_results[res].keys():
            between_cluster_results[res][cl] = sorted(between_cluster_results[res][cl], key=lambda x: x["p-Value"])[:n]
    results["between_cluster_results"] =  between_cluster_results

    return results


def extract_clusters(df, clusters, selected_clusters=None, save=True):
    """
    Extract the patients belonging to one or all clusters found by a clustering algorithm, and exports them into a new
    excel sheet
    :param df: base data frame
    :type df:
    :param clusters: clusters found by clustering algorithm
    :type clusters:
    :param selected_clusters: which clusters to export
    :return:
    :rtype:
    """
    # TODO extend to save data to other formats
    cluster_dfs = []
    if selected_clusters is not None:
        for cluster in selected_clusters:
            selector = (clusters == (cluster - 1))
            cluster_dfs.append(df[selector])
        for i in range(len(selected_clusters)):
            cluster_dfs[i].to_excel(f"cluster_{selected_clusters[i]}.xlsx")
    else:
        for cluster in np.unique(clusters):
            selector = (clusters == (cluster - 1))
            cluster_dfs.append(df[selector])

    return cluster_dfs


if __name__ == '__main__':
    df_sars = load_data("C:\\hypothesis\\repositories\\server\\walzLabBackend\\notebook\\15052020SARS-CoV-2_final.xlsx")

    excluded_categorical_columns = ['Patienten-ID', ]
    excluded_numerical_columns = []

    num_columns, cat_columns = find_variables(df_sars,
                                              excluded_categorical_columns,
                                              excluded_numerical_columns,
                                              min_available=20,
                                              display=True
                                              )
    cluster_df = create_cluster_data(df_sars, num_columns, cat_columns)
    # k_means_cluster(cluster_df)
    # gmm_cluster(data=dummy_df)
    clusters = vbgmm_cluster(data=cluster_df, n_models=3)
    # umap_clustering(cluster_df, 7)
    cluster_dfs = extract_clusters(cluster_df, clusters)
    pprint(compare_clusters(cluster_df, cluster_dfs))
