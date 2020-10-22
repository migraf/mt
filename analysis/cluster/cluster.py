import pandas as pd
import numpy as np
from util import create_training_data, reduce_dims, load_data
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import plotly.graph_objects as go
import hdbscan
import scipy.stats as ss
import itertools
from pprint import pprint


def k_means_cluster(data, n_clusters, excluded_variables=[], prepare_data=True, random_state=0, max_iter=1000,
                    display=True, dims=3, reduction_algorithm="isomap", evaluate_clusters=False, dim_red_params=None):
    """
    Attempts k means clustering for the selected data
    :param data:
    :type data:
    :return:
    :rtype:
    """
    # Perform k means clustering

    if prepare_data:
        data = create_training_data(data, excluded_variables=excluded_variables, test_train_split=False)
        numpy_data = data.values
    else:
        numpy_data = data.values
    # TODO give more configuration options
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter).fit(numpy_data)
    labels = kmeans.predict(numpy_data)
    if display:
        plot_clusters(numpy_data, labels, dims=dims, reduction_algorithm=reduction_algorithm)
    if evaluate_clusters:
        cluster_dfs = extract_clusters(data, labels)
        comparison_result = compare_clusters(data, cluster_dfs)
        pprint(comparison_result)

    return labels


def gmm_cluster(data, n_clusters, covariance_type="full", excluded_variables=[], prepare_data=True, random_state=0,
                max_iter=1000, display=True, dims=3, reduction_algorithm="isomap", evaluate_clusters=False,
                dim_red_params=None):
    """
    Attempt clustering using Gaussian Mixture Models and different variations of
    Expectation Maximization Algorithms
    :param max_iter:
    :type max_iter:
    :param three_dimensional:
    :type three_dimensional:
    :param n_clusters:
    :type n_clusters:
    :param data:
    :type data:
    :return:
    :rtype:
    """
    if prepare_data:
        data = create_training_data(data, excluded_variables=excluded_variables, test_train_split=False)
        numpy_data = data.values
    else:
        numpy_data = data.values
    gmm = GaussianMixture(n_components=n_clusters,
                          covariance_type=covariance_type,
                          max_iter=max_iter).fit(numpy_data)
    # Get the number/names of the cluster
    labels = gmm.predict(numpy_data)
    # Create a figure with the data grouped by associated cluster
    if display:
        plot_clusters(numpy_data, labels, dims=dims, reduction_algorithm=reduction_algorithm)
    if evaluate_clusters:
        cluster_dfs = extract_clusters(data, labels)
        comparison_result = compare_clusters(data, cluster_dfs)
        pprint(comparison_result)

    return labels


def vbgmm_cluster(data, n_clusters, covariance_type="full", weight_concentration_prior=None, excluded_variables=[],
                  prepare_data=True, random_state=0, max_iter=1000, display=True, dims=3, reduction_algorithm="isomap",
                  evaluate_clusters=False, dim_red_params=None):
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
    if prepare_data:
        data = create_training_data(data, excluded_variables=excluded_variables, test_train_split=False)
        numpy_data = data.values
    else:
        numpy_data = data.values
    # TODO give more configuration options
    bgmm = BayesianGaussianMixture(n_components=n_clusters,
                                   covariance_type=covariance_type,
                                   weight_concentration_prior=weight_concentration_prior,
                                   max_iter=max_iter).fit(numpy_data)
    # Get the number/names of the cluster
    labels = bgmm.predict(numpy_data)

    if display:
        plot_clusters(numpy_data, labels, dims=dims, reduction_algorithm=reduction_algorithm)
    if evaluate_clusters:
        cluster_dfs = extract_clusters(data, labels)
        comparison_result = compare_clusters(data, cluster_dfs)
        pprint(comparison_result)

    return labels


def hdbscan_clustering(data, min_cluster_size, min_samples=None, excluded_variables=[],
                  prepare_data=True, random_state=0, max_iter=1000, display=True, dims=3, reduction_algorithm="isomap",
                  evaluate_clusters=False, dim_red_params=None):
    """
    Perform clustering using hdb scan while at the same time reducing the dimensions using umap
    :param data: data to cluster
    :type data:
    :param min_cluster_size: min samples that should make up a cluster
    :type min_cluster_size:
    :return:
    :rtype:
    """

    if prepare_data:
        data = create_training_data(data, excluded_variables=excluded_variables, test_train_split=False)
        numpy_data = data.values
    else:
        numpy_data = data.values

    labels = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    ).fit_predict(numpy_data)
    print(labels)

    if display:
        plot_clusters(numpy_data, labels, dims=dims, reduction_algorithm=reduction_algorithm)
    if evaluate_clusters:
        cluster_dfs = extract_clusters(data, labels)
        comparison_result = compare_clusters(data, cluster_dfs)
        pprint(comparison_result)

    return labels
    # plot_clusters(data, labels)


def plot_clusters(data, clusters, dims=3, reduction_algorithm="isomap"):
    """
    Visualize the data from clustering in either a two dimensional or three dimensional scatter plot,
    based on the clusters. To display the data dimensionality reduction techniques are applied.
    :param reduction_algorithm:
    :type reduction_algorithm:
    :param dims:
    :type dims:
    :param data: data used for calculating the clusters as well as performing dimensionality reduction
    :type data:
    :param clusters: the clusters found by the clustering algorithm
    :type clusters:
    :return: visualization for the clustering algorithm
    :rtype:
    """
    fig = go.Figure()
    # Create three dimensional scatter plot
    if dims == 3:
        mapped_components = reduce_dims(data, dims=dims, reduction_algorithm=reduction_algorithm)
        for cluster in np.unique(clusters):
            fig.add_trace(go.Scatter3d(x=mapped_components[:, 0][clusters == cluster],
                                       y=mapped_components[:, 1][clusters == cluster],
                                       z=mapped_components[:, 2][clusters == cluster],
                                       name=f"Cluster {int(cluster)}",
                                       mode="markers",
                                       marker=dict(
                                           size=4),
                                       )
                          )
    elif dims == 2:
        mapped_components = reduce_dims(data, dims=dims, reduction_algorithm=reduction_algorithm)
        for cluster in np.unique(clusters):
            fig.add_trace(go.Scatter(x=mapped_components[:, 0][clusters == cluster],
                                     y=mapped_components[:, 1][clusters == cluster],
                                     name=f"Cluster {int(cluster)}",
                                     mode="markers"
                                     )
                          )
    else:
        raise ValueError(f"dims = {dims}, but visualization is only possible for 2 or 3 dimensions")

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
        try:

            cont_table = pd.DataFrame(index=data[variable].unique())
            for ind in range(len(values)):
                cont_table[ind] = pd.Series(values[ind]).value_counts()
            cont_table = cont_table.fillna(0)
            chi2, p, dof, expected = ss.chi2_contingency(cont_table)
            return chi2, p
        except ValueError as e:
            return np.inf, np.inf



def compare_clusters(data, cluster_dfs, n=20):
    """
    Compare the details of  patient clusters
    :param data: dataframes the cluster dfs are extracted from
    :param cluster_dfs: list of dataframes containing clustered samples
    :param n: number of most different results to extract
    :return:
    :rtype:
    """
    columns = list(cluster_dfs[0].columns)
    general_anova_results = []

    # Get indices of all combination of clusters
    combination_indices = list((i, j) for ((i, _), (j, _)) in itertools.combinations(enumerate(cluster_dfs), 2))
    between_cluster_results = {x: {} for x in range(len(cluster_dfs))}
    # TODO implement between item comparisons more efficiently
    for combination in combination_indices:
        between_cluster_results[combination[0]][combination[1]] = []
        if combination[0] not in between_cluster_results[combination[1]].keys():
            between_cluster_results[combination[1]][combination[0]] = []

    for variable in columns:
        var_type = "numeric" if pd.api.types.is_numeric_dtype(data[variable]) else "categorical"
        # Extract values for each variable
        samples = []
        for df in cluster_dfs:
            if len(list(df[variable])) > 1:
                samples.append(list(df[variable]))
        # General comparison for the current variable
        h, p = perform_test(data, samples, variable, type=var_type)
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
                                        variable,
                                        type=var_type)
            between_cluster_results[combination[0]][combination[1]].append({"variable": variable, "p-Value": bet_p})
            between_cluster_results[combination[1]][combination[0]].append({"variable": variable, "p-Value": bet_p})

    # Sort general results and take
    general_anova_results = sorted(general_anova_results, key=lambda x: x["p-Value"])
    results = {"overall_results": general_anova_results[:n]}
    for res in between_cluster_results.keys():
        for cl in between_cluster_results[res].keys():
            between_cluster_results[res][cl] = sorted(between_cluster_results[res][cl], key=lambda x: x["p-Value"])[:n]
    results["between_cluster_results"] = between_cluster_results

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
    df_sars = load_data("../../datasets/walz_data.csv", na_values=["<NA>"])

    # clusters = k_means_cluster(df_sars, n_clusters=3, prepare_data=True, dims=2, evaluate_clusters=False)
    # print(clusters)
    # clusters = gmm_cluster(df_sars, n_clusters=3, prepare_data=True, dims=2, evaluate_clusters=False)
    # clusters = vbgmm_cluster(df_sars, n_clusters=3, prepare_data=True, dims=2, evaluate_clusters=False)
    clusters = hdbscan_clustering(df_sars, min_cluster_size=3, min_samples=1, dims=2, evaluate_clusters=False)
    # cluster_dfs = extract_clusters(cluster_df, clusters)
    # pprint(compare_clusters(cluster_df, cluster_dfs))
