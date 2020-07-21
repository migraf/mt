import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
# from analysis import *
from sklearn.preprocessing import StandardScaler


def encode_categorical_variables(df, cat_cols):
    """
    Takes a dataframe and a list of categorical columns, and creates dummy variables for the categorical
    variables leaving numeric variables
    :param df:
    :type df:
    :param cat_cols:
    :type cat_cols:
    :return:
    :rtype:
    """
    dummy_df = pd.DataFrame()
    for col in cat_cols:
        var_dummy = pd.get_dummies(df[col], prefix=col, dtype=float, dummy_na=True)
        dummy_df = pd.concat([dummy_df, var_dummy], axis=1)
    return dummy_df


def create_cluster_data(df, num_cols, cat_cols):
    """
    Create data for clustering algorithms based on selected categorical and numerical columns
    :param df:
    :type df:
    :param num_cols:
    :type num_cols:
    :param cat_cols:
    :type cat_cols:
    :return:
    :rtype:
    """
    dummy_df = encode_categorical_variables(df, cat_cols)
    num_df = df[num_cols]
    data = pd.concat([num_df, dummy_df], axis=1)
    # data = fill_na_with_mean(data)
    data = data.dropna(how="all")
    # TODO handle still existing nans better than this
    data = data.fillna(0)
    return data


def revert_to_categorical(df, cat_cols):
    """
    Changes the onehot encoded columns in the given data frame back to a single categorical column based on
    prefix
    :param df:
    :type df:
    :return:
    :rtype:
    """
    pass


def create_training_data(df, num_cols, cat_cols, target_col, test_train_split=True):
    """
    Split the given data frame into train/test/val split and process the data, to fit to the selected algorithm type
    :param na_strategy: how to handle nan values, in target column either drop or fill
    :type na_strategy:
    :param df: intial data frame
    :type df:
    :return:
    :rtype:
    """

    # Remove target from data initially
    cat_cols_copy = cat_cols.copy()
    num_cols_copy = num_cols.copy()
    if target_col in cat_cols_copy:
        cat_cols_copy.remove(target_col)
        target = df[target_col].copy()
    else:
        num_cols_copy.remove(target_col)
        target = df[target_col].copy()
        target = target.astype(float)

    dummy_df = encode_categorical_variables(df, cat_cols_copy)
    # Standardize numerical variables
    num_df = df[num_cols_copy]
    cols = num_df.columns
    num_df = fill_na_with_median(num_df)
    scaler = StandardScaler()

    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    num_df = imp.fit_transform(num_df)
    num_df = scaler.fit_transform(num_df)
    num_df = pd.DataFrame(columns=cols, data=num_df)

    data = pd.concat([num_df, dummy_df], axis=1)
    data = data.dropna(how="all")
    # TODO handle still existing nans better than this

    data = data[target.notnull()]
    target = target[target.notnull()]
    for col in data.columns:
        data[col] = data[col].astype("float")
    data = data.fillna(method="pad", axis="columns")
    # data = data.fillna(data.mean())
    if test_train_split:
        x_train, x_test, y_train, y_test = train_test_split(data, target)
        return x_train, x_test, y_train, y_test
    else:
        return data, target


def fill_na_with_median(df):
    """
    Fills the NA values in a dataframe with the column means, per column
    :param df:
    :type df:
    :return:
    :rtype:
    """
    cleaned_df = df.copy()
    for col in cleaned_df.columns:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    return cleaned_df


def reduce_dims(data, dims=3, reduction_algorithm="tsne"):
    """
    Reduce the dimensionality of the data to the selected number of dimensions
    :param data:
    :type data:
    :param dims:
    :type dims:
    :return:
    :rtype:
    """
    # Dimensionality reduction using PCA
    if reduction_algorithm == "pca":
        pca = PCA(n_components=dims).fit(data)
        mapped_components = pca.predict(data)
        embedding = pca.components_
        return mapped_components, embedding
    # Dimensionality Reduction using t-SNE
    elif reduction_algorithm == "tsne":
        # If the number is dimensions is higher than 50 perform an initial PCA
        if data.shape[1] > 50:
            pca = PCA(n_components=50)
            mapped_components = pca.fit_transform(data)
            tsne = TSNE(n_components=dims, perplexity=10, learning_rate=10, n_iter=10000).fit(mapped_components)
            embedded = TSNE(n_components=dims, perplexity=10, learning_rate=10, n_iter=10000).fit_transform(
                mapped_components)
        else:
            tsne = TSNE(n_components=dims).fit(data)
            embedded = TSNE(n_components=dims).fit_transform(data)
        embedding = tsne.embedding_
        return embedded, embedding
    # Dimensionality reduction using Isomap
    elif reduction_algorithm == "isomap":
        isomap = Isomap(n_components=dims).fit(data)
        isomap_embedded = isomap.transform(data)
        embedding = isomap.embedding_
        return isomap_embedded, embedding


def cross_validation_tuning(estimator, param_grid, data, target):
    """
    Perform cross validation for a given estimator, attempting hyper parameter using exhaustive grid search on the given
    options
    :param estimator:
    :type estimator:
    :param param_grid:
    :type param_grid:
    :return:
    :rtype:
    """
    # TODO implement random seach cross validation as an option
    grid_search = GridSearchCV(estimator, param_grid=param_grid, n_jobs=8)
    grid_search.fit(data, target)

    return grid_search.best_estimator_, grid_search.cv_results_, grid_search.best_params_


def create_data_summary(df, dict, filenname="data_overview.csv"):
    overview = pd.DataFrame()
    overview["types"] = df.dtypes
    overview["num_nulls"] = np.sum(df.isnull())
    mapping = pd.Series(df.columns).map(dict)
    overview["description"] = list(mapping)
    overview.to_csv(filenname)


def create_scatter_matrix(df, selected_cols=None):
    """
    Create a scatter matrix of all the selected variables against each other
    :param df:
    :type df:
    :param selected_cols:
    :type selected_cols:
    :return:
    :rtype:
    """
    # All columns containing antibody information??
    cols = ["VII.1A",
            "VII.1B",
            "VII.1C",
            "VII.2A",
            "VII.2B",
            "VII.2C",
            "VII.3A",
            "VII.3B",
            "VII.3C", ]
    # "IX.1A",
    # "IX.2A",
    # "IX.1B",
    # "IX.2B",
    # "IX.1C",
    # "IX.2C"]
    if selected_cols:
        data = df[selected_cols]
    else:
        data = df[cols]

    # Create figure
    fig = plt.figure(figsize=(20, 20))
    plots = scatter_matrix(data, figsize=(20, 20))
    fig.axes.append(plots)
    fig.show()


def find_variables(data, excluded_categorical, excluded_numeric, min_available=20, display=False):
    """
    Find and output variables to include in the data based on different criteria,
    sort into categorical variables
    :param data: base dataframe
    :type data:
    :param excluded_categorical: categorical varaibles to be excluded
    :type excluded_categorical:
    :param excluded_numeric: numeric variables to be included
    :type excluded_numeric:
    :param min_available: min number of available samples
    :type min_available:
    :return: list of numeric variables, list of categorical variables
    :rtype:
    """
    numerical_columns = []
    categorical_columns = []
    for col in data.columns:
        if data[col].dtype in ["float64", "Int64"]:
            if col not in excluded_numeric:
                n_available = np.sum(data[col].notnull())
                # dont include variables with less than the desired number of entries
                if n_available >= min_available:
                    numerical_columns.append(col)
        elif data[col].dtype in ["object", "string", "category"]:
            if col not in excluded_categorical:
                n_available = np.sum(data[col].notnull())
                # dont include variables with less than the desired number of entries
                if n_available >= min_available:
                    categorical_columns.append(col)
    if display:
        print("Included numerical variables:\n\n", numerical_columns)
        print("\n")
        print("Included categorical variables:\n\n", categorical_columns)
    return numerical_columns, categorical_columns


if __name__ == '__main__':
    df, dict = load_data(
        "/server/walzLabBackend/flaskr/user_data/Datentabelle_CoVid19_SARS.xlsx",
        two_sheets=True)
    # print(df.info())
    # dummy_df = encode_categorical_variables(df, categorical_columns)
    # print(create_training_data(df, numerical_columns, categorical_columns, "III.9"))
    # train_x, test_x, train_y, test_y = create_training_data(df, numerical_columns, categorical_columns, "VII.3C")
    # print(train_x[train_x.isnull()])
    # print(reduce_dims(train_x, reduction_algorithm="isomap"))
    create_scatter_matrix(df)
