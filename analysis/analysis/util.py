import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from analysis import *
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype
import plotly.express as px


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


def create_training_data(data, target_var=None, excluded_variables=[], test_train_split=True, test_size=0.2,
                         min_available=0.5, imputer="simple", n_neighbors=5):
    """
    Split the given data frame into train/test/val split and process the data, to fit to the selected algorithm type
    :param na_strategy: how to handle nan values, in target column either drop or fill
    :type na_strategy:
    :param data: intial data frame
    :type data:
    :return:
    :rtype:
    """

    # # Remove target from data initially
    # cat_cols_copy = cat_cols.copy()
    # num_cols_copy = num_cols.copy()
    # if target_col in cat_cols_copy:
    #     if target_col:
    #         target = data[target_col].copy()
    #         cat_cols_copy.remove(target_col)
    # else:
    #     if target_col:
    #         num_cols_copy.remove(target_col)
    #         target = data[target_col].copy()
    #         target = target.astype(float)
    train_data = data.copy()
    if target_var:
        target = data[target_var].copy()
        train_data = train_data.drop(columns=[target_var])

    # remove excluded columns
    train_data = train_data.drop(columns=excluded_variables)

    # remove columns with less than the given percentage of available values
    train_data = train_data[train_data.columns[train_data.isnull().mean() < min_available]]

    # remove columns with more than 75% unique values in categorical columns
    high_cardinality_variables = []
    for col in train_data.columns:
        if not pd.api.types.is_numeric_dtype(train_data[col]):
            n_unique = len(train_data[col].unique())
            if float(n_unique) / len(train_data[col][train_data[col].notnull()]) >= 0.75:
                high_cardinality_variables.append(col)

    print(high_cardinality_variables)
    train_data = train_data.drop(high_cardinality_variables, axis=1)

    num_vars, cat_vars = find_variables(train_data, display=False)

    dummy_df = encode_categorical_variables(train_data, cat_vars)
    # Standardize numerical variables
    num_df = train_data[num_vars]
    cols = num_df.columns
    # num_df = fill_na_with_median(num_df)
    scaler = StandardScaler()

    if imputer == "knn":
        imp = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors)
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy="median")
    num_df = imp.fit_transform(num_df)
    num_df = scaler.fit_transform(num_df)
    num_df = pd.DataFrame(columns=cols, data=num_df)

    train_data = pd.concat([num_df, dummy_df], axis=1)
    train_data = train_data.dropna(how="all")
    if target_var:
        # Keep only samples where the target variable is present
        train_data = train_data[target.notnull()]
        target = target[target.notnull()]
    train_data = train_data.fillna(method="pad", axis="columns")

    if not target_var:
        return train_data
    elif test_train_split:
        x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size=test_size)
        return x_train, x_test, y_train, y_test
    else:
        return train_data, target


def find_variables(data, display=False):
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
        if pd.api.types.is_numeric_dtype(data[col]):
            numerical_columns.append(col)
        else:
            categorical_columns.append(col)

    if display:
        print("Included numerical variables:\n\n", numerical_columns)
        print("\n")
        print("Included categorical variables:\n\n", categorical_columns)
    return numerical_columns, categorical_columns
    #
    # for col in data.columns:
    #     if data[col].dtype in ["float64", "Int64"]:
    #         if col not in excluded_numeric:
    #             n_available = np.sum(data[col].notnull())
    #             # dont include variables with less than the desired number of entries
    #             if n_available >= min_available:
    #                 numerical_columns.append(col)
    #     elif data[col].dtype in ["object", "string", "category"]:
    #         if col not in excluded_categorical:
    #             n_available = np.sum(data[col].notnull())
    #             # dont include variables with less than the desired number of entries
    #             if n_available >= min_available:
    #                 categorical_columns.append(col)


def detect_prediction_type(data, target):
    """
    Examines the values in the target column and picks a model sub type appropriate for this
    Parameters
    ----------
    data : pandas Dataframe containing the data
    target : name of the target column

    Returns
    -------
    String containing the model sub type

    """
    if is_numeric_dtype(data[target]):
        if len(data[target][data[target].notnull()].unique()) == 2:
            return "binary"
        else:
            return "regression"
    else:
        if len(data[target].unique()) == 2:
            return "binary"
        else:
            return "multi-class"


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


def reduce_dims(data, dims=3, reduction_algorithm="tsne", prepare_data=False, display=False,
                max_iter=1000, tsne_perp=10, tsne_lr=50, isomap_neighbors=5):
    """
    Reduce the dimensionality of the data to the selected number of dimensions
    :param data:
    :type data:
    :param dims:
    :type dims:
    :return:
    :rtype:
    """

    if prepare_data:
        _data = create_training_data(data)
    else:
        _data = data.copy()
    # Dimensionality reduction using PCA
    if reduction_algorithm == "pca":
        embedded = PCA(n_components=dims).fit_transform(_data)
    # Dimensionality Reduction using t-SNE
    elif reduction_algorithm == "tsne":
        # If the number of dimensions is higher than 50 perform an initial PCA
        if data.shape[1] > 50:
            pca = PCA(n_components=50)
            mapped_components = pca.fit_transform(_data)

            embedded = TSNE(n_components=dims, perplexity=tsne_perp, learning_rate=tsne_lr, n_iter=max_iter). \
                fit_transform(mapped_components)
        else:
            embedded = TSNE(n_components=dims, perplexity=tsne_perp, learning_rate=tsne_lr).fit_transform(data)
    # Dimensionality reduction using Isomap
    elif reduction_algorithm == "isomap":
        isomap = Isomap(n_components=dims, n_neighbors=isomap_neighbors).fit(data)
        embedded = isomap.transform(data)

    if display:
        if dims == 2:
            fig = px.scatter(x=list(embedded[:, 0]),
                             y=list([embedded[:, 1]]))

            fig.update_layout(showlegend=False)
            fig.show()
            # fig.write_image(f"{reduction_algorithm}_{dims}d.png", height=1000, width=1000, scale=2)
        elif dims == 3:
            fig = px.scatter_3d(x=list(embedded[:, 0]),
                                y=list(embedded[:, 1]),
                                z=list(embedded[:, 2]),
                            )
            fig.update_layout(showlegend=False)
            fig.update_traces(marker=dict(size=8))
            fig.show()
            # fig.write_image(f"{reduction_algorithm}_{dims}d.png", height=1000, width=1000, scale=2)
        else:
            raise ValueError("Data can only be displayed in two or three dimensions")
    return embedded


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
    grid_search = GridSearchCV(estimator, param_grid=param_grid)
    grid_search.fit(data, target)

    return grid_search.best_estimator_, grid_search.cv_results_, grid_search.best_params_


def create_data_summary(df, dict, filenname="data_overview.csv"):
    overview = pd.DataFrame()
    overview["types"] = df.dtypes
    overview["num_nulls"] = np.sum(df.isnull())
    mapping = pd.Series(df.columns).map(dict)
    overview["description"] = list(mapping)
    overview.to_csv(filenname)




if __name__ == '__main__':
    df = load_data("walz_data.csv")
    training_data = create_training_data(df)

    algs = ["tsne", "pca", "isomap"]

    for alg in algs:
        for dim in [2,3]:
            reduce_dims(training_data, dims=dim, reduction_algorithm=alg, tsne_lr=10, tsne_perp=50, max_iter= 10000,
                        display=True)
