import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, LogisticRegression
from sklearn.preprocessing import normalize
# from process_data import load_data
# from util import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# # from analysis import create_training_data, find_variables
# from .util import create_training_data, find_variables
# import shap
#
# from .process_data import load_data

from analysis import *

from util import detect_prediction_type, cross_validation_tuning


def linear_model(data, target, excluded_variables=[], prediction_type=None, l1_ratio=0.2, max_iter=1000,
                 cv=True, cv_params=None, display=True, shap=True, prepare_data=True):
    if prediction_type:
        model_subtype = prediction_type
    else:
        model_subtype = detect_prediction_type(data, target)

    # Create fitting predictor
    from analysis.analysis import create_training_data
    if prepare_data:
        x_train, x_test, y_train, y_test = create_training_data(data, target, excluded_variables)
    else:
        x_train, x_test = data[0], data[1]
        y_train, y_test = target[0], target[1]
    print(f"Creating a linear {model_subtype} model")
    if model_subtype in ["binary", "multi-class"]:
        pred = LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, max_iter=max_iter, solver="saga")
        y_train = y_train.astype("str")
        y_test = y_test.astype("str")

        # Kernel function for shap value prediction
        f = lambda x: pred.predict_proba(x)[:,1]
    else:
        pred = ElasticNet(l1_ratio=l1_ratio, max_iter=max_iter)
        y_train = y_train.astype("float")
        y_test = y_test.astype("float")

    if cv:
        # Perform cross validation hyper parameter tuning
        if not cv_params:
            cv_params = {
                "l1_ratio": [0, 0.2, 0.5, 0.75, 1],
                "max_iter": [100, 1000, 10000]
            }
        pred, cv_results, param_results = cross_validation_tuning(pred, cv_params, x_train, y_train)
        print(param_results)
    else:
        pred.fit(x_train, y_train)

    if display:
        display_model_performance(pred, model_subtype, x_test, y_test, target)
        if model_subtype != "regression":
            shap_values = display_feature_importances(pred.predict_proba, x_train, x_test, return_shap=shap)
        else:
            shap_values = display_feature_importances(pred.predict, x_train, x_test, return_shap=shap)
    else:
        print(f"Score: {pred.score(x_test, y_test)}")
        # TODO print additional information
    if shap:
        return pred, shap_values
    return pred




def linear_regression(data, num_cols, cat_cols, target):
    """
    Fit a basic linear regression model to the data
    :return:
    :rtype:
    """
    x_train, x_test, y_train, y_test = create_training_data(data, num_cols, cat_cols, target)
    print(len(x_train), len(x_test))
    regr = LinearRegression()
    regr.fit(x_train, y_train)
    print(regr.score(x_test, y_test))
    print(regr.coef_)


def logistic_regression(data=None, num_cols=None, cat_cols=None, target=None, train_data=None, train_labels=None):
    """
    Train a classifier using logistic regression
    Parameters
    ----------
    data :
    num_cols :
    cat_cols :
    target :
    train_data :
    train_labels :

    Returns
    -------

    """
    if train_data is not None and train_labels is not None:
        clf = LogisticRegression(penalty="elasticnet")
        clf.fit(train_data, train_labels)
        return clf
    else:
        x_train, x_test, y_train, y_test = create_training_data(data, num_cols, cat_cols, target)
        clf = LogisticRegression(penalty="elasticnet")
        clf.fit(x_train, y_train)
        # TODO display results
        return clf



def elastic_net(data=None, num_cols=None, cat_cols=None, target=None, train_data=None, train_labels=None):
    """
    Fit an elastic net regression model to the provided data
    :param data: numpy array containing numeric training data
    :type data: list-like numeric
    :param target: target column to predict
    :type target: list-like numeric
    :return:
    :rtype:
    """
    if train_data is not None and train_labels is not None:
        regr = ElasticNet()
        regr.fit(train_data, train_labels)
        return regr
    else:
        x_train, x_test, y_train, y_test = create_training_data(data, num_cols, cat_cols, target)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        # print(x_train)
        # print(list(y_train))
        clf = ElasticNet()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(clf.score(x_test, y_test))
        explainer = shap.KernelExplainer(clf.predict, x_train)
        shap_values = explainer.shap_values(x_train, nsamples=100)
        print(shap_values)
        # shap.summary_plot(shap_values, x_train)
        plt.tight_layout()
        # plt.show()
        print(explainer.expected_value)
        shap.force_plot(explainer.expected_value, shap_values[0], x_train.iloc[0,:])
        plt.show()
        # probs = np.max(clf.predict_proba(x_test), axis=1)
        # print(probs)
        # TODO make this more general
        # if len(y_train.unique()) == 2:
        #     roc_auc = create_roc_auc_plot(y_test.values, probs)
        # else:
        #     # TODO plot regression result
        #     pass
        # feature_importances = plot_feature_importances(x_train, clf.feature_importances_)
        return clf



# def plot_regression_results(x, pred_y, y, target):
#     """
#     Plot the results of a regression analysis
#     :param x:
#     :type x:
#     :param pred_y:
#     :type pred_y:
#     :param y:
#     :type y:
#     :return:
#     :rtype:
#     """
#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(x=list(x), y=pred_y, name=f"Predicted {target}", mode="markers+lines"))
#     fig_pred.add_trace(go.Scatter(x=list(x), y=y, name=f"True {target}", mode="markers+lines"))
#
#     fig_pred.update_layout(
#         title="Predictions of Regression Model compared to true Values in held out test set",
#         xaxis_title="Patients",
#         yaxis_title=f"{target}",
#         font={
#             "family": "Courier New, monospace",
#             "size":18}
#     )
#     fig_pred.show()
#
#     fig_perf = go.Figure()
#     y_pred_scaled = pred_y/np.linalg.norm(pred_y)
#     y_scaled = y/np.linalg.norm(y)
#     fig_perf.add_trace(go.Scatter(x=y_pred_scaled, y=y_scaled, mode="markers"))
#     fig_perf.add_trace(go.Scatter(x=[0,1], y=[0, 1], mode='lines'))
#
#     fig_perf.update_layout(
#         title="Prediction Performance of Regression",
#         xaxis_title=f"Predicted {target}",
#         yaxis_title=f"True {target}",
#         font={
#             "family": "Courier New, monospace",
#             "size":18},
#     )
#     fig_perf.show()
#
# def plot_regression_coefficients(df, regression_coefficients, n=50):
#     """
#     Plot the coefficients of a regression model sorted by absolute contribution
#     :param df:
#     :type df:
#     :param regression_coefficients:
#     :type regression_coefficients:
#     :param n:
#     :type n:
#     :return:
#     :rtype:
#     """
#     # get indices of max n elements
#     idx = np.flip(np.abs(regression_coefficients).argsort()[-n:][::-1])
#     sorted_reg = regression_coefficients[idx]
#     fig = go.Figure(go.Bar(x=np.abs(sorted_reg), y=list(df.columns[idx]), orientation="h"))
#     fig.show()


if __name__ == '__main__':
    df_sars = load_data("walz_data.csv", na_values=["<NA>"])

    excluded_variables = ['Patienten-ID']

    print("Linear models main")
    binary_target = "Überhaput Antikörperantwort 0=nein"
    multi_target = "III.6: Haben Sie sich krank gefühlt?"
    df_sars[multi_target] = df_sars[multi_target].astype("category")
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"

    linear_model(df_sars, regr_target, cv=False)

