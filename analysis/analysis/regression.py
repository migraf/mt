import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, LogisticRegression
from sklearn.preprocessing import normalize
# from process_data import load_data
# from util import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# from analysis import create_training_data, find_variables
from .util import create_training_data, find_variables
import shap

from .process_data import load_data


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



def plot_regression_results(x, pred_y, y, target):
    """
    Plot the results of a regression analysis
    :param x:
    :type x:
    :param pred_y:
    :type pred_y:
    :param y:
    :type y:
    :return:
    :rtype:
    """
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=list(x), y=pred_y, name=f"Predicted {target}", mode="markers+lines"))
    fig_pred.add_trace(go.Scatter(x=list(x), y=y, name=f"True {target}", mode="markers+lines"))

    fig_pred.update_layout(
        title="Predictions of Regression Model compared to true Values in held out test set",
        xaxis_title="Patients",
        yaxis_title=f"{target}",
        font={
            "family": "Courier New, monospace",
            "size":18}
    )
    fig_pred.show()

    fig_perf = go.Figure()
    y_pred_scaled = pred_y/np.linalg.norm(pred_y)
    y_scaled = y/np.linalg.norm(y)
    fig_perf.add_trace(go.Scatter(x=y_pred_scaled, y=y_scaled, mode="markers"))
    fig_perf.add_trace(go.Scatter(x=[0,1], y=[0, 1], mode='lines'))

    fig_perf.update_layout(
        title="Prediction Performance of Regression",
        xaxis_title=f"Predicted {target}",
        yaxis_title=f"True {target}",
        font={
            "family": "Courier New, monospace",
            "size":18},
    )
    fig_perf.show()

def plot_regression_coefficients(df, regression_coefficients, n=50):
    """
    Plot the coefficients of a regression model sorted by absolute contribution
    :param df:
    :type df:
    :param regression_coefficients:
    :type regression_coefficients:
    :param n:
    :type n:
    :return:
    :rtype:
    """
    # get indices of max n elements
    idx = np.flip(np.abs(regression_coefficients).argsort()[-n:][::-1])
    sorted_reg = regression_coefficients[idx]
    fig = go.Figure(go.Bar(x=np.abs(sorted_reg), y=list(df.columns[idx]), orientation="h"))
    fig.show()


if __name__ == '__main__':
    data = df_sars = load_data(
        "D:\\hypothesis\\hypothesis\\server\\walzLabBackend\\flaskr\\user_data\\15052020SARS-CoV-2_final.xlsx")

    excluded_categorical_columns = ['Patienten-ID', 'Eingabedatum', 'III.2Wann wurde der Abstrich durchgeführt(Datum)?',
                                    'III.4b: wenn ja, seit wann(Datum)?']
    excluded_numerical_columns = []
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"

    num_columns, cat_columns = find_variables(df_sars,
                                              excluded_categorical_columns,
                                              excluded_numerical_columns,
                                              min_available=20,
                                              display=True
                                              )
    # svm_classifier(df_sars, num_columns, cat_columns, target)
    elastic_net(df_sars, num_columns, cat_columns, regr_target)

