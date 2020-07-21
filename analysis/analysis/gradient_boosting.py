import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import shap
from sklearn.metrics import accuracy_score
from analysis import *
from catboost.utils import get_roc_curve, get_confusion_matrix


def gradient_boosting_regressor(data, excluded_num_cols, excluded_cat_cols, target, tune_parameters=False,
                                test_data=None, display_results=True):
    """

    :param data: base dataframe
    :type data:
    :param excluded_num_cols: numerical columns to to be excluded
    :type excluded_num_cols:
    :param excluded_cat_cols: categorical columns to be excluded
    :param target: numerical column to predict
    :type target:
    :param test_data: separate dataset containing held out test data set to test the finished and validated model
    :return:
    :rtype:
    """

    if not tune_parameters:
        train_pool, test_pool, x_train, y_train, y_test = create_boost_training_data(data,
                                                                                     excluded_num_cols,
                                                                                     excluded_cat_cols, target)
        model = CatBoostRegressor(iterations=40, depth=8, learning_rate=1, verbose=False)
        model.fit(train_pool)
        preds = model.predict(test_pool)
        # preds_proba = model.predict_proba(test_data)
        # res = model.calc_feature_statistics(train_data, feature=2)
        # print(res)

        score = r2_score(y_test, preds)
        print(f"Score: {score}")
        # print(np.mean(shap_values, axis=1))
        print(model.get_best_score())
        if display_results:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_train)
            shap.summary_plot(shap_values, x_train)
            shap.summary_plot(shap_values, x_train, plot_type="bar")
            # shap.dependence_plot("SARS-CoV-2 Ak roche", shap_values, x_train)
            interaction_values = explainer.shap_interaction_values(train_pool)
            shap.summary_plot(interaction_values, x_train)
            plot_regression_results(list(range(len(preds))), preds, y_train, target)
            # plot_regression_coefficients(x_train, model.feature_importances_, n=10)
        return model, score
    else:
        if test_data:
            pass
        else:
            train_pool, test_pool, x_train, y_train, y_test = create_boost_training_data(data,
                                                                                         excluded_num_cols,
                                                                                         excluded_cat_cols,
                                                                                         target)
        model = CatBoostRegressor()
        grid = {
            "iterations": [40, 60, 100],
            "learning_rate": [0.01, 0.1, 1],
            "depth": [4, 6, 8, 10],
            "l2_leaf_reg": [3, 5, 7, 9],

        }
        grid_result = model.grid_search(grid, X=train_pool, verbose=False)
        preds = model.predict(test_pool)
        # preds_proba = model.predict_proba(test_data)
        # res = model.calc_feature_statistics(train_data, feature=2)
        # print(res)
        score = r2_score(y_test, preds)
        print(f"Score: {score}")
        if display_results:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_train)
            # print(np.mean(shap_values, axis=1))
            print(model.get_best_score())
            shap.summary_plot(shap_values, x_train)
            shap.summary_plot(shap_values, x_train, plot_type="bar")
            plot_regression_results(list(range(len(preds))), preds, y_test, target)
            # plot_regression_coefficients(x_train, model.feature_importances_, n=10)
        return model, score


def gradient_boosting_classifier(data, exluded_num_cols, excluded_cat_cols, target, sub_category=None,
                                 tune_parameters=False, display_results=True):
    """
    Create a catboost classifier with the given input and display the results
    :param data: base dataframe
    :type data:
    :param exluded_num_cols: numerical columns to include in analysis
    :type exluded_num_cols:
    :param excluded_cat_cols: categorical columns to use in analysis
    :type excluded_cat_cols:
    :param target: categorical column which to use for classifying
    :type target:
    :return:
    :rtype:
    """

    if not tune_parameters:

        # Binary classification case when a binary column is picked or a subcategory is set
        if len(data[target].dropna().unique()) == 2 or sub_category:
            # Train the model in standard configuration
            train_pool, test_pool, x_train, y_train, y_test = create_boost_training_data(data,
                                                                                         exluded_num_cols,
                                                                                         excluded_cat_cols,
                                                                                         target,
                                                                                         target_cat=sub_category)
            model = CatBoostClassifier(iterations=40,
                                       learning_rate=1,
                                       depth=8,
                                       loss_function="Logloss",
                                       custom_metric=["Logloss", "AUC"])
        else:
            # Train the model in standard configuration
            train_pool, test_pool, x_train, y_train, y_test = create_boost_training_data(data,
                                                                                         exluded_num_cols,
                                                                                         excluded_cat_cols,
                                                                                         target)
            model = CatBoostClassifier(iterations=40,
                                       learning_rate=1,
                                       depth=8,
                                       loss_function="MultiClassOneVsAll",
                                       custom_metric=["MultiClassOneVsAll", "AUC"])
        model.fit(train_pool)
    else:
        # Perform cross validated hyper parameter tuning on the training set
        train_pool, test_pool, x_train, y_train, y_test = create_boost_training_data(data,
                                                                                     exluded_num_cols,
                                                                                     excluded_cat_cols,
                                                                                     target,
                                                                                     target_cat=sub_category)

        # Binary classification case when a binary column is picked or a subcategory is set
        if len(data[target].dropna().unique()) == 2 or sub_category:
            model = CatBoostClassifier(
                loss_function="Logloss",
                custom_metric=["Logloss", "AUC"])
        else:

            model = CatBoostClassifier(
                loss_function="MultiClassOneVsAll",
                custom_metric=["MultiClassOneVsAll", "AUC"]
            )
        grid = {
            "iterations": [40, 60, 100],
            "learning_rate": [0.01, 0.1, 1],
            "depth": [4, 6, 10],
            "l2_leaf_reg": [3, 5, 7]

        }
        grid_result = model.grid_search(grid, X=train_pool, plot=False, verbose=True)
    pred = model.predict(test_pool)
    if len(data[target].dropna().unique()) == 2:
        score = accuracy_score(y_test, pred == "True")
    else:
        # TODO implement multiclass scoring
        score = accuracy_score(y_test, np.squeeze(pred))
    if display_results:
        print(score)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train)
        # print(model.get_best_score())
        shap.summary_plot(shap_values, x_train)
        # print(model.eval_metrics(test_pool, metrics=["AUC"]))
        # # If binary plot roc curve
        if len(data[target].dropna().unique()) == 2 or sub_category:
            get_roc_curve(model, test_pool, plot=True)
        # otherwise plot confusion matrix
        else:
            confusion_matrix = get_confusion_matrix(model, test_pool)
            plot_confusion_matrix(confusion_matrix)

    return model, score


def create_boost_training_data(data, excluded_numerical_columns, excluded_categorical_columns, target, target_cat=None,
                               test=False):
    """
    Create training data in the required format for using catboost gradient boosting,
    if test splits the data into a training and testing test.

    :param test:
    :type test:
    :param data:
    :type data:
    :param num_cols:
    :type num_cols:
    :param cat_cols:
    :type cat_cols:
    :param target:
    :type target:
    :param target_cat:
    :type target_cat:
    :return:
    :rtype:
    """
    # TODO check for binary classification case
    # Clean up selection for creating training data
    # print(list(data.columns))
    # num_cols = num_cols.copy()
    # cat_cols = cat_cols.copy()
    numerical_columns, categorical_columns = find_variables(data,
                                                            excluded_categorical_columns,
                                                            excluded_numerical_columns,
                                                            min_available=20)

    selected_cols = numerical_columns + categorical_columns
    selected_cols.remove(target)
    # TODO clean this up to not remove the column from in memory list

    if target in categorical_columns:
        categorical_columns.remove(target)
        target = data[target].copy()
    else:
        numerical_columns.remove(target)
        target = data[target].copy()
        target = target.astype(float)

    # Remove data points that do not have a value in the target column
    train_data = data[selected_cols].copy()
    train_data = train_data[target.notnull()]
    target = target[target.notnull()]
    # convert to binary when column has only two values or sub target is set
    if target_cat:
        target = target == target_cat
    elif len(target.dropna().unique()) == 2:
        target = target == target.unique()[0]

    # Clean up data
    for float_col in numerical_columns:
        train_data[float_col] = train_data[float_col].astype(float)

    for cat_col in categorical_columns:
        train_data[cat_col] = train_data[cat_col].fillna("Unknown")
        if train_data[cat_col].dtype.name == "string":
            train_data[cat_col] = train_data[cat_col].astype("object")
    x_train, x_test, y_train, y_test = train_test_split(train_data, target)
    train_pool = Pool(x_train, y_train, cat_features=categorical_columns)
    test_pool = Pool(x_test, y_test, cat_features=categorical_columns)
    return train_pool, test_pool, x_train, y_train, y_test


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
            "size": 18}
    )
    fig_pred.show()

    fig_perf = go.Figure()
    y_pred_scaled = pred_y / np.linalg.norm(pred_y)
    y_scaled = y / np.linalg.norm(y)
    fig_perf.add_trace(go.Scatter(x=y_pred_scaled, y=y_scaled, mode="markers"))
    fig_perf.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines'))

    fig_perf.update_layout(
        title="Prediction Performance of Regression",
        xaxis_title=f"Predicted {target}",
        yaxis_title=f"True {target}",
        font={
            "family": "Courier New, monospace",
            "size": 18},
    )
    fig_perf.show()


def plot_confusion_matrix(cm, classes=None):
    """
    Plot a confusion matrix for a multi label classification case
    :param cm:
    :type cm:
    :param classes:
    :type classes:
    :return:
    :rtype:
    """
    accuracy = cm / np.sum(cm, axis=0)
    fig = go.Figure(
        data=go.Heatmap(
            z=accuracy,
            x=[f"Class {i}" for i in range(len(cm))],
            y=[f"Class {i}" for i in range(len(cm))]

        )
    )
    fig.show()


if __name__ == '__main__':
    df_sars = load_data(
        "C:\\hypothesis\\repositories\\server\\walzLabBackend\\notebook\\15052020SARS-CoV-2_final.xlsx",
        two_sheets=False)
    excluded_categorical_columns = ['Patienten-ID', 'III.4b: wenn ja, seit wann(Datum)?',
                                    'III.2Wann wurde der Abstrich durchgeführt(Datum)?', "Eingabedatum",
                                    'III.18: Bis ungefähr wann hatten Sie Symptome(Datum)?']
    excluded_numerical_columns = ["VII.1B: OD IgG Spike 1 Protein rekombinant",
                                  "VII.1C: OD IgG Nucleocapsid Protein rekombinant",
                                  "VIII.1A: Bewertung IgG RBD Peptid rekombinant",
                                  "VIII.1C: Bewertung IgG Nucleocapsid Protein rekombinant",
                                  "SARS-COV-2 IgG Euroimmun",
                                  "VIII.1B: Bewertung IgG Spike 1 Protein rekombinant",
                                  "SARS-CoV-2 IgG Euroimmun"]
    numerical_columns, categorical_columns = find_variables(df_sars,
                                                            excluded_categorical_columns,
                                                            excluded_numerical_columns,
                                                            min_available=20)
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"
    gradient_boosting_regressor(df_sars, excluded_numerical_columns, excluded_categorical_columns, regr_target,
                                tune_parameters=False)
    # gradient_boosting_classifier(df_sars, excluded_numerical_columns, excluded_categorical_columns,
    #                              "IX.1C HLA",
    #                              tune_parameters=True)
