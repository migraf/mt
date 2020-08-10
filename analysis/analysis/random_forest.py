import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
from analysis import *


def random_forest(data=None, num_cols=None, cat_cols=None, target=None, cv=True, cv_params=None, display=True):
    model_subtype = detect_prediction_type(data, target)

    # Create fitting predictor
    x_train, x_test, y_train, y_test = create_training_data(data, num_cols, cat_cols, target)
    if model_subtype in ["binary", "multi-class"]:
        pred = RandomForestClassifier(random_state=0)
        # TODO check if this is really necessary
        y_train = y_train.astype("str")
        y_test = y_test.astype("str")
    else:
        pred = RandomForestRegressor(random_state=0)
        y_train = y_train.astype("float")
        y_test = y_test.astype("float")
    if cv:
        # TODO change to use cross validation
        if not cv_params:
            cv_params = {
                "n_estimators": [10, 100, 1000],
                "max_depth": [None, 4, 6, 8],
                "max_features": [None, "auto", "log2"],
                "min_samples_leaf": [1, 5, 10]
            }
        pred, cv_results, param_results = cross_validation_tuning(pred, cv_params, x_train, y_train)
        print(param_results)

    else:
        pred.fit(x_train, y_train)

    if display:
        display_model_performance(pred, model_subtype, x_test, y_test, target)
        display_feature_importances(pred, x_train, x_test, model_type="tree")
    else:
        print(f"Score: {pred.score(x_test, y_test)}")
        # TODO print additional information

    return pred


def random_forest_classifier(data=None, num_cols=None, cat_cols=None, target=None, train_data=None, train_labels=None):
    """
    Train a random forest model on the with the selected columns on the selected target using data from the
    given dataframe
    :param data:
    :type data:
    :param num_cols:
    :type num_cols:
    :param cat_cols:
    :type cat_cols:
    :param target:
    :type target:
    :return:
    :rtype:
    """
    # Only train the classifier
    if train_data is not None and train_labels is not None:
        clf = RandomForestClassifier(random_state=0)
        clf.fit(train_data, train_labels)
        return clf
    else:

        x_train, x_test, y_train, y_test = create_training_data(data, num_cols, cat_cols, target)
        y_train = y_train.astype("str")
        y_test = y_test.astype("str")
        print(x_train)
        print(list(y_train))
        clf = RandomForestClassifier(random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(clf.score(x_test, y_test))
        probs = np.max(clf.predict_proba(x_test), axis=1)
        print(probs)
        # TODO make this more general
        if len(y_train.unique()) == 2:
            roc_auc = create_roc_auc_plot(y_test.values, probs)
        else:
            # TODO plot confusion matrix
            pass
        feature_importances = plot_feature_importances(x_train, clf.feature_importances_)
        return clf


def random_forest_regressor(data=None, num_cols=None, cat_cols=None, target=None, train_data=None, train_labels=None):
    if train_data is not None and train_labels is not None:
        clf = RandomForestRegressor(random_state=0)
        clf.fit(train_data, train_labels)
        return clf
    else:
        x_train, x_test, y_train, y_test = create_training_data(data, num_cols, cat_cols, target)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        clf = RandomForestRegressor(random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(clf.score(x_test, y_test))
        # probs = np.max(clf.predict_proba(x_test), axis=1)
        # print(probs)
        # TODO make this more general
        if len(y_train.unique()) == 2:
            roc_auc = create_roc_auc_plot(y_test.values, probs)
        else:
            # TODO plot regression result
            pass
        feature_importances = plot_feature_importances(x_train, clf.feature_importances_)
        return clf


def create_roc_auc_plot(y_true, y_pred, pos_label=None):
    """
    Create a roc curve with the predictions of the classifier
    :param y_true:
    :type y_true:
    :param y_pred:
    :type y_pred:
    :param pos_label:
    :type pos_label:
    :return:
    :rtype:
    """
    # TODO improve plot design
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    auc_score = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
    fig.show()
    return fig


def plot_feature_importances(df, feature_importances, n_features=10):
    """
    Create a correctly named plot of feature importances based on the underlying data frame and the feature importances
    of a model
    :param df:
    :type df:
    :param feature_importances:
    :type feature_importances:
    :return:
    :rtype:
    """
    idx = np.flip(np.abs(feature_importances).argsort()[-n_features:][::-1])
    fig = go.Figure(go.Bar(x=feature_importances[idx], y=list(df.columns[idx]), orientation="h"))
    fig.show()
    return fig


if __name__ == '__main__':
    df_sars = load_data("walz_data.csv")

    excluded_categorical_columns = ['Patienten-ID', 'Eingabedatum', 'III.2Wann wurde der Abstrich durchgeführt(Datum)?',
                                    'III.4b: wenn ja, seit wann(Datum)?']
    # IGG Spike prediction dependencies to be removed
    excluded_numerical_columns = ["VII.1B: OD IgG Spike 1 Protein rekombinant",
                                  "VII.1C: OD IgG Nucleocapsid Protein rekombinant",
                                  "VIII.1A: Bewertung IgG RBD Peptid rekombinant",
                                  "VIII.1C: Bewertung IgG Nucleocapsid Protein rekombinant"]
    # excluded_numerical_columns = []
    num_columns, cat_columns = find_variables(df_sars,
                                              excluded_categorical_columns,
                                              excluded_numerical_columns,
                                              min_available=20,
                                              display=True
                                              )
    print("Random forest main")
    target = "IX.1C HLA"
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"

    # random_forest_classifier(df_sars, num_columns, cat_columns, target)
    # random_forest_regressor(df_sars, num_columns, cat_columns, regr_target)
    pred = random_forest(df_sars, num_columns, cat_columns, regr_target, cv=False)
