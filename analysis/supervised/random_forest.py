import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from util import detect_prediction_type, create_training_data, cross_validation_tuning, load_data
from display import display_model_performance, display_feature_importances


def random_forest(data, target, excluded_variables=[], prediction_type=None, n_estimators=100, criterion=None,
                  max_depth=None, max_features=None, min_samples_leaf=1, cv=True, cv_params=None, display=True,
                  shap=True, prepare_data=True):
    if prediction_type:
        model_subtype = prediction_type
    else:
        model_subtype = detect_prediction_type(data, target)

    if prepare_data:
        x_train, x_test, y_train, y_test = create_training_data(data, target, excluded_variables)
    else:
        x_train, x_test = data[0], data[1]
        y_train, y_test = target[0], target[1]
    print(f"Creating a random forest {model_subtype} model")
    if model_subtype in ["binary", "multi-class"]:
        if criterion:
            pred = RandomForestClassifier(random_state=0, n_estimators=n_estimators, criterion=criterion,
                                          max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          max_features=max_features)
        else:
            pred = RandomForestClassifier(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf, max_features=max_features)

        # TODO check if this is really necessary
        y_train = y_train.astype("str")
        y_test = y_test.astype("str")
    else:
        if criterion:
            pred = RandomForestRegressor(random_state=0, n_estimators=n_estimators, criterion=criterion,
                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                         max_features=max_features)
        else:
            pred = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf, max_features=max_features)

        y_train = y_train.astype("float")
        y_test = y_test.astype("float")
    if cv:
        if not cv_params:
            cv_params = {
                "n_estimators": [10, 100, 500],
                "max_depth": [None, 6, 8],
                "max_features": [None, "auto", "log2"],
                "min_samples_leaf": [1, 5, 10]
            }
        pred, cv_results, param_results = cross_validation_tuning(pred, cv_params, x_train, y_train)
        print(param_results)

    else:
        pred.fit(x_train, y_train)

    if display:
        display_model_performance(pred, model_subtype, x_test, y_test, target)
        shap_values = display_feature_importances(pred, x_train, x_test, model_type="tree", return_shap=shap)
    else:
        print(f"Score: {pred.score(x_test, y_test)}")
        # TODO print additional information
    if shap:
        return pred, shap_values
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
    df_sars = load_data("../../datasets/walz_data.csv", na_values=["<NA>"])

    # excluded_variables = ['Patienten-ID']
    excluded_variables = ['Patienten-ID', "VII.1B: OD IgG Spike 1 Protein rekombinant",
                          "VII.1C: OD IgG Nucleocapsid Protein rekombinant",
                          "VIII.1A: Bewertung IgG RBD Peptid rekombinant",
                          "SARS-CoV-2 IgG Euroimmun",
                          "VIII.1B: Bewertung IgG Spike 1 Protein rekombinant"
                          ]

    print("Random forest main")
    binary_target = "Überhaput Antikörperantwort 0=nein"
    multi_target = "III.6: Haben Sie sich krank gefühlt?"
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"


    # random_forest_classifier(df_sars, num_columns, cat_columns, target)
    # random_forest_regressor(df_sars, num_columns, cat_columns, regr_target)
    pred, shap = random_forest(df_sars, regr_target, excluded_variables=excluded_variables, cv=False, shap=True)
