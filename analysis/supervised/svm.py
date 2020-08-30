from sklearn.svm import SVC, SVR
from util import detect_prediction_type, create_training_data, cross_validation_tuning, load_data
from display import display_model_performance, display_feature_importances


def svm(data, target, excluded_variables=[], prediction_type=None, kernel='rbf', C=1.0, degree=3,
                 cv=True, cv_params=None, display=True, shap=True, prepare_data=True):
    if prediction_type:
        model_subtype = prediction_type
    else:
        model_subtype = detect_prediction_type(data, target)

    if prepare_data:
        x_train, x_test, y_train, y_test = create_training_data(data, target, excluded_variables)
    else:
        x_train, x_test = data[0], data[1]
        y_train, y_test = target[0], target[1]
    print(f"Creating a svm {model_subtype} model")
    if model_subtype in ["binary", "multi-class"]:
        pred = SVC(kernel=kernel, C=C, degree=degree, probability=True)
        y_train = y_train.astype("str")
        y_test = y_test.astype("str")

        # Kernel function for shap value prediction
        f = lambda x: pred.predict_proba(x)[:,1]
    else:
        pred = SVR(kernel=kernel, C=C, degree=degree)
        y_train = y_train.astype("float")
        y_test = y_test.astype("float")

    if cv:
        # Perform cross validation hyper parameter tuning
        if not cv_params:
            cv_params = {
                "C": [1, 10, 100],
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "gamma": ["auto", "scale"]
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
            shap_values = display_feature_importances(pred, x_train, x_test, return_shap=shap)
    else:
        print(f"Score: {pred.score(x_test, y_test)}")
        # TODO print additional information
    if shap:
        return pred, shap_values
    return pred



def svm_classifier(data=None, num_cols=None, cat_cols=None, target=None, train_data=None, train_labels=None):
    """
    Train an svm classifier with the given data and target
    :param cat_cols:
    :type cat_cols:
    :param num_cols:
    :type num_cols:
    :param data: raw training data
    :type data:
    :param target: target column in the data
    :type target:
    :return:
    :rtype:
    """
    if train_data is not None and train_labels is not None:
        clf = SVC()
        clf.fit(train_data, train_labels)
        return clf
    else:

        x_train, x_test, y_train, y_test = create_training_data(data, num_cols, cat_cols, target, na_strategy="fill")
        # TODO multiclass classification
        y_train = y_train.astype("str")
        y_test = y_test.astype("str")
        print(x_train)
        print(list(y_train))
        clf = SVC(gamma="auto")
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(y_pred)
        print(clf.score(x_test, y_test))
        # probs = np.max(clf.predict_proba(x_test), axis=1)
        # print(probs)
        # # TODO make this more general
        # roc_auc = create_roc_auc_plot(y_test.values, probs)
        # feature_importances = plot_feature_importances(x_train, clf.feature_importances_)
        return clf



def svm_regression(data=None, num_cols=None, cat_cols=None, target=None, train_data=None, train_labels=None):
    """
    Train an svm regressor with the given data and target
    :param cat_cols:
    :type cat_cols:
    :param num_cols:
    :type num_cols:
    :param data:
    :type data:
    :param target:
    :type target:
    :return:
    :rtype:
    """
    if train_data is not None and train_labels is not None:
        clf = SVR()
        clf.fit(train_data, train_labels)
        return clf
    else:
        x_train, x_test, y_train, y_test = create_training_data(data, num_cols, cat_cols, target, na_strategy="fill")
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        # print(x_train)
        # print(list(y_train))
        clf = SVR()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(clf.score(x_test, y_test))
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



if __name__ == '__main__':
    df_sars = load_data("walz_data.csv", na_values=["<NA>"])

    excluded_variables = ['Patienten-ID']

    print("Linear models main")
    binary_target = "Überhaput Antikörperantwort 0=nein"
    multi_target = "III.6: Haben Sie sich krank gefühlt?"
    df_sars[multi_target] = df_sars[multi_target].astype("category")
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"

    svm(df_sars, binary_target, cv=False)

