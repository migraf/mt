import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score

from analysis import load_data, find_variables, create_training_data


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
    target = "IX.1C HLA"
    data = df_sars = load_data(
        "C:\\hypothesis\\repositories\\server\\walzLabBackend\\notebook\\15052020SARS-CoV-2_final.xlsx")

    excluded_categorical_columns = ['Patienten-ID','Eingabedatum', 'III.2Wann wurde der Abstrich durchgeführt(Datum)?',
                                    'III.4b: wenn ja, seit wann(Datum)?' ]
    excluded_numerical_columns = []
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"

    num_columns, cat_columns = find_variables(df_sars,
                                              excluded_categorical_columns,
                                              excluded_numerical_columns,
                                              min_available=20,
                                              display=True
                                              )
    # svm_classifier(df_sars, num_columns, cat_columns, target)
    svm_regression(df_sars, num_columns, cat_columns, regr_target)

