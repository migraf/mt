from analysis import *
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import pandas as pd

from util import detect_prediction_type, find_variables


def gradient_boosted_trees(data, target, excluded_variables=[], prediction_type=None, iterations=100, lr=1,
                           l2_leaf_reg=3, depth=8, cv=True, cv_params=None, display=True, shap=True,
                           test_indices=None, score=False):

    train_data = data.copy()
    targ = data[target].copy()
    train_data = train_data.drop(target, axis=1)
    train_data = train_data.drop(columns=excluded_variables)

    train_data = train_data[train_data.columns[train_data.isnull().mean() < 0.5]]
    high_cardinality_variables = []
    for col in train_data.columns:
        if not pd.api.types.is_numeric_dtype(train_data[col]):
            n_unique = len(train_data[col].unique())
            if float(n_unique) / len(train_data[col][train_data[col].notnull()]) >= 0.75:
                high_cardinality_variables.append(col)

    print("removed: ", high_cardinality_variables)
    train_data = train_data.drop(high_cardinality_variables, axis=1)
    num_vars, cat_vars = find_variables(train_data, display=False)
    train_data : pd.DataFrame = train_data[targ.notnull()]
    targ = targ[targ.notnull()]

    # convert numerical variables to float
    for col in num_vars:
        train_data[col] = train_data[col].astype(float)
    # convert categorical variables to object
    for col in cat_vars:
        train_data[col] = train_data[col].fillna("NA")
        train_data[col] = train_data[col].astype("object")

    # Split data into training and test set, either based on given test indices or randomly
    if test_indices:
        train_ind = list(set(train_data.index).difference(test_indices))
        x_test = train_data.iloc[test_indices, :]
        x_train = train_data.iloc[train_ind, :]
        y_test = targ.iloc[test_indices, :]
        y_train = targ.iloc[train_ind, :]
    else:
        x_train, x_test, y_train, y_test = train_test_split(train_data, targ, test_size=0.2)

    if prediction_type:
        model_subtype = prediction_type
    else:
        model_subtype = detect_prediction_type(data, target)

    if model_subtype == "regression":
        pred = CatBoostRegressor(iterations=iterations, learning_rate=lr, l2_leaf_reg=l2_leaf_reg, depth=depth,
                                 cat_features=cat_vars)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
    else:
        pred = CatBoostClassifier(iterations=iterations, learning_rate=lr, l2_leaf_reg=l2_leaf_reg, depth=depth,
                                  cat_features=cat_vars)
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

    train_pool = Pool(x_train, y_train, cat_features=cat_vars)
    test_pool = Pool(x_test, y_test, cat_features=cat_vars)
    if cv:
        if not cv_params:
            cv_params = {
                "iterations": [40, 60, 100],
                "learning_rate": [0.01, 0.1, 1],
                "depth": [4, 8, 10],
                "l2_leaf_reg": [3, 5, 9],

            }
        pred.grid_search(cv_params, x_train, y_train)
    else:
        pred.fit(x_train, y_train)

    if display:
        display_model_performance(pred, model_subtype, x_test, y_test, target)
        shap_values = display_feature_importances(pred, x_train, x_test, model_type="tree", return_shap=shap)
    else:
        print(f"Score: {pred.score(x_test, y_test)}")
    if shap:
        return pred, shap_values
    if score:
        return pred, pred.score(x_test, y_test)
    return pred





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




if __name__ == '__main__':
    df_sars = load_data("walz_data.csv")

    excluded_variables = ['Patienten-ID']



    print("gradient boosting main")
    binary_target = "Überhaput Antikörperantwort 0=nein"
    multi_target = "III.6: Haben Sie sich krank gefühlt?"
    # df_sars[multi_target] = df_sars[multi_target].astype("category")
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"


    gradient_boosted_trees(df_sars, regr_target, excluded_variables=excluded_variables, cv=False)
