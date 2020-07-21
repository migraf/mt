from analysis import *
from random_forest import random_forest_classifier, random_forest_regressor
from time import time
import json
from svm import svm_classifier, svm_regression
from gradient_boosting import gradient_boosting_classifier, gradient_boosting_regressor
from regression import elastic_net


def evaluate_binary_prediction():
    """

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
    n_evaluation_runs = 50
    results = {
        "svm": {"untuned": {}, "tuned": {}},
        "random_forest": {"untuned": {}, "tuned": {}},
        "gradient_boosting": {"untuned": {}, "tuned": {}},
        "elastic_net": {"untuned": {}, "tuned": {}}
    }

    target = "VII.1A: OD IgG RBD Peptid rekombinant"
    data = load_data(
        "C:\\hypothesis\\repositories\\server\\walzLabBackend\\notebook\\15052020SARS-CoV-2_final.xlsx")

    excluded_categorical_columns = ['Patienten-ID', 'Eingabedatum', 'III.2Wann wurde der Abstrich durchgeführt(Datum)?',
                                    'III.4b: wenn ja, seit wann(Datum)?']

    excluded_numerical_columns = ["VII.1B: OD IgG Spike 1 Protein rekombinant",
                                  "VII.1C: OD IgG Nucleocapsid Protein rekombinant",
                                  "VIII.1A: Bewertung IgG RBD Peptid rekombinant",
                                  "VIII.1C: Bewertung IgG Nucleocapsid Protein rekombinant",
                                  "SARS-COV-2 IgG Euroimmun",
                                  "VIII.1B: Bewertung IgG Spike 1 Protein rekombinant",
                                  "SARS-CoV-2 IgG Euroimmun"]

    num_columns, cat_columns = find_variables(data,
                                              excluded_categorical_columns,
                                              excluded_numerical_columns,
                                              min_available=20,
                                              display=True
                                              )

    # Define hyper parameter tuning parameters

    random_forest_tuning_params = {
        "n_estimators": [10, 100, 1000],
        "max_depth": [None, 4, 6, 8],
        "max_features": [None, "auto", "log2"],
        "min_samples_leaf": [1, 5, 10]
    }

    svm_tuning_params = {
        "C": [1, 10, 100],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["auto", "scale"]
    }
    elastic_net_tuning_params = {
        "alpha": [0.1, 0.5, 1.0],
        "l1_ratio": [0, 0.2, 0.5, 0.7],
        "max_iter": [1000, 10000, 20000]
    }

    rf_untuned_times = []
    rf_tuned_times = []
    rf_scores = []
    rf_tuned_scores = []
    rf_tuned_params = []
    svm_untuned_times = []
    svm_tuned_times = []
    svm_scores = []
    svm_tuned_scores = []
    svm_tuned_params = []
    cb_untuned_times = []
    cb_tuned_times = []
    cb_scores = []
    cb_tuned_scores = []
    cb_tuned_params = []
    elastic_untuned_times = []
    elastic_tuned_times = []
    elastic_scores = []
    elastic_tuned_scores = []
    elastic_tuned_params = []

    for i in range(n_evaluation_runs):
        # Track total run time
        evaluation_start = time()
        x_train, x_test, y_train, y_test = create_training_data(data, num_columns, cat_columns, target)
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)
        print(f"Evaluation run {i + 1}/{n_evaluation_runs}")

        # Random forest evaluation

        print("Default Random Forest Classifier:")
        rf_start = time()
        rf_clf = random_forest_regressor(train_data=x_train, train_labels=y_train)
        rf_score = rf_clf.score(x_test, y_test)
        rf_stop = time()
        rf_untuned_times.append(rf_stop - rf_start)
        print(f"\tRF Finished - Score: {rf_score}, Time {rf_untuned_times[i]}")
        rf_scores.append(rf_score)
        print("CV Tuning Random Forest Classifier:")
        rf_tuned_clf, cv_results, rf_params = cross_validation_tuning(rf_clf, random_forest_tuning_params, x_train,
                                                                      y_train)
        rf_tuning_stop = time()
        rf_tuned_times.append(rf_tuning_stop - rf_stop)
        rf_tuned_score = rf_tuned_clf.score(x_test, y_test)
        rf_tuned_scores.append(rf_tuned_score)
        rf_tuned_params.append(rf_params)
        print(f"\tRF parameter tuning finished - Score: {rf_tuned_score}, Time: {rf_tuned_times[i]}")

        # SVM Evaluation

        print("Default SVM Classifier:")
        svm_start = time()
        svm_clf = svm_regression(train_data=x_train, train_labels=y_train)
        svm_score = svm_clf.score(x_test, y_test)
        svm_stop = time()
        svm_untuned_times.append(svm_stop - svm_start)
        svm_scores.append(svm_score)
        print(f"\tSVM Finished - Score: {svm_score}, Time {svm_untuned_times[i]}")

        print("CV Tuning SVM Classifier:")
        svm_tuned_clf, cv_results, svm_params = cross_validation_tuning(svm_clf, svm_tuning_params, x_train, y_train)
        svm_tuning_stop = time()
        svm_tuned_times.append(svm_tuning_stop - svm_stop)
        svm_tuned_score = svm_tuned_clf.score(x_test, y_test)
        svm_tuned_scores.append(svm_tuned_score)
        svm_tuned_params.append(svm_params)
        print(f"\tSVM parameter tuning finished - Score: {svm_tuned_score}, Time: {svm_tuned_times[i]}")

        # Gradient Boosting Evaluation
        print("Catboost Model:")
        cb_start_time = time()
        gb_clf, cb_score = gradient_boosting_regressor(data, excluded_numerical_columns, excluded_categorical_columns,
                                                       target=target, display_results=False)
        cb_untuned_times.append(time() - cb_start_time)
        cb_scores.append(cb_score)
        print(f"\tCB training finished - Score: {cb_score}, Time: {cb_untuned_times[i]}")
        print("CV Tuning Catboost Model:")
        cb_tuning_start_time = time()
        gb_clf, cb_score = gradient_boosting_regressor(data, excluded_numerical_columns, excluded_categorical_columns,
                                                       target=target, tune_parameters=True, display_results=False)
        cb_tuned_times.append(time() - cb_tuning_start_time)
        cb_tuned_scores.append(cb_score)
        print(f"\tCV tuning Catboost finished - Score: {cb_score}, Time: {cb_tuned_times[i]}")

        # ElasticNet Evaluation
        print("Elastic Net")
        elastic_start_time = time()
        regr = elastic_net(train_data=x_train, train_labels=y_train)
        elastic_score = regr.score(x_test, y_test)

        elastic_stop = time()
        elastic_untuned_times.append(elastic_stop - elastic_start_time)
        elastic_scores.append(elastic_score)
        print(f"Elastic net finished - Score {elastic_score}, Time: {elastic_untuned_times[i]}")

        print("CV Tuning ElasticNet:")
        elastic_tuned_regr, cv_results, elastic_params = cross_validation_tuning(regr,
                                                                                 elastic_net_tuning_params, x_train,
                                                                                 y_train)
        elastic_tuning_stop = time()
        elastic_tuned_times.append(elastic_tuning_stop - elastic_stop)
        elastic_tuned_score = elastic_tuned_regr.score(x_test, y_test)
        elastic_tuned_scores.append(elastic_tuned_score)
        elastic_tuned_params.append(elastic_params)
        print(f"\tElasticNet parameter tuning finished - Score: {elastic_tuned_score}, Time: {elastic_tuned_times[i]}")

        single_run_time = time() - evaluation_start
        print(f"Completed Evaluation {i + 1}:\n total: {single_run_time}s,"
              f" remaining: {(n_evaluation_runs - i) * single_run_time}")

    results["random_forest"]["tuned"]["times"] = rf_tuned_times
    results["random_forest"]["untuned"]["times"] = rf_untuned_times
    results["random_forest"]["tuned"]["scores"] = rf_tuned_scores
    results["random_forest"]["untuned"]["scores"] = rf_scores
    results["random_forest"]["tuned"]["avg_score"] = np.mean(rf_tuned_scores)
    results["random_forest"]["untuned"]["avg_score"] = np.mean(rf_scores)
    results["random_forest"]["tuned"]["params"] = rf_tuned_params
    results["svm"]["tuned"]["times"] = svm_tuned_times
    results["svm"]["untuned"]["times"] = svm_untuned_times
    results["svm"]["tuned"]["scores"] = svm_tuned_scores
    results["svm"]["untuned"]["scores"] = svm_scores
    results["svm"]["tuned"]["avg_score"] = np.mean(svm_tuned_scores)
    results["svm"]["untuned"]["avg_score"] = np.mean(svm_scores)
    results["svm"]["tuned"]["params"] = svm_tuned_params
    results["gradient_boosting"]["tuned"]["times"] = cb_tuned_times
    results["gradient_boosting"]["untuned"]["times"] = cb_untuned_times
    results["gradient_boosting"]["tuned"]["scores"] = cb_tuned_scores
    results["gradient_boosting"]["untuned"]["scores"] = cb_scores
    results["gradient_boosting"]["tuned"]["avg_score"] = np.mean(cb_tuned_scores)
    results["gradient_boosting"]["untuned"]["avg_score"] = np.mean(cb_scores)
    results["elastic_net"]["tuned"]["times"] = elastic_tuned_times
    results["elastic_net"]["untuned"]["times"] = elastic_untuned_times
    results["elastic_net"]["tuned"]["scores"] = elastic_tuned_scores
    results["elastic_net"]["untuned"]["scores"] = elastic_scores
    results["elastic_net"]["tuned"]["avg_score"] = np.mean(elastic_tuned_scores)
    results["elastic_net"]["untuned"]["avg_score"] = np.mean(elastic_scores)
    results["elastic_net"]["tuned"]["params"] = elastic_tuned_params

    # Store results in file
    with open("regression_results_subset.json", "w") as br:
        json.dump(results, br, indent=4)


if __name__ == '__main__':
    evaluate_binary_prediction()
