from analysis import *
from time import time
from analysis import detect_prediction_type


def evaluate_models():
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
    n_evaluation_runs = 20
    results = {
        "svm": {"untuned": {}, "tuned": {}},
        "random_forest": {"untuned": {}, "tuned": {}},
        "gradient_boosting": {"untuned": {}, "tuned": {}},
        "linear_models": {"untuned": {}, "tuned": {}}
    }

    data = load_data("walz_data.csv", na_values=["<NA>"])

    excluded_variables = ['Patienten-ID']

    binary_target = "Überhaput Antikörperantwort 0=nein"
    multi_target = "III.6: Haben Sie sich krank gefühlt?"
    # df_sars[multi_target] = df_sars[multi_target].astype("category")
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"

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
    lm_untuned_times = []
    lm_tuned_times = []
    lm_scores = []
    lm_tuned_scores = []
    lm_tuned_params = []

    prediction_type = "multi-class"

    for i in range(n_evaluation_runs):
        # Track total run time
        evaluation_start = time()
        x_train, x_test, y_train, y_test, ind_train, ind_test = create_training_data(data, multi_target,
                                                                                     excluded_variables=excluded_variables,
                                                                                     test_train_indices=True)
        y_test = y_test.astype(str)
        print(f"Evaluation run {i + 1}/{n_evaluation_runs}")

        # Random forest evaluation

        print("Default Random Forest Classifier:")
        rf_start = time()
        rf_clf = random_forest([x_train, x_test], [y_train, y_test], display=False, shap=False, cv=False,
                               prediction_type=prediction_type, prepare_data=False)
        rf_score = rf_clf.score(x_test, y_test)
        rf_stop = time()
        rf_untuned_times.append(rf_stop - rf_start)
        print(f"\tRF Finished - Score: {rf_score}, Time {rf_untuned_times[i]}")
        rf_scores.append(rf_score)
        print("CV Tuning Random Forest Classifier:")
        rf_tuned_clf = random_forest([x_train, x_test], [y_train, y_test], display=False, shap=False, cv=True,
                                     prediction_type=prediction_type, prepare_data=False)
        rf_tuning_stop = time()
        rf_tuned_times.append(rf_tuning_stop - rf_stop)
        rf_tuned_score = rf_tuned_clf.score(x_test, y_test)
        rf_tuned_scores.append(rf_tuned_score)
        print(f"\tRF parameter tuning finished - Score: {rf_tuned_score}, Time: {rf_tuned_times[i]}")

        # SVM Evaluation

        print("Default SVM Classifier:")
        svm_start = time()
        svm_clf = svm([x_train, x_test], [y_train, y_test], display=False, shap=False, cv=False,
                      prediction_type=prediction_type, prepare_data=False)
        svm_score = svm_clf.score(x_test, y_test)
        svm_stop = time()
        svm_untuned_times.append(svm_stop - svm_start)
        svm_scores.append(svm_score)
        print(f"\tSVM Finished - Score: {svm_score}, Time {svm_untuned_times[i]}")

        print("CV Tuning SVM Classifier:")
        svm_tuned_clf = svm([x_train, x_test], [y_train, y_test], display=False, shap=False,
                            cv=True,
                            prediction_type=prediction_type, prepare_data=False)
        svm_tuning_stop = time()
        svm_tuned_times.append(svm_tuning_stop - svm_stop)
        svm_tuned_score = svm_tuned_clf.score(x_test, y_test)
        svm_tuned_scores.append(svm_tuned_score)
        print(f"\tSVM parameter tuning finished - Score: {svm_tuned_score}, Time: {svm_tuned_times[i]}")

        # Gradient Boosting Evaluation
        print("Catboost Model:")
        cb_start_time = time()
        gb_clf, gb_score = gradient_boosted_trees(data, multi_target, cv=False, display=False,
                                                  shap=False,
                                                  prediction_type=prediction_type, score=True)
        cb_untuned_times.append(time() - cb_start_time)
        cb_scores.append(gb_score)
        print(f"\tCB training finished - Score: {gb_score}, Time: {cb_untuned_times[i]}")
        print("CV Tuning Catboost Model:")
        cb_tuning_start_time = time()
        gb_clf, cb_score = gradient_boosted_trees(data, multi_target, cv=True, display=False,
                                                  shap=False,
                                                  prediction_type=prediction_type, score=True)
        cb_tuned_times.append(time() - cb_tuning_start_time)
        cb_tuned_scores.append(cb_score)
        print(f"\tCV tuning Catboost finished - Score: {cb_score}, Time: {cb_tuned_times[i]}")

        # Linear Model Evaluation
        print("Linear Models")
        elastic_start_time = time()
        lm = linear_model([x_train, x_test], [y_train, y_test], display=False, shap=False, cv=False,
                          prediction_type=prediction_type, prepare_data=False)
        elastic_score = lm.score(x_test, y_test)

        elastic_stop = time()
        lm_untuned_times.append(elastic_stop - elastic_start_time)
        lm_scores.append(elastic_score)
        print(f"Elastic net finished - Score {elastic_score}, Time: {lm_untuned_times[i]}")

        print("CV Tuning ElasticNet:")
        lm_tuned_regr = linear_model([x_train, x_test], [y_train, y_test], display=False, shap=False, cv=True,
                                     prediction_type=prediction_type, prepare_data=False)
        elastic_tuning_stop = time()
        lm_tuned_times.append(elastic_tuning_stop - elastic_stop)
        elastic_tuned_score = lm_tuned_regr.score(x_test, y_test)
        lm_tuned_scores.append(elastic_tuned_score)
        print(f"\tElasticNet parameter tuning finished - Score: {elastic_tuned_score}, Time: {lm_tuned_times[i]}")

        single_run_time = time() - evaluation_start
        print(f"Completed Evaluation {i + 1}:\n total: {single_run_time}s,"
              f" remaining: {(n_evaluation_runs - i) * single_run_time}")

    results["random_forest"]["tuned"]["times"] = rf_tuned_times
    results["random_forest"]["untuned"]["times"] = rf_untuned_times
    results["random_forest"]["tuned"]["scores"] = rf_tuned_scores
    results["random_forest"]["untuned"]["scores"] = rf_scores
    results["random_forest"]["tuned"]["avg_score"] = np.mean(rf_tuned_scores)
    results["random_forest"]["untuned"]["avg_score"] = np.mean(rf_scores)
    # results["random_forest"]["tuned"]["params"] = rf_tuned_params
    results["svm"]["tuned"]["times"] = svm_tuned_times
    results["svm"]["untuned"]["times"] = svm_untuned_times
    results["svm"]["tuned"]["scores"] = svm_tuned_scores
    results["svm"]["untuned"]["scores"] = svm_scores
    results["svm"]["tuned"]["avg_score"] = np.mean(svm_tuned_scores)
    results["svm"]["untuned"]["avg_score"] = np.mean(svm_scores)
    # results["svm"]["tuned"]["params"] = svm_tuned_params
    results["gradient_boosting"]["tuned"]["times"] = cb_tuned_times
    results["gradient_boosting"]["untuned"]["times"] = cb_untuned_times
    results["gradient_boosting"]["tuned"]["scores"] = cb_tuned_scores
    results["gradient_boosting"]["untuned"]["scores"] = cb_scores
    results["gradient_boosting"]["tuned"]["avg_score"] = np.mean(cb_tuned_scores)
    results["gradient_boosting"]["untuned"]["avg_score"] = np.mean(cb_scores)
    results["linear_models"]["tuned"]["times"] = lm_tuned_times
    results["linear_models"]["untuned"]["times"] = lm_untuned_times
    results["linear_models"]["tuned"]["scores"] = lm_tuned_scores
    results["linear_models"]["untuned"]["scores"] = lm_scores
    results["linear_models"]["tuned"]["avg_score"] = np.mean(lm_tuned_scores)
    results["linear_models"]["untuned"]["avg_score"] = np.mean(lm_scores)
    # results["linear_models"]["tuned"]["params"] = elastic_tuned_params

    # Store results in file
    with open("../../results/multiclass_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    evaluate_models()
