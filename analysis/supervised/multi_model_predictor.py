import catboost
import plotly.graph_objects as go
from scipy import stats
from time import time
from supervised import linear_model, svm, random_forest, gradient_boosted_trees
from util import detect_prediction_type, create_training_data, load_data
from display import display_model_performance, display_feature_importances


def multi_model_predictor(data, target, excluded_variables=[], prediction_type=None, linear_model_params=None,
                          svm_params=None, random_forest_params=None, gradient_boosting_params=None, cv=True,
                          display=True, shap=True, prepare_data=True, all_models=True):
    if prediction_type:
        model_subtype = prediction_type
    else:
        model_subtype = detect_prediction_type(data, target)

    if prepare_data:
        x_train, x_test, y_train, y_test, train_ind, test_ind = create_training_data(data, target, excluded_variables,
                                                                                     test_train_indices=True)
    else:
        x_train, x_test = data[0], data[1]
        y_train, y_test = target[0], target[1]

    # Extract data for catboost pool
    # TODO

    # create the models
    print("Training models")
    if linear_model_params:
        lin_m = linear_model([x_train, x_test], [y_train, y_test], prediction_type=model_subtype, cv=cv, display=False,
                         shap=False, prepare_data=False, **linear_model_params)
    else:
        lin_m = linear_model([x_train, x_test], [y_train, y_test], prediction_type=model_subtype, cv=cv, display=False,
                         shap=False, prepare_data=False)

    if svm_params:
        svm_m = svm([x_train, x_test], [y_train, y_test], prediction_type=model_subtype, cv=cv, display=False,
                    shap=False, prepare_data=False, **svm_params)
    else:
        svm_m = svm([x_train, x_test], [y_train, y_test], prediction_type=model_subtype, cv=cv, display=False,
                    shap=False, prepare_data=False)

    if random_forest_params:
        rf_m = random_forest([x_train, x_test], [y_train, y_test], prediction_type=model_subtype, cv=cv, display=False,
                             shap=False, prepare_data=False, **random_forest_params)
    else:
        rf_m = random_forest([x_train, x_test], [y_train, y_test], prediction_type=model_subtype, cv=cv, display=False,
                             shap=False, prepare_data=False)
    if gradient_boosting_params:
        gb_m, gb_m_score = gradient_boosted_trees(data=data, target=target, prediction_type=model_subtype, cv=cv,
                                                  display=False, shap=False, **gradient_boosting_params,
                                                  score=True)
    else:
        gb_m, gb_m_score = gradient_boosted_trees(data=data, target=target, prediction_type=model_subtype, cv=cv,
                                                  display=False, shap=False,
                                                  score=True)
    # Display scores
    lm_score = lin_m.score(x_test, y_test)
    print(f"Linear model score: {lm_score}")

    rf_score = rf_m.score(x_test, y_test)
    print(f"Random forest model score: {rf_score}")

    svm_score = svm_m.score(x_test, y_test)
    print(f"SVM model score: {svm_score}")

    print(f"Catboost model score: {gb_m_score}")

    models = [("linear_model", lm_score, lin_m), ("random forest", rf_score, rf_m), ("svm", svm_score, svm_m),
              ("catboost", gb_m_score, gb_m)]

    top_model = sorted(models, key= lambda x: x[1])[-1]
    print(top_model)
    top_pred = top_model[2]
    if display:
        display_model_performance(top_pred, model_subtype, x_test, y_test, target)
        if top_model[0] in {"linear_model", "svm"}:
            if model_subtype == "regression":
                shap_values = display_feature_importances(top_pred.predict, x_train, x_test, return_shap=shap)
            else:
                shap_values = display_feature_importances(top_pred.predict_proba, x_train, x_test, return_shap=shap)
        else:
            shap_values = display_feature_importances(top_pred, x_train, x_test, model_type="tree", return_shap=shap)
    if shap:
        return top_pred, models, shap_values
    else:
        return top_pred, models





class MultiModelPredictor:
    def __init__(self, data, num_cols, cat_cols, target, mode=None):
        self.data = data
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target = target
        self.mode = mode
        self.detect_prediction_mode()

    def detect_prediction_mode(self):
        """
        Detect the type of prediction based on the give target variable
        """
        # If mode is not set detect it based on the target variable
        if self.mode is None:
            if self.data[self.target].dtype.name in ["float64", "Int64"]:
                # Detect binary integer targets and s
                if len(self.data[self.target].unique()) == 2:
                    self.mode = "binary"
                # Otherwise set the mode to regression
                else:
                    self.mode = "regression"
            else:
                # if target is not numeric check if it only has two unique values
                if len(self.data[self.target].unique()) == 2:
                    self.mode = "binary"
                else:
                    self.mode = "multi-class"
        print(f"Prediction mode is {self.mode}")

    def train_models(self, tuning=False, verbose=True, test_set=True):
        """
        Train all available models that fit the selected target and store the results

        Parameters
        ----------
        tuning : bool indicating whether to perform cross validated hyper parameter tuning
        verbose : bool indicating the level of output the training generates
        test_set : bool setting wether to split the data into a training and testing set

        Returns
        -------

        """
        if test_set:
            x_train, x_test, y_train, y_test = create_training_data(self.data,
                                                                    self.num_cols,
                                                                    self.cat_cols,
                                                                    self.target)
            # TODO add gradient boosting models
            # Train regression models
            predictions = []
            if self.mode == "regression":
                if verbose:
                    print("Training ElasticNet model")
                    elastic_net_model = elastic_net(train_data=x_train, train_labels=y_train)
                    print(f"Elasticnet result: {elastic_net_model.score(x_test, y_test)}")
                    print("Training SVM model")
                    svm_regressor = svm_regression(train_data=x_train, train_labels=y_train)
                    print(f"SVM model result: {svm_regressor.score(x_test, y_test)}")
                    print(f"Training Random Forest Regressor")
                    rf_regressor = random_forest_regressor(train_data=x_train, train_labels=y_train)
                    print(f"Random Forest Results: {rf_regressor.score(x_test, y_test)}")


                else:
                    elastic_net_model = elastic_net(train_data=x_train, train_labels=y_train)
                    svm_regressor = svm_regression(train_data=x_train, train_labels=y_train)
                    rf_regressor = random_forest_regressor(train_data=x_train, train_labels=y_train)

                predictions.append(("elastic_net", elastic_net_model.predict(x_test)))
                predictions.append(("svm", svm_regressor.predict(x_test)))
                predictions.append(("random forest", rf_regressor.predict(x_test)))

            # Binary prediction models
            elif self.mode == "binary":
                if verbose:
                    print("Training Logistic regression classifier")
                    logreg_clf = logistic_regression(train_data=x_train, train_labels=y_train)
                    print(f"SVM classifier result: {logreg_clf.score(x_test, y_test)}")
                    print("Training SVM classifier")
                    svm_clf = svm_classifier(train_data=x_train, train_labels=y_train)
                    print(f"SVM classifier result: {svm_clf.score(x_test, y_test)}")
                    print("Training RF Classifier")
                    rf_clf = random_forest_classifier(train_data=x_train, train_labels=y_train)
                    print(f"RF classifier score: {rf_clf.score(x_test, y_test)}")
                else:
                    logreg_clf = logistic_regression(train_data=x_train, train_labels=y_train)
                    svm_clf = svm_classifier(train_data=x_train, train_labels=y_train)
                    rf_clf = random_forest_classifier(train_data=x_train, train_labels=y_train)
                predictions.append(("logistic regression", logreg_clf.predict(x_test)))
                predictions.append(("svm", svm_clf.predict(x_test)))
                predictions.append(("random forest", rf_clf.predict(x_test)))

            elif self.mode == "multi-class":
                if verbose:
                    print("Training logistic regression")
                    logreg_clf = logistic_regression(train_data=x_train, train_labels=y_train)
                    print(f"SVM classifier result: {logreg_clf.score(x_test, y_test)}")
                    print("Training SVM classifier")
                    svm_clf = svm_classifier(train_data=x_train, train_labels=y_train)
                    print(f"SVM classifier result: {svm_clf.score(x_test, y_test)}")
                    print("Training RF Classifier")
                    rf_clf = random_forest_classifier(train_data=x_train, train_labels=y_train)
                    print(f"RF classifier score: {rf_clf.score(x_test, y_test)}")
                else:
                    logreg_clf = logistic_regression(train_data=x_train, train_labels=y_train)
                    svm_clf = svm_classifier(train_data=x_train, train_labels=y_train)
                    rf_clf = random_forest_classifier(train_data=x_train, train_labels=y_train)
                predictions.append(("logistic regression", logreg_clf.predict(x_test)))
                predictions.append(("svm", svm_clf.predict(x_test)))
                predictions.append(("random forest", rf_clf.predict(x_test)))

            self.display_results(predictions, y_test, self.mode)

    def display_results(self, predictions, true, mode, save=True):
        """

        Parameters
        ----------
        save :

        Returns
        -------

        """
        if mode == "regression":
            self.plot_regression_results(predictions, true, save)

    def plot_regression_results(self, predictions, true, save):

        # Scatter plot displaying the predictions of all models
        fig_pred = go.Figure()
        x = range(len(predictions[0][1]))
        fig_pred.add_trace(go.Scatter(x=list(x), y=true, name=f"True {self.target}", mode="markers+lines"))
        for pred in predictions:
            fig_pred.add_trace(go.Scatter(x=list(x), y=pred[1], name=f"{pred[0]} prediction", mode="markers+lines"))

        fig_pred.update_layout(
            title="Predictions of Regression Model compared to true Values in held out test set",
            xaxis_title="Patients",
            yaxis_title=f"{self.target}",
            font={
                "family": "Courier New, monospace",
                "size": 18}
        )
        fig_pred.show()

        fig_perf = go.Figure()
        # fig_perf.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines'))
        for pred in predictions:
            fig_perf.add_trace(go.Scatter(x=true,
                                          y=pred[1],
                                          name=pred[0],
                                          mode="markers"))

        fig_perf.show()

        # TODO add r2 scores to title of traces

        if save:
            fig_pred.write_image("prediction_scatter.png")


if __name__ == '__main__':
    df_sars = load_data("../../datasets/walz_data.csv", na_values=["<NA>"])

    excluded_variables = ['Patienten-ID']

    print("Linear models main")
    binary_target = "Überhaput Antikörperantwort 0=nein"
    multi_target = "III.6: Haben Sie sich krank gefühlt?"
    # df_sars[multi_target] = df_sars[multi_target].astype("category")
    regr_target = "VII.1A: OD IgG RBD Peptid rekombinant"

    pred = multi_model_predictor(data=df_sars, target=regr_target, cv=False)
