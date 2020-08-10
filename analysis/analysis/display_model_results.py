import plotly.graph_objects as go
import json
import numpy as np
import statsmodels.api as sm


def display_model_performance(predictor, pred_type, test_data, test_target, target_name):


    # Generate different results based on prediction type
    if pred_type == "regression":
        display_regression_results(predictor, test_data, test_target, target_name)
    elif pred_type == "binary":
        display_binary_classification_results(predictor, test_data, test_target, target_name)
    elif pred_type == "multi-class":
        display_mc_classification_results(predictor, test_data, test_target, target_name)



def display_regression_results(pred, test_data, test_target, target_name):

    # Make prediction on given test set and display the predictions visually
    # Output the score of the predictor on the given test set
    # TODO some more printed model descriptions
    score = pred.score(test_data, test_target)
    print(f"Score: {score}")
    prediction = pred.predict(test_data)
    plot_regression_results(range(len(prediction)), prediction, test_target, target_name)


def display_binary_classification_results(pred, test_data, test_target, target_name):
    pass


def display_mc_classification_results(pred, test_data, test_target, target_name):
    pass


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

    # Scatter plot containing true and predicted values per patient
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=list(x), y=pred_y, name=f"Predicted {target}", mode="markers+lines"))
    fig_pred.add_trace(go.Scatter(x=list(x), y=y, name=f"True {target}", mode="markers+lines"))

    fig_pred.update_layout(
        title={
            'text': "Predictions of Regression Model compared to true Values in held out test set",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Samples",
        yaxis_title=f"{target}",
        font={
            "family": "Courier New, monospace",
            "size": 18}
    )
    fig_pred.show()

    # Fit regression line into model predictions
    fitted_line = sm.OLS(pred_y, sm.add_constant(y)).fit().fittedvalues
    print(fitted_line)

    fig_perf = go.Figure()
    y_pred_scaled = pred_y / np.linalg.norm(pred_y)
    y_scaled = y / np.linalg.norm(y)
    fig_perf.add_trace(go.Scatter(x=y_pred_scaled, y=y_scaled, mode="markers"))
    fig_perf.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines'))

    fig_perf.update_layout(
        title={
            'text': "Predictions of Regression Model compared to true Values in held out test set",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title=f"Predicted {target}",
        yaxis_title=f"True {target}",
        font={
            "family": "Courier New, monospace",
            "size": 18},
    )
    fig_perf.show()
