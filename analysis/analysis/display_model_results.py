import plotly.graph_objects as go
import json
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import textwrap
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff


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
    plot_regression_results(x=range(len(prediction)),
                            y=test_target,
                            pred_y=prediction,
                            score=score,
                            target=target_name)


def display_binary_classification_results(pred, test_data, test_target, target_name=None, pos_class=None):
    score = pred.score(test_data, test_target)
    print(f"Score: {score}")
    if not pos_class:
        if "1" in pred.classes_:
            pos_class = "1"
        else:
            pos_class = pred.classes_[0]

    prediction_proba = pred.predict_proba(test_data)
    pos_proba = prediction_proba[:, list(pred.classes_).index(pos_class)]
    plot_binary_results(test_target, pos_proba, pos_class, target_name)



def display_mc_classification_results(pred, test_data, test_target, target_name):
    score = pred.score(test_data, test_target)
    print(f"Score: {score}")

    prediction = pred.predict(test_data)
    labels = list(test_target.unique())
    plot_multi_class_results(test_target, prediction, target_name, labels)


def plot_multi_class_results(target, prediction, target_name, labels=None, save=True):
    if not labels:
        labels = sorted(list(target.unique()))
    conf_matrix = confusion_matrix(target, prediction, sorted(labels))
    conf_matrix_norm = confusion_matrix(target, prediction, sorted(labels), normalize="true")

    fig_hm = ff.create_annotated_heatmap(z=conf_matrix_norm, x=sorted(labels), y=sorted(labels), annotation_text=conf_matrix)

    layout = dict(
        title={
            'text': f"Confusion matrix for predicting {target_name}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font={
            "family": "Courier New, monospace",
            "size": 18})

    fig_hm.update_layout(layout)
    fig_hm['layout']['xaxis']['side'] = 'bottom'

    fig_hm.show()
    if save:
        fig_hm.write_image("confusion_matrix.png", height=1000, width=1200, scale=2)


def plot_binary_results(y, proba, pos_class, target, save=True):
    fpr, tpr, _ = roc_curve(y, proba, pos_label=pos_class)
    auc_score = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, line=dict(color='red'), showlegend=False))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='royalblue', width=4, dash='dash'),
                                 showlegend=False))
    layout = dict(
        title={
            'text': f"Roc curve for predicting {target} (positive = {pos_class})",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title=f"False positive rate",
        yaxis_title=f"True positive rate",
        font={
            "family": "Courier New, monospace",
            "size": 18},
        annotations=[
            go.layout.Annotation(
                text=f'AUC: {round(auc_score, 4)}',
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.98,
                y=0.02,
                bordercolor='black',
                borderwidth=1
            )
        ]
    )
    fig_roc.update_layout(layout)
    fig_roc.show()

    if save:
        fig_roc.write_image("roc_plot.png", height=800, width=1200, scale=2)


def plot_regression_results(x, y, pred_y, target, score, save=True):
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
    fig_pred.add_trace(go.Scatter(x=list(x), y=y,
                                  name="<br>".join(textwrap.wrap(f"True {target}", 30)),
                                  mode="markers+lines"))
    fig_pred.add_trace(go.Scatter(x=list(x), y=pred_y,
                                  name="<br>".join(textwrap.wrap(f"Predicted {target}", 30)),
                                  mode="markers+lines"))


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

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=y, y=pred_y, mode="markers", name="Predicted vs True"))
    fig_reg.add_trace(go.Scatter(x=y, y=fitted_line, mode="lines", showlegend=False))

    layout = dict(
        xaxis_title=f"True {target}",
        yaxis_title=f"Predicted {target}",
        font={
            "family": "Courier New, monospace",
            "size": 18},
        annotations=[
            go.layout.Annotation(
                text=f'Score: {round(score, 4)}',
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.98,
                y=0.02,
                bordercolor='black',
                borderwidth=1
            )
        ]
    )
    fig_reg.layout = layout

    fig_reg.show()

    if save:
        fig_pred.write_image("reg_predictions.png", height=800, width=1200, scale=2)

        fig_reg.write_image("reg_perfomance.png", height=800, width=1200, scale=2)


def plot_confusion_matrix(y, pred_y, labels):
    pass
