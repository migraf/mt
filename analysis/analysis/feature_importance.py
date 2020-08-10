from analysis import *
import shap
import matplotlib.pyplot as plt
import numpy as np
import itertools
shap.initjs()


def display_feature_importances(model, data, shap_data, model_type=None, n_features=5, feature_interactions=None):
    # Get shap explainer and values based on model type
    if model_type == "tree":
        explainer, shap_values = calculate_shap_values(model, data, shap_data, model_type=model_type)
    else:
        explainer, shap_values = calculate_shap_values(model, data, shap_data)
    # Show summary plot
    shap.summary_plot(shap_values, shap_data)

    # display dependance plots either between specified features or the most important ones
    if feature_interactions is not None:
        # TODO display interaction plot for the selected variables
        pass
    else:
        # For the most important features show dependance plots
        # get top features ( highest absolute shap value)
        top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))[:n_features]
        combinations = itertools.combinations(top_inds, 2)
        print(list(combinations))
        for comb in combinations:
            shap.dependence_plot(comb[0], shap_values, interaction_index=comb[1])



def calculate_shap_values(model, train_data, shap_data, model_type="kernel"):
    # use faster tree explainer for tree based models
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model, train_data)

    shap_values = explainer.shap_values(shap_data)
    return explainer, shap_values
