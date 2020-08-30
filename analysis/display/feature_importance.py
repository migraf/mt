import shap
import matplotlib.pyplot as plt
import numpy as np
import itertools
import textwrap



def display_feature_importances(model, data, shap_data, model_type=None, n_features=5, feature_interactions=None,
                                return_shap=True):
    # Get shap explainer and values based on model type
    if model_type == "tree":
        explainer, shap_values = calculate_shap_values(model, data, shap_data, model_type=model_type)
    else:
        explainer, shap_values = calculate_shap_values(model, data, shap_data)
    # Show summary plot
    shap.summary_plot(shap_values, shap_data)


    # For the most important features show dependance plots
    # get top features ( highest absolute shap value)
    top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))[:n_features]
    # print(list(combinations))
    for ind in top_inds:
        # shap.dependence_plot(comb[0], shap_values[0], data, interaction_index=comb[1])
        shap.dependence_plot(ind, shap_values, shap_data)
    shap_decision_plot(explainer.expected_value, shap_values, shap_data)
    if return_shap:
        return shap_values


def shap_dependance_plot(data, shap_values, feature, interaction_feature=None, save=True):

    if interaction_feature:
        fig = shap.dependence_plot(feature, shap_values, data)
        file_name = f"interaction_plot_{feature}.png"
    else:
        fig = shap.dependence_plot(feature, shap_values, data, interaction_index=interaction_feature)
        file_name = f"interaction_plot_{feature}_vs_{interaction_feature}"
    if save:
        fig.savefig(file_name)


def shap_decision_plot(expected_value, shap_values, samples, save=True, crop_feature_names=20):

    if crop_feature_names:
        feature_names = []
        for col in samples.columns:
            if len(col) > crop_feature_names:
                feature_names.append(col[:20])
            else:
                feature_names.append(col)

        shap.decision_plot(expected_value, shap_values, features=samples, feature_names=feature_names, show=False)
    else:
        shap.decision_plot(expected_value, shap_values, features=samples, show=False)
    f = plt.gcf()
    f.show()
    if save:
        f.savefig("shap_decision_plot.png")



def calculate_shap_values(model, train_data, shap_data, model_type="kernel"):
    # use faster tree explainer for tree based models
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        if len(train_data) > 100:
            explainer = shap.KernelExplainer(model, shap.kmeans(train_data, 100))
        else:
            explainer = shap.KernelExplainer(model, train_data)

    shap_values = explainer.shap_values(shap_data)
    return explainer, shap_values
