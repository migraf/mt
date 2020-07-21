import plotly.graph_objects as go
import json
import numpy as np

# Load model results

with open("binary_results.json", "r") as result:
    binary_results = json.load(result)
with open("binary_results_subset.json", "r") as result:
    binary_results_subset = json.load(result)
with open("multiclass_results.json", "r") as br:
    multiclass_result = json.load(br)
with open("regression_results.json", "r") as br:
    regression_result = json.load(br)
with open("regression_results_subset.json", "r") as br:
    regression_result_subset = json.load(br)

# Calculate Performance statistics
svm_acc = np.mean(binary_results_subset["svm"]["untuned"]["scores"])
svm_std = np.std(binary_results_subset["svm"]["untuned"]["scores"])
svm_time = np.mean(binary_results_subset["svm"]["untuned"]["times"])
svm_time_std = np.std(binary_results_subset["svm"]["untuned"]["times"])
print(f'SVM Binary - Accuracy: {svm_acc}, std: {svm_std} - Time: {svm_time} std: {svm_time_std}')
svm_acc = np.mean(binary_results_subset["svm"]["tuned"]["scores"])
svm_std = np.std(binary_results_subset["svm"]["tuned"]["scores"])
svm_time = np.mean(binary_results_subset["svm"]["tuned"]["times"])
svm_time_std = np.std(binary_results_subset["svm"]["tuned"]["times"])
print(f'SVM Binary tuned - Accuracy: {svm_acc}, std: {svm_std} - Time: {svm_time} std: {svm_time_std}')
svm_acc = np.mean(binary_results_subset["random_forest"]["untuned"]["scores"])
svm_std = np.std(binary_results_subset["random_forest"]["untuned"]["scores"])
svm_time = np.mean(binary_results_subset["random_forest"]["untuned"]["times"])
svm_time_std = np.std(binary_results_subset["random_forest"]["untuned"]["times"])
print(f'RF Binary - Accuracy: {svm_acc}, std: {svm_std} - Time: {svm_time} std: {svm_time_std}')
svm_acc = np.mean(binary_results_subset["random_forest"]["tuned"]["scores"])
svm_std = np.std(binary_results_subset["random_forest"]["tuned"]["scores"])
svm_time = np.mean(binary_results_subset["random_forest"]["tuned"]["times"])
svm_time_std = np.std(binary_results_subset["random_forest"]["tuned"]["times"])
print(f'RF Binary tuned - Accuracy: {svm_acc}, std: {svm_std} - Time: {svm_time} std: {svm_time_std}')
svm_acc = np.mean(binary_results_subset["gradient_boosting"]["untuned"]["scores"])
svm_std = np.std(binary_results_subset["gradient_boosting"]["untuned"]["scores"])
svm_time = np.mean(binary_results_subset["gradient_boosting"]["untuned"]["times"])
svm_time_std = np.std(binary_results_subset["gradient_boosting"]["untuned"]["times"])
print(f'GB Binary - Accuracy: {svm_acc}, std: {svm_std} - Time: {svm_time} std: {svm_time_std}')
svm_acc = np.mean(binary_results_subset["gradient_boosting"]["tuned"]["scores"])
svm_std = np.std(binary_results_subset["gradient_boosting"]["tuned"]["scores"])
svm_time = np.mean(binary_results_subset["gradient_boosting"]["tuned"]["times"])
svm_time_std = np.std(binary_results_subset["gradient_boosting"]["tuned"]["times"])
print(f'GB Binary tuned - Accuracy: {svm_acc}, std: {svm_std} - Time: {svm_time} std: {svm_time_std}')


# Create overview plot
fig = go.Figure()

# SVM results
svm_results_y = binary_results["svm"]["untuned"]["scores"] + binary_results_subset["svm"]["untuned"]["scores"] + \
                multiclass_result["svm"]["untuned"]["scores"] + regression_result["svm"]["untuned"]["scores"] +\
                regression_result_subset["svm"]["untuned"]["scores"]

x = ["Binary" for i in range(50)] + ["Binary (subset)" for i in range(50)] + ["Multi-Class" for i in range(50)] + \
    ["Regression" for i in range(50)] + ["Regression (subset)" for i in range(50)]

fig.add_trace(go.Box(y=svm_results_y, x=x, name="SVM default"))

svm_results_tuned_y = binary_results["svm"]["tuned"]["scores"] + binary_results_subset["svm"]["tuned"]["scores"] + \
                      multiclass_result["svm"]["tuned"]["scores"] + regression_result["svm"]["tuned"]["scores"] + \
                      regression_result_subset["svm"]["tuned"]["scores"]

fig.add_trace(go.Box(y=svm_results_tuned_y, x=x, name="SVM tuned"))

# Random Forest Results
rf_results_y = binary_results["random_forest"]["untuned"]["scores"] + \
               binary_results_subset["random_forest"]["untuned"]["scores"] + \
               multiclass_result["random_forest"]["untuned"]["scores"] + \
               regression_result["random_forest"]["untuned"]["scores"] + \
               regression_result_subset["random_forest"]["untuned"]["scores"]

fig.add_trace(go.Box(y=rf_results_y, x=x, name="RF default"))

rf_results_tuned_y = binary_results["random_forest"]["tuned"]["scores"] + \
                     binary_results_subset["random_forest"]["tuned"]["scores"] + \
                     multiclass_result["random_forest"]["tuned"]["scores"] + \
                     regression_result["random_forest"]["tuned"]["scores"] + \
                     regression_result_subset["random_forest"]["tuned"]["scores"]

fig.add_trace(go.Box(y=rf_results_tuned_y, x=x, name="RF tuned"))

# Gradient Boosting Results
gb_results_y = binary_results["gradient_boosting"]["untuned"]["scores"] + \
               binary_results_subset["gradient_boosting"]["untuned"]["scores"] + \
               multiclass_result["gradient_boosting"]["untuned"]["scores"] + \
               regression_result["gradient_boosting"]["untuned"]["scores"] + \
               regression_result_subset["gradient_boosting"]["untuned"]["scores"]

fig.add_trace(go.Box(y=gb_results_y, x=x, name="GB default"))

gb_results_tuned_y = binary_results["gradient_boosting"]["tuned"]["scores"] + \
                     binary_results_subset["gradient_boosting"]["tuned"]["scores"] + \
                     multiclass_result["gradient_boosting"]["tuned"]["scores"] + \
                     regression_result["gradient_boosting"]["tuned"]["scores"] + \
                     regression_result_subset["gradient_boosting"]["tuned"]["scores"]

fig.add_trace(go.Box(y=gb_results_tuned_y, x=x, name="GB tuned"))

fig.add_trace(go.Box(y=regression_result["elastic_net"]["tuned"]["scores"] +
                       regression_result_subset["elastic_net"]["tuned"]["scores"],
                     x=["Regression" for i in range(50)] + ["Regression (subset)" for i in range(50)],
                     name="ElasticNet Tuned"))

fig.add_trace(go.Box(y=regression_result["elastic_net"]["untuned"]["scores"] +
                       regression_result_subset["elastic_net"]["untuned"]["scores"],
                     x=["Regression" for i in range(50)] + ["Regression (subset)" for i in range(50)],
                     name="ElasticNet"))

fig.update_layout(
    title="Model Performance",
    title_x=0.5,
    yaxis_title="Accuracy/R^2",
    boxmode='group'
)

# fig.show()
