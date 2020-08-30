import plotly.graph_objects as go
import json
import numpy as np

# Load model results

with open("../../results/binary_results.json", "r") as result:
    binary_results = json.load(result)
with open("../../results/multiclass_results.json", "r") as br:
    multiclass_result = json.load(br)
with open("../../results/regression_results.json", "r") as br:
    regression_result = json.load(br)

with open("../../results/regression_results_subset.json", "r") as br:
    regression_result_subset = json.load(br)

def print_summary_statistics(results):
    # Calculate Performance statistics
    svm_acc = np.mean(results["svm"]["untuned"]["scores"])
    svm_std = np.std(results["svm"]["untuned"]["scores"])
    svm_time = np.mean(results["svm"]["untuned"]["times"])
    svm_time_std = np.std(results["svm"]["untuned"]["times"])
    print(f'SVM - Accuracy: {round(svm_acc, 3)}, std: {round(svm_std, 3)} - Time: {round(svm_time,3)}'
          f' std: {round(svm_time_std,3)}')
    print(f"& {round(svm_acc, 3)} & {round(svm_std, 3)} & {round(svm_time,3)}")
    svm_acc = np.mean(results["svm"]["tuned"]["scores"])
    svm_std = np.std(results["svm"]["tuned"]["scores"])
    svm_time = np.mean(results["svm"]["tuned"]["times"])
    svm_time_std = np.std(results["svm"]["tuned"]["times"])
    print(f'SVM tuned- Accuracy: {round(svm_acc, 3)}, std: {round(svm_std, 3)} - Time: {round(svm_time, 3)}'
          f' std: {round(svm_time_std, 3)}')
    print(f"& {round(svm_acc, 3)} & {round(svm_std, 3)} & {round(svm_time, 3)}")
    svm_acc = np.mean(results["random_forest"]["untuned"]["scores"])
    svm_std = np.std(results["random_forest"]["untuned"]["scores"])
    svm_time = np.mean(results["random_forest"]["untuned"]["times"])
    svm_time_std = np.std(results["random_forest"]["untuned"]["times"])
    print(f'RF - Accuracy: {round(svm_acc, 3)}, std: {round(svm_std, 3)} - Time: {round(svm_time, 3)}'
          f' std: {round(svm_time_std, 3)}')
    print(f"& {round(svm_acc, 3)} & {round(svm_std, 3)} & {round(svm_time, 3)}")
    svm_acc = np.mean(results["random_forest"]["tuned"]["scores"])
    svm_std = np.std(results["random_forest"]["tuned"]["scores"])
    svm_time = np.mean(results["random_forest"]["tuned"]["times"])
    svm_time_std = np.std(results["random_forest"]["tuned"]["times"])
    print(f'RF tuned- Accuracy: {round(svm_acc, 3)}, std: {round(svm_std, 3)} - Time: {round(svm_time, 3)}'
          f' std: {round(svm_time_std, 3)}')
    print(f"& {round(svm_acc, 3)} & {round(svm_std, 3)} & {round(svm_time, 3)}")
    svm_acc = np.mean(results["gradient_boosting"]["untuned"]["scores"])
    svm_std = np.std(results["gradient_boosting"]["untuned"]["scores"])
    svm_time = np.mean(results["gradient_boosting"]["untuned"]["times"])
    svm_time_std = np.std(results["gradient_boosting"]["untuned"]["times"])
    print(f'GB - Accuracy: {round(svm_acc, 3)}, std: {round(svm_std, 3)} - Time: {round(svm_time, 3)}'
          f' std: {round(svm_time_std, 3)}')
    print(f"& {round(svm_acc, 3)} & {round(svm_std, 3)} & {round(svm_time, 3)}")
    svm_acc = np.mean(results["gradient_boosting"]["tuned"]["scores"])
    svm_std = np.std(results["gradient_boosting"]["tuned"]["scores"])
    svm_time = np.mean(results["gradient_boosting"]["tuned"]["times"])
    svm_time_std = np.std(results["gradient_boosting"]["tuned"]["times"])
    print(f'GB tuned - Accuracy: {round(svm_acc, 3)}, std: {round(svm_std, 3)} - Time: {round(svm_time, 3)}'
          f' std: {round(svm_time_std, 3)}')
    print(f"& {round(svm_acc, 3)} & {round(svm_std, 3)} & {round(svm_time, 3)}")
    svm_acc = np.mean(results["linear_models"]["untuned"]["scores"])
    svm_std = np.std(results["linear_models"]["untuned"]["scores"])
    svm_time = np.mean(results["linear_models"]["untuned"]["times"])
    svm_time_std = np.std(results["linear_models"]["untuned"]["times"])
    print(f'LM - Accuracy: {round(svm_acc, 3)}, std: {round(svm_std, 3)} - Time: {round(svm_time, 3)}'
          f' std: {round(svm_time_std, 3)}')
    print(f"& {round(svm_acc, 3)} & {round(svm_std, 3)} & {round(svm_time, 3)}")
    svm_acc = np.mean(results["linear_models"]["tuned"]["scores"])
    svm_std = np.std(results["linear_models"]["tuned"]["scores"])
    svm_time = np.mean(results["linear_models"]["tuned"]["times"])
    svm_time_std = np.std(results["linear_models"]["tuned"]["times"])
    print(f'LM tuned - Accuracy: {round(svm_acc, 3)}, std: {round(svm_std, 3)} - Time: {round(svm_time, 3)}'
          f' std: {round(svm_time_std, 3)}')
    print(f"& {round(svm_acc, 3)} & {round(svm_std, 3)} & {round(svm_time, 3)}")



def plot_model_results():

    # Create overview plot
    fig = go.Figure()

    # SVM results
    svm_results_y = binary_results["svm"]["untuned"]["scores"] + multiclass_result["svm"]["untuned"]["scores"] + \
                    regression_result["svm"]["untuned"]["scores"]

    x = ["Binary" for i in range(20)] + ["Multi-Class" for i in range(20)] + ["Regression" for i in range(50)]

    fig.add_trace(go.Box(y=svm_results_y, x=x, name="SVM default"))

    svm_results_tuned_y = binary_results["svm"]["tuned"]["scores"] + multiclass_result["svm"]["tuned"]["scores"] + \
                          regression_result["svm"]["tuned"]["scores"]

    fig.add_trace(go.Box(y=svm_results_tuned_y, x=x, name="SVM tuned"))

    # Random Forest Results
    rf_results_y = binary_results["random_forest"]["untuned"]["scores"] + \
                   multiclass_result["random_forest"]["untuned"]["scores"] + \
                   regression_result["random_forest"]["untuned"]["scores"]

    fig.add_trace(go.Box(y=rf_results_y, x=x, name="RF default"))

    rf_results_tuned_y = binary_results["random_forest"]["tuned"]["scores"] + \
                         multiclass_result["random_forest"]["tuned"]["scores"] + \
                         regression_result["random_forest"]["tuned"]["scores"]

    fig.add_trace(go.Box(y=rf_results_tuned_y, x=x, name="RF tuned"))

    # Gradient Boosting Results
    gb_results_y = binary_results["gradient_boosting"]["untuned"]["scores"] + \
                   multiclass_result["gradient_boosting"]["untuned"]["scores"] + \
                   regression_result["gradient_boosting"]["untuned"]["scores"]

    fig.add_trace(go.Box(y=gb_results_y, x=x, name="GB default"))

    gb_results_tuned_y = binary_results["gradient_boosting"]["tuned"]["scores"] + \
                         multiclass_result["gradient_boosting"]["tuned"]["scores"] + \
                         regression_result["gradient_boosting"]["tuned"]["scores"]

    fig.add_trace(go.Box(y=gb_results_tuned_y, x=x, name="GB tuned"))


    gb_results_y = binary_results["linear_models"]["untuned"]["scores"] + \
                   multiclass_result["linear_models"]["untuned"]["scores"] + \
                   regression_result["linear_models"]["untuned"]["scores"]

    fig.add_trace(go.Box(y=gb_results_y, x=x, name="Linear Model default"))

    gb_results_tuned_y = binary_results["linear_models"]["tuned"]["scores"] + \
                         multiclass_result["linear_models"]["tuned"]["scores"] + \
                         regression_result["linear_models"]["tuned"]["scores"]

    fig.add_trace(go.Box(y=gb_results_tuned_y, x=x, name="Linear Model tuned"))


    fig.update_layout(
        title="Model Performance",
        title_x=0.5,
        yaxis_title="Accuracy/R²-Score",
        boxmode='group',
        font_size=22

    )

    fig.show()
    fig.write_image("model_performance.png", width=2000, height=1000)

if __name__ == '__main__':
    # plot_model_results()
    print_summary_statistics(regression_result_subset)
