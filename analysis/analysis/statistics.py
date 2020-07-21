import pandas as pd
import numpy as np
import scipy.stats as ss
from analysis import *
from collections import Counter
import scikit_posthocs as sp
from itertools import combinations


class StatisticsError(Exception):
    """
    Custom Error to raise when something goes wrong calculating statistics
    """

    def __init__(self, message):
        super().__init__(message)


def conditional_entropy(x, y, nan_strategy="replace", nan_replace_value="0.0"):
    """
    Calcultate conditional entropy of x given y
    :param x: categorical values
    :type x: pd.Series
    :param y: categorical values
    :type y: pd.Series
    :param nan_stragy:
    :type nan_stragy:
    :param nan_replace_value:
    :type nan_replace_value:
    :return:
    :rtype:
    """
    if nan_strategy == "replace":
        x = x.fillna(nan_replace_value)
        y = y.fillna(nan_replace_value)

    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total
        p_y = y_counter[xy[1]] / total
        entropy += p_xy * np.log(p_y / p_xy)
    return entropy


def theils_u(x, y, nan_strategy="replace", nan_replace_value="0.0"):
    """
    Calculate Theil's U uncertainty coefficient for cross categorical association
    --> Uncertainty of x given y
    :param x: categorical series
    :type x: pd.Series
    :param y: categorical series
    :type y: pd.Series
    :param nan_strategy: How to handel nan values
    :type nan_strategy:
    :param nan_replace_value: default value for replacing nan values
    :type nan_replace_value:
    :return: Theils u uncertainty coefficient
    :rtype: float
    """
    # TODO add options for different na handling strategies
    if nan_strategy == "replace":
        x = x.fillna(nan_replace_value)
        y = y.fillna(nan_replace_value)

    s_xy = conditional_entropy(x, y, nan_strategy, nan_replace_value)
    x_counter = Counter(x)
    total = sum(x_counter.values())
    p_x = list(map(lambda n: n / total, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def calculate_categorical_correlation(df, selected_columns):
    """
    Calculate theil u for all categorical values against each other
    :param df:
    :type df:
    :param selected_columns:
    :type selected_columns:
    :return:
    :rtype:
    """
    corr = pd.DataFrame(index=selected_columns, columns=selected_columns)
    for i in range(len(selected_columns)):
        for j in range(len(selected_columns)):
            if i == j:
                corr.loc[selected_columns[i], selected_columns[j]] = 1.0
            else:
                ji = theils_u(df[selected_columns[i]], df[selected_columns[j]])
                ij = theils_u(df[selected_columns[j]], df[selected_columns[i]])

                corr.loc[selected_columns[i], selected_columns[j]] = ji if not np.isnan(ij) and np.abs(
                    ij) < np.inf else 0.0
                corr.loc[selected_columns[j], selected_columns[i]] = ij if not np.isnan(ij) and np.abs(
                    ij) < np.inf else 0.0
    return corr


def chi_2(df, col1, col2, notebook=True):
    """
    Calculate the chi2 contigency test between two columns in the given df
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html
    :param df: dataframe
    :param col1: first column to test
    :param col2: second column to test
    :return:
    """
    group_sizes = df.groupby([col1, col2]).size()
    ct_sum = group_sizes.unstack(col1)
    chi2, p, dof, _ = ss.chi2_contingency(ct_sum.fillna(0))
    if notebook:
        print(f"Chi2: {chi2}, p-Value: {p}, DoF: {dof}")
    return chi2, p, dof


def fishers_exact(df, col1, col2):
    """
    Calculate fishers exact test between the two columns,
    requires there to be only two options for both selected columns
    :param df:
    :type df:
    :param col1:
    :type col1:
    :param col2:
    :type col2:
    :return:
    :rtype:
    """
    if len(df[col1].unique()) != 2 or len(df[col2].unique()) != 2:
        raise StatisticsError("Too many categories for Fishers Exact Test, consider"
                              " using chi2 test")
    group_sizes = df.groupby([col1, col2]).size()
    ct_sum = group_sizes.unstack(col1)
    odds, p = ss.fisher_exact(ct_sum)
    return odds, p


def mann_whitney_u(df, col1, col2):
    """
    Perform the mann whitney u test between the selected columns of the dataframe
    https://de.wikipedia.org/wiki/Wilcoxon-Mann-Whitney-Test
    :param df:
    :param col1:
    :param col2:
    :return:
    """
    pass


def wilcoxon(df, cat_col, num_col):
    """
    Calculate the wilcoxon signed rank test for two samples
    :param df:
    :param num_col:
    :type num_col:
    :param cat_col:
    :type cat_col:
    :return:
    """
    groups = []
    for idx, group in df.groupby(cat_col):
        groups.append(group[num_col])
    print(len(groups[0]))
    print(len(groups[1]))
    wilcoxon_result = ss.wilcoxon(*groups)
    print(wilcoxon_result)


def kruskal_wallis(df, cat_col, num_col, notebook=True):
    """
    Perform kruskal wallis test between the selected columns of the given dataframe.
    Columns need to be continuous
    :param df:
    :param cat_group:
    :param num_col:
    :return:
    """
    variables = []
    for idx, cat_group in df.groupby(cat_col):
        # NAN values not included in computation
        variables.append(cat_group[num_col][cat_group[num_col].notnull()])
    kruskal_h, kruskal_p = ss.kruskal(*variables)
    if print:
        print(f"H-Value: {kruskal_h}, p-Value: {kruskal_p}")
    output = f"\tTest: Kruskal-Wallis\n"
    output += f"\tH-Value: {kruskal_h}, p-Value: {kruskal_p}\n"
    if kruskal_p <= 0.05:
        output += "\tSignificance found \n"
        output += "\tPost-Hoc Tests: Dunns with Bonferonni Correction\n"
        # Remove nan values
        selector = df[cat_col].notnull() & df[num_col].notnull()
        posthoc_data = df[selector]
        posthoc_result = sp.posthoc_dunn(posthoc_data, num_col, cat_col, p_adjust="bonferroni")
        if notebook:
            print(posthoc_result)
        output += str(posthoc_result)
        output += "\n"

    return output


def numerical_correlation(data, var1, var2, values=False):
    """
    Compute the correlation coefficient for two continours variables in the data, automatically checks if both variables
    are normally distributed and based on this outputs either Pearsons or Spearmans correlation coefficient
    :param data:
    :type data:
    :param var1:
    :type var1:
    :param var2:
    :type var2:
    :return:
    :rtype:
    """
    if values:
        selector = data[var1].notnull() & data[var2].notnull()
        pearson_r, pearson_p = ss.pearsonr(data[var1][selector], data[var2][selector])
        if values:
            return pearson_r, pearson_p

    # Check if variables are normally distributed
    ks_1, shapiro_p_1 = ss.shapiro(data[var1][data[var1].notnull()])
    ks_2, shapiro_p_2 = ss.shapiro(data[var2][data[var2].notnull()])
    output = ""
    # If both variables are assumed to be normally distributed use pearson correlation test
    if shapiro_p_1 > 0.05 and shapiro_p_2 > 0.05:

        output += "Test: Pearson Correlation: \n"
        selector = data[var1].notnull() & data[var2].notnull()
        pearson_r, pearson_p = ss.pearsonr(data[var1][selector], data[var2][selector])
        output += f"Pearson R: {pearson_r}, p-Value: {pearson_p} \n"
    # otherwise use Spearman Rank correlation test
    else:
        output += "Test: Spearman Correlation \n"
        selector = data[var1].notnull() & data[var2].notnull()
        rho, spearman_p = ss.spearmanr(data[var1][selector], data[var2][selector])
        output += f"Spearman Rho: {rho}, p-Value: {spearman_p} \n"

    return output


def test_battery(data, dict, variables, save=True):
    """
    Calculate appropriate test statistics for all combinations in the given list of variables and display the results
    and save the results to a file if desired.
    :param data: data which contains the values of the selected variables
    :param dict: dictionary containing descriptions of the variables
    :param variables: list of variables of interest
    :param save: boolean indicating wether to save results to file or not
    :return:
    """
    output = ""
    for combo in combinations(variables, 2):
        var1 = combo[0]
        var2 = combo[1]

        output += f"{var1} vs {var2}\n"
        # compare two categorical variables
        if data[var1].dtype == "object" and data[var2].dtype == "object":
            output += f"\tTest: Chi2 Goodness of Fit:\n"
            chi2, chi_p, dof = chi_2(data, var1, var2)
            output += f"\tChi2: {chi2}, p-Value: {chi_p}, DoF: {dof}\n"

        elif data[var1].dtype == "object" and data[var2].dtype == "float":
            output += kruskal_wallis(data, var1, var2, notebook=False)

        elif data[var1].dtype == "float" and data[var2].dtype == "object":

            output += kruskal_wallis(data, var2, var1)
        # If both variables are continuous calculate correlations
        elif data[var1].dtype == "float" and data[var2].dtype == "float":
            output += numerical_correlation(data, var1, var2)
        output += "\n"
    # Save as file if desired
    if save:
        with open("output.txt", "w") as f:
            f.write(output)

    return output


if __name__ == '__main__':
    df_sars, sars_dict = load_data(
        "C:\\hypothesis\\repositories\\server\\walzLabBackend\\flaskr\\user_data\\Datentabelle_CoVid19_SARS_new.xlsx",
        two_sheets=True)
    print(list(df_sars.columns))
    # print("Theils u:")
    # theils_coef = theils_u(df_sars["IX.2C"], df_sars["Geschlecht"])
    # corr = calculate_categorical_correlation(df, categorical_columns)
    # print(corr)
    chi2, chi_p, dof = chi_2(df_sars, "Geschlecht", "III.7")
    print(chi2, chi_p, dof)
    # odds, p = fishers_exact(df, "positive", "Geschlecht")
    # print(odds, p)
    # print(list(df_sars["III.7"]))
    # kruskal_columns = ["Geschlecht", "VII.2A"]
    # kruskal_wallis(df_sars, "Geschlecht", "VII.2A")
    print(kruskal_wallis(df_sars, "III.7", "VII.2A"))
    # wilcoxon(df_sars, "Geschlecht", "VII.1C")
    print(ss.shapiro(df_sars['VII.2A'][df_sars['VII.2A'].notnull()]))
    variables = ['VII.1A', 'VII.1B', 'VII.1C', 'VII.2A', 'VII.2B', 'VII.2C', 'Geschlecht',
                 'III.7', 'III.8', 'III.10', 'III.11', 'III.12', 'IX.1A', 'IX.2A', 'IX.1B', 'IX.2B', 'IX.1C', 'IX.2C']
    output = test_battery(df_sars, sars_dict, variables)
    print(numerical_correlation(df_sars, 'VII.1C', 'VII.2A'))
    # print(output)
