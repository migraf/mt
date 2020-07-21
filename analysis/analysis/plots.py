import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from process_data import load_data
from pandas_profiling import ProfileReport

def grouped_box_plot(data, cat_col, num_cols, title=None, y_title=None, x_title=None, points=True):
    """
    Create a box plot for a continous variable grouped by a categorical variable
    :param data:
    :type data:
    :param cat_col:
    :type cat_col:
    :param num_col:
    :type num_col:
    :return:
    :rtype:
    """
    groups = data.groupby(cat_col)
    fig = go.Figure()
    for variable in num_cols:
        trace_x = []
        trace_y = []
        for idx, group in groups:
            y = list(group[variable][group[variable].notnull()])
            x = [idx for i in range(len(y))]
            trace_y += y
            trace_x += x
        if points:
            fig.add_trace(go.Box(y=trace_y, x=trace_x, name=variable, boxpoints='all', jitter=0.2))
        else:
            fig.add_trace(go.Box(y=trace_y, x=trace_x, name=variable))

    fig.update_layout(
        title=title,
        title_x=0.5,
        yaxis_title=y_title,
        xaxis_title=x_title,
        boxmode='group'
    )
    fig.show()


def scatter_correlation_plot(data, var1, var2):
    """
    Plot two continous variables against each, also displaying the correlation p value
    :param data: pandas dataframe containing the data
    :type data:
    :param var1: continuous variable
    :type var1: str
    :param var2: continuous variable
    :type var2: str
    :return:
    :rtype:
    """
    selector = data[var1].notnull() & data[var2].notnull()
    x = data[var1][selector]
    y = data[var2][selector]
    colormap = (x.values + y.values) / np.linalg.norm((x.values + y.values))
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker_color=colormap
    ))

    stat, p = numerical_correlation(data, var1, var2, values=True)
    if p < 0.0001:
        p = "0.0000"
    else:
        p = np.round(p, decimals=4)
    print(str(p))
    fig.add_annotation(
        x=np.max(x),
        y=np.max(y),
        xref="x",
        yref="y",
        text="p={:.4}".format(p),
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        align="center",
        bordercolor="#13181a",
        borderwidth=2,
        borderpad=4,
        bgcolor="#3a6b94",
        opacity=0.8
    )
    fig.update_layout(
        title=f"{var1} vs {var2}",
        title_x=0.5,
        xaxis_title=var1,
        yaxis_title=var2
    )
    fig.show()


def grouped_plot_matrix(data, num_cols, cat_col):
    """
    Create a seaborn pairplot with hue based on the selected columns extracted from the given data
    :param data:
    :type data:
    :param num_cols:
    :type num_cols:
    :param cat_col:
    :type cat_col:
    :return:
    :rtype:
    """
    # TODO column mappings
    selected_cols = num_cols.copy()
    selected_cols.append(cat_col)
    plot_data = data[selected_cols]
    new_cols = [col[:10] for col in plot_data.columns]
    plot_data.columns = new_cols
    sns.set(style="ticks")
    sns.pairplot(plot_data, hue=cat_col[:10])
    # plt.tight_layout()
    plt.show()
    # return pair_plot


if __name__ == '__main__':
    df_sars = load_data(
        "D:\\hypothesis\\hypothesis\\server\\walzLabBackend\\flaskr\\user_data\\15052020SARS-CoV-2_final.xlsx")
    #
    # grouped_box_plot(df_sars, "III.12", ["VII.1A", "VII.2A", "VII.3A"], y_title="RBD Peptid Rekombinant",
    #                  title="RBD Peptid Rekombinant for loss of taste/smell")
    # scatter_correlation_plot(df_sars, "VII.1A", "VII.2A")
    print(list(df_sars.columns))
    print(df_sars.info())
    # plot = sns.pairplot(df, hue="species")
    # plot.show()
    num_cols = ['III.9: Wenn ja, bis zu welcher maximalenTemperatur?', 'VII.2A: OD IgM RBD Peptid rekombinant',
                'VII.1A: OD IgG RBD Peptid rekombinant', 'VII.3A: OD IgA RBD Peptid rekombinant']
    grouped_plot_matrix(df_sars,
                        num_cols=num_cols,
                        cat_col='III.6: Haben Sie sich krank gefühlt?')
