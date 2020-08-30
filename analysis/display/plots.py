import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import plotly.express as px
import statsmodels.api as sm
from statistics import numerical_correlation


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
        # Wrap the plot titles
        if len(variable) > 30:
            display_var = "<br>".join(textwrap.wrap(variable, width=30))
        else:
            display_var = variable

        if points:
            fig.add_trace(go.Box(y=trace_y, x=trace_x, name=display_var, boxpoints='all', jitter=0.2))
        else:
            fig.add_trace(go.Box(y=trace_y, x=trace_x, name=display_var))

    fig.update_layout(
        title=title,
        title_x=0.5,
        yaxis_title=y_title,
        xaxis_title=cat_col,
        boxmode='group',
        width=1000
    )
    if x_title:
        fig.update_layout(
            xaxis_title=x_title,
        )

    legend = dict(
        yanchor="top",
        y=0.99,
        x=1.02
    )
    fig.update_layout(legend=legend)

    fig.write_image("grouped_box_plot.png")

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
        marker_color=colormap,
        showlegend=False
    ))
    fitted_line = sm.OLS(list(y), sm.add_constant(list(x))).fit().fittedvalues
    fig.add_trace(go.Scatter(x=x, y=fitted_line, line=dict(color="red"), showlegend=False))

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


def grouped_plot_matrix(data, num_vars, cat_var):
    """
    Create a seaborn pairplot with hue based on the selected columns extracted from the given data
    :param data:
    :type data:
    :param num_vars:
    :type num_vars:
    :param cat_var:
    :type cat_var:
    :return:
    :rtype:
    """
    selected_cols = num_vars.copy()
    selected_cols.append(cat_var)
    plot_data = data[selected_cols]
    # TODO find a smarter way to shorten variable names
    new_cols = [col[:10] for col in plot_data.columns]
    plot_data.columns = new_cols
    sns.set(style="ticks")
    sns.pairplot(plot_data, hue=cat_var[:10])
    # plt.tight_layout()
    plt.show()
    # return pair_plot


def regression_scatter_matrix(data, variables):
    """
    Create a scatter matrix with fitted regression lines
    :param data: pandas dataframe containing the data to be analyzed
    :param variables: list of variable names
    :return:
    """
    plot_data = data[variables]

    plot_data = plot_data.dropna(how="any")
    # Wrap the column names for better plot layouting
    wrapped_cols = ["\n".join(textwrap.wrap(col, 30)) for col in plot_data.columns]
    plot_data.columns = wrapped_cols
    plot_data = plot_data.astype(np.float64)
    sns.set(style="ticks")
    sns.pairplot(plot_data, kind="reg", vars=wrapped_cols)
    plt.show()


def parallel_coordinates_plot(data, variables, group_var=None):
    """
    Create a parallel coordinates plot based on a given dataframe and a list of variable names
    Args:
        data: dataframe containing the data
        variables: list of variable names
        group_var: variable name to set the color scales for the created lines

    Returns:

    """

    # Remove na values
    plot_data_vars = variables.copy() + [group_var]

    print(plot_data_vars)
    plot_data = data[plot_data_vars].copy()
    plot_data = plot_data.dropna(how="any")

    # create dimensions for parallel
    dimensions = []

    for var in variables:
        label = "<br>".join(textwrap.wrap(var, 30))
        print(label)
        dim = dict(
            range=[plot_data[var].min(), plot_data[var].max()],
            label=label,
            values=plot_data[var]
        )
        dimensions.append(dim)

    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=plot_data[group_var],
                colorscale=px.colors.diverging.Geyser,
                showscale=True
            ),
            dimensions=dimensions
        )

    )

    fig.show()
    fig.update_layout(
        width=1200
    )
    fig.write_image("parallel_coordinates.png")


if __name__ == '__main__':
    df_sars = load_data("walz_data.csv")

    # grouped_box_plot(df_sars, "III.12", ["VII.1A", "VII.2A", "VII.3A"], y_title="RBD Peptid Rekombinant",
    #                  title="RBD Peptid Rekombinant for loss of taste/smell")
    # scatter_correlation_plot(df_sars, "VII.1A", "VII.2A")
    # plot = sns.pairplot(df, hue="species")
    # plot.show()
    num_cols = ['III.9: Wenn ja, bis zu welcher maximalenTemperatur?', 'VII.2A: OD IgM RBD Peptid rekombinant',
                'VII.1A: OD IgG RBD Peptid rekombinant', 'VII.3A: OD IgA RBD Peptid rekombinant']
    cat_col = 'III.6: Haben Sie sich krank gefühlt?'

    find_variables(df_sars, display=True)
    # grouped_plot_matrix(df_sars,
    #                     num_cols=num_cols,
    #                     cat_col='III.6: Haben Sie sich krank gefühlt?')
    # grouped_box_plot(df_sars, cat_col, num_cols, points=False)
    # parallel_coordinates_plot(df_sars, num_cols, cat_col)
    #print()
    # scatter_correlation_plot(df_sars, num_cols[1], num_cols[2])

    regression_scatter_matrix(df_sars, num_cols)

