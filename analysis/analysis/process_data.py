import pandas as pd
import numpy as np


def load_data(path, two_sheets=None):
    """
    Load a file into a pandas dataframe
    :param path: filepath pointing to a datafile, file must be xlsx or csv
    :return : pandas dataframe based on the data, if walz style upload also return mapping dict
    """
    # load data into dataframe, depending on file path
    file_type = path.split(".")[1]
    if file_type == "csv":
        dfs = pd.read_csv(path)
    elif file_type == "xlsx":
        dfs = pd.read_excel(path, sheet_name=None, )
        dfs = dfs["15052020SARS-CoV-2_final"]
        # TODO remove this
        # specific code for loading the sheets provided by Walz Lab
        if two_sheets:
            data_df = dfs["Sheet1"]
            data_dict = dfs["Sheet2"]

            # Remove fully empty rows from the data
            data_df = data_df.dropna(how="all")
            data_dict = data_dict.dropna(how="all")
            mapping_dict = {}
            # TODO vectorize this
            for col in data_dict.columns:
                data_col = col.split(" ")[0].replace(":", "")
                mapping_dict[data_col] = col
            # TODO use dict get for mapping
            mapping_dict = {col: mapping_dict.get(col, col) for col in data_df.columns}
            return data_df, mapping_dict
    dfs = dfs.dropna(how="all")
    dfs = dfs.convert_dtypes()
    # dfs.to_csv("walz_data.csv", index=False)

    # TODO maybe convert all object columns to categorical based on number of unique values
    return dfs


def create_data_table_df(df, mapping):
    """
    Create a table with summary information based on a given dataframe and mapping
    :param df:
    :type df:
    :return:
    :rtype:
    """
    table_columns = ["Variable Name", "Variable ID", "Type", "Summary", "Availability", "Select as Target",
                     "Include in computation"]
    table_df = pd.DataFrame(columns=table_columns)
    table_df["Variable Name"] = df.columns.map(mapping)
    table_df["Variable ID"] = df.columns
    table_df["Variable Name"] = table_df["Variable Name"].replace({"Eingabedatum": "Blutabnahmetag"})
    types = []
    available = []
    # Add the types for each variable
    dtypes = df.dtypes
    for col in table_df["Variable ID"]:
        if dtypes[col] == "object":
            types.append("categorical")
        elif dtypes[col] in ["float64", "int64"]:
            types.append("continuous")
        else:
            types.append("Date")
        available.append(f"{round(np.sum(df[col].notnull()) / len(df) * 100, 2)} %")
    table_df["Type"] = types
    table_df["Availability"] = available
    summary = []
    for i, type in table_df["Type"].items():
        if type == "categorical":
            val_counts = df[table_df["Variable ID"][i]].value_counts()
            cat_sum = f"Top categories: {val_counts[:2].to_dict()}," \
                      f" n unique: {len(df[table_df['Variable ID'][i]].unique())}"
            summary.append(cat_sum)
        if type == "continuous":
            mean = df[table_df["Variable ID"][i]].mean()
            std = df[table_df["Variable ID"][i]].std()
            summary.append(f"Mean: {mean}, STD: {std}")
        if type == "Date":
            # TODO implement this
            summary.append("Date ranges")
    table_df["Summary"] = summary
    table_df["Select as Target"] = "no"
    table_df["Include in computation"] = "yes"

    return table_df.to_dict(orient="records")


def get_column_details(df, col):
    """
    Get extended summary information about a column in the dataframe
    returns different results based on the type of the column
    :return : dictionary containing, summary plot,
    """
    dtype = df[col].dtype
    details = {}
    details["availability"] = f"{round(np.sum(df[col].notnull()) / len(df) * 100, 2)} %"
    if dtype in ["float64", "int64"]:
        print("numeric column")
        plot_data = {"type": "bar"}
        y, x = np.histogram(df[col][df[col].notnull()])
        plot_data["x"] = [float(i) for i in list(x)]
        plot_data["y"] = [float(i) for i in list(y)]
        details["plot"] = plot_data
        details["min"] = round(float(df[col].min()))
        details["max"] = round(float(df[col].max()))
        details["type"] = "numeric"
        return details
    elif dtype == "object":
        plot_data = {"type": "bar"}
        val_counts = df[col].value_counts()
        plot_data["x"] = list(val_counts.index)
        plot_data["y"] = [int(i) for i in list(val_counts.values)]
        details["plot"] = plot_data
        details["type"] = "categorical"
        details["values"] = list(df[col].unique())
        return details

    elif dtype == "datetime64[ns]":
        plot_data = {"type": "bar"}
        val_counts = df[col].value_counts().sort_index()
        plot_data["x"] = list(val_counts.index.strftime("%d-%m-%Y"))
        plot_data["y"] = [int(i) for i in list(val_counts.values)]
        details["plot"] = plot_data
        details["minDate"] = df[col].min().strftime("%d-%m-%Y")
        details["maxDate"] = df[col].max().strftime("%d-%m-%Y")
        details["type"] = "datetime"
        details["timeSpan"] = (df[col].max() - df[col].min()).days
        return details

def concatenate_dfs(df_1, df_2, merge_mapping=None, force=True):
    """
    Append the values of one dataframe with the same (or similar) schema to the original dataframe
    :param df_1: initial pandas df
    :type df_1:
    :param df_2: pandas df to be appended
    :type df_2:
    :return:
    :rtype:
    """
    if force:
        mapping_dict = dict(zip(list(df_2.columns), list(df_1.columns)))
        renamed_df = df_2.rename(columns=mapping_dict)

    merged_df = pd.concat([df_1, renamed_df], ignore_index=True)
    return merged_df


if __name__ == '__main__':
    df_sars, dict_sars = load_data(
        "/server/walzLabBackend/flaskr/user_data/",
        two_sheets=True)
    # df_hv, dict_hv = load_data(
    #     "C:\\hypothesis\\repositories\\server\\walzLabBackend\\flaskr\\user_data\\Datentabelle_CoVid19_HV.xlsx",
    #     two_sheets=True)
    # full_df = concatenate_dfs(df_sars, df_hv)

    print(df_sars.info())
