import numpy as np
import pandas as pd
from server.walzLabBackend.flaskr.models.util import load_data as ld
from server.walzLabBackend.flaskr.models.util import *
from server.walzLabBackend.flaskr.data import *
import plotly.graph_objects as go
from fastai.tabular import *


def create_nn_training_data(df, num_cols, cat_cols, target):
    """
    Perform the preprocessing steps required to train a neural network
    :param df:
    :type df:
    :return:
    :rtype:
    """
    # Remove columns with nan values in target
    selected_columns = num_cols + cat_cols

    df = df[df[target].notnull()]
    df = df[selected_columns].copy()
    procs = [FillMissing, Categorify, Normalize]
    valid_idx = range(int(len(df) * 0.8), len(df))

    path = "/server/walzLabBackend/flaskr/user_data"
    data = TabularDataBunch.from_df(path, df, dep_var=target, valid_idx=valid_idx, procs=procs, cat_names=cat_cols)
    print(data.train_ds.cont_names)
    (cat_x, cont_x), y = next(iter(data.train_dl))
    for o in (cat_x, cont_x, y): print(to_np(o[:5]))

    # Create embeddings for categorical variables
    embzs = {}
    for col in cat_cols:
        embzs[col] = 10

    return data, embzs


def train_model(db, embs):
    """
    Train a tabular neural network for a regression task
    :param db:
    :type db:
    :return:
    :rtype:
    """
    learn = tabular_learner(db, layers=[200, 100], emb_szs=embs, metrics=accuracy)
    learn.fit_one_cycle(1, 1e-2)
    print("test")


if __name__ == '__main__':
    df_sars, dict = ld(
        "C:\\hypothesis\\repositories\\server\\walzLabBackend\\flaskr\\user_data\\Datentabelle_CoVid19_SARS_new.xlsx",
        two_sheets=True)

    db, embs = create_nn_training_data(df_sars, numerical_columns, categorical_columns, "IX.1C")
    train_model(db, embs)
