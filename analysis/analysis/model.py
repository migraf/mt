"""
Base Classes for machine learning models
"""

import numpy as np
from joblib import dump, load
import pandas as pd



class Model:
    def __init__(self, name, data, numeric_columns, categorical_columns, target, config, model_id=None):
        """
        Create a model instance
        :param data:
        :type data:
        :param numeric_columns:
        :type numeric_columns:
        :param categorical_columns:
        :type categorical_columns:
        :param target:
        :type target:
        :param config:
        :type config:
        """
        self.name = name
        self.data = data
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.config = config
        if model_id:
            self.id = model_id
        else:
            self.id = np.random.randint(low=1000000, high=100000000)


    def save(self, filepath):
        """
        Stores the model to disk under a unique identifier
        :return:
        :rtype:
        """
        with open(filepath, "wb") as fo:
            dump(self, fo)

if __name__ == '__main__':
    model = Model(name="test_model", data=pd.Ser)