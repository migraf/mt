import catboost
from ..analysis import *

class MultiModelPredictor:
    def __init__(self, data, num_cols, cat_cols, target):
        self.data = data
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target = target

    def detect_prediction_mode(self):
        pass

    def train_models(self):
        pass

    def display_results(self):
        pass


if __name__ == '__main__':
    data = load_data("walz_data.csv")
    print(data.info())