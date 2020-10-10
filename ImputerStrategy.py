import re
from enum import Enum

regex = re.compile('[^\sa-zA-Z]')
age_mean = {}


class ImputerStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'
    CONSTANT = 'constant'
    REGRESSOR_MODEL = 'regressor_model'
    CLASSIFICATION_MODEL = 'clasification_model'


    fill_mean = lambda col: col.fillna(col.mean())
    fill_median = lambda col: col.fillna(col.median())
    fill_mode = lambda col: col.fillna(col.mode()[0])

    impute_strategies = {
        MEAN: fill_mean,
        MEDIAN: fill_median,
        MODE: fill_mode

    }

    def impute(self, dataset, impute_strategy):
        if impute_strategy in [ImputerStrategy.MEAN, ImputerStrategy.MEDIAN, ImputerStrategy.MODE]:
            return dataset.apply(self.impute_strategies[impute_strategy], axis=0)
        else:
            return dataset
