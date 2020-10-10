from enum import Enum
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


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
    ImputerStrategy.MEAN: fill_mean,
    ImputerStrategy.MEDIAN: fill_median,
    ImputerStrategy.MODE: fill_mode

}


def impute(df, impute_strategy):
    if impute_strategy in [ImputerStrategy.MEAN, ImputerStrategy.MEDIAN, ImputerStrategy.MODE]:
        return df.apply(impute_strategies[impute_strategy], axis=0)
    else:
        return df


def categorize(df, column, bins):
    data = pd.cut(np.array(df[column]), bins=bins)
    data = pd.Series(data)
    data = pd.DataFrame(data, columns=[f'{column}_Range'])
    data = data[f'{column}_Range'].apply(lambda value: str(value).replace('(', '').replace(']', '').replace(', ', '_'))

    df = df.join(pd.DataFrame(data, columns=[f'{column}_Range']))
    df = df.join(pd.get_dummies(df[f'{column}_Range']))
    df = df.drop([column], axis=1)
    return df.drop([f'{column}_Range'], axis=1)


def one_hot_encode(df, column):
    df = df.join(pd.get_dummies(df[column], prefix=column))
    return df.drop([column], axis=1)


def label_encode(df, column):
    le = preprocessing.LabelEncoder()
    values = list(df[column].values)
    le.fit(values)
    df[column] = le.transform(values)
    return df


def scale_normalize(df, columns):
    df[columns] = MinMaxScaler().fit_transform(df[columns])
    for column in columns:
        df[column] = df[column].apply(lambda x: np.log(x + 1))
    return df


def pre_process(df):
    df = df.drop(['PassengerId', 'Cabin'], axis=1)
    df['Age'] = impute(df[['Age']], ImputerStrategy.MEAN)
    df['Fare'] = impute(df[['Fare']], ImputerStrategy.MEAN)
    df['Relatives'] = df['SibSp'] + df['Parch']

    df = categorize(df, 'Age', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    df = one_hot_encode(df, 'Embarked')
    df = one_hot_encode(df, 'Sex')

    df['Relatives'] = df['SibSp'] + df['Parch']

    df['title'] = df['Name'].apply(lambda name: name.split(' ')[1])

    df = label_encode(df, 'title')
    df = label_encode(df, 'Ticket')

    df = df.drop(['Name'], axis=1)
    if 'Survived' in list(df.columns):
        df = df.drop_duplicates()

    columns = ['Pclass', 'SibSp', 'Parch', 'Ticket', 'Fare', 'title', 'Relatives']
    df = scale_normalize(df, columns)
    return df


if __name__ == '__main__':
    dataset = pd.read_csv('datasets/titanic.csv')
    processed_df = pre_process(dataset)
    processed_df.to_csv('datasets/train.csv', index=False)
