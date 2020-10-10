import json
import sys
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import classifiers
import warnings
warnings.filterwarnings('ignore')


def split_data(df, config):
    from sklearn.model_selection import train_test_split
    features = df.drop(['Survived'], axis=1)
    labels = df['Survived']

    return train_test_split(features, labels, test_size=config['test_size'], random_state=config['seed'], stratify=labels)


def tune_classifier(clf, parameters, X_train, X_test, y_train, y_test):
    scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=scorer, iid=False)
    grid_fit = grid_obj.fit(X_train, y_train)
    best_clf = grid_fit.best_estimator_

    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    default_score = accuracy_score(y_test, predictions)
    tuned_score = accuracy_score(y_test, best_predictions)

    cnf_matrix = confusion_matrix(y_test, best_predictions)

    return best_clf, default_score, tuned_score, cnf_matrix


def get_config(path):
    config = {}
    with open(path) as json_file:
        config = json.load(json_file)
    return config


if __name__ == '__main__':
    config = get_config(sys.argv[1])
    train_df = pd.read_csv(config['dataset'])
    X_train, X_test, y_train, y_test = split_data(train_df, config)
    parameters = config['classifier_hyperparams']

    classifier = classifiers.classifiers_list[config['classifier']]
    classifier.set_params(**parameters)

    classifier = classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    model_score = accuracy_score(y_test, predictions)

    print("Model score: {:.4f}".format(model_score))
    joblib.dump(classifier, 'model.joblib')
