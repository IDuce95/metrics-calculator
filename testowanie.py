import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from metrics_calculator.main import calculate_metrics


''' Test regresji '''
X_train_regression, y_train_regression, _ = make_regression(n_samples=1000,
                                                            n_features=2,
                                                            coef=True)

X_val_regression, y_val_regression, _ = make_regression(n_samples=1000,
                                                        n_features=2,
                                                        coef=True)

X_test_regression, y_test_regression, _ = make_regression(n_samples=1000,
                                                          n_features=2,
                                                          coef=True)


regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X_train_regression, y_train_regression)


data = {
    'train': {
        'y_real': pd.Series(y_train_regression, dtype=float),
        'y_pred': pd.Series(regressor.predict(X_train_regression), dtype=float)
    },
    'val': {
        'y_real': pd.Series(y_val_regression, dtype=float),
        'y_pred': pd.Series(regressor.predict(X_val_regression), dtype=float)
    },
    'test': {
        'y_real': pd.Series(y_test_regression, dtype=float),
        'y_pred': pd.Series(regressor.predict(X_test_regression), dtype=float)
    }
}


metadata = {
    'problem_type': 'regression',
    'metrics': ('rmse', 'r2', 'mape', 'mse'),
    'results_file_name': 'metrics_regression',
    'results_structure': 'single_row',
    'comment': 'Komentarz testowy',
    'save': False
}


results = calculate_metrics(data=data, metadata=metadata)
print('\n\n@@@@@ REGRESJA @@@@@')
print(results)


''' Test klasyfikacji '''
X_train_classification, y_train_classification = \
    make_classification(n_samples=1000,
                        n_features=20)

X_val_classification, y_val_classification = \
    make_classification(n_samples=1000,
                        n_features=20)

X_test_classification, y_test_classification = \
    make_classification(n_samples=1000,
                        n_features=20)

classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train_classification, y_train_classification)


data = {
    'train': {
        'y_real': pd.Series(y_train_classification, dtype=float),
        'y_pred': pd.Series(classifier.predict(X_train_classification),
                            dtype=float)
    },
    'val': {
        'y_real': pd.Series(y_val_classification, dtype=float),
        'y_pred': pd.Series(classifier.predict(X_val_classification),
                            dtype=float)
    },
    'test': {
        'y_real': pd.Series(y_test_classification, dtype=float),
        'y_pred': pd.Series(classifier.predict(X_test_classification),
                            dtype=float)
    }
}


metadata = {
    'problem_type': 'classification',
    'metrics': ('acc', 'b_acc', 'recall', 'f1', 'precision'),
    'results_file_name': 'metrics_classification',
    'results_structure': 'multiple_rows',
    'comment': 'Komentarz testowy',
    'save': False
}


results = calculate_metrics(data=data, metadata=metadata)
print('\n\n@@@@@ KLASYFIKACJA @@@@@')
print(results)
