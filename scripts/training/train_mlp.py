from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPRegressor
import numpy as np
import joblib

from load import load_model_data, get_train_test_split


def search_hyperparams(X_train, y_train, seed: int = 42):
    clf = MLPRegressor()
    param_dist = {
        "hidden_layer_sizes": [(50,), (10, 5), (20, 15)],
        "alpha": [0.01, 0.1, 0.5],
        "learning_rate": ["constant", "adaptive"],
        "max_iter": [100, 200, 500],
    }

    scoring = {"log_loss": "neg_log_loss", "roc_auc": "roc_auc"}

    # Create the random search object
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        scoring=scoring,
        random_state=seed,
        n_iter=4,
        cv=4,
        refit="roc_auc",
    )
    search = search.fit(X_train, y_train)

    print("Best parameters found: ", search.best_params_)
    print("Best ROC AUC score: ", search.best_score_)

    # Get the scores for all metrics
    results = search.cv_results_
    print("All metrics:")
    for scorer in scoring:
        print(f"Best {scorer} score: ", np.max(results[f"mean_test_{scorer}"]))
    return search


def main():
    df = load_model_data()
    X_train, X_test, y_train, y_test = get_train_test_split()
    search = search_hyperparams(X_train, y_train)
    joblib.dump(search, "../../src/models/mlp_model.pkl")
