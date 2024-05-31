from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

from load import load_model_data, get_train_test_split


def search_hyperparams(X_train, y_train, seed: int = 42):
    rf_regressor = RandomForestRegressor()
    param_dist = {
        "n_estimators": np.random.randint(10, 100, 10).tolist(),
        "max_depth": np.random.randint(2, 20, 10).tolist(),
        "min_samples_split": np.random.randint(2, 20, 10).tolist(),
        "min_samples_leaf": np.random.randint(1, 20, 10).tolist(),
    }
    scoring = {"log_loss": "neg_log_loss", "roc_auc": "roc_auc"}

    # Create the random search object
    search = RandomizedSearchCV(
        estimator=rf_regressor,
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
    joblib.dump(search, "../../src/models/rf_model.pkl")
