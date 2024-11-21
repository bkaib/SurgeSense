#---
#  Optimize Hyperparameters
#---
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def optimize_hyperparameter(X_train, y_train, clf, optimizer, param_grid, k, n_iter=None, n_jobs=-1):
    """
    Description: 
        Return best hyperparameters for a model based on chosen optimizer
    Parameters:
        X_train (): Predictor train data
        y_train (): Predictand train data
        clf (): Base Model
        optimizer (): GridSearchCV or RandomizedSearchCV
        param_grid (dict): Dictionary with hyperparameter ranges
        k (int): k-fold Cross-Validation
        n_iter (int): Number of combinations used for RandomizedSearchCV (Defaults:None)
        n_jobs (int): Number of processor used. (Defaults:-1, e.g. all processors)
    """
    # Modules
    #---
    from sklearn.model_selection import GridSearchCV 
    from sklearn.model_selection import RandomizedSearchCV
    # RandomSearchCV
    #---
    if optimizer == "RandomSearchCV":
        assert n_iter != None, f"{optimizer} needs number of combinations."
        opt_model = RandomizedSearchCV(estimator=clf, 
        param_distributions = param_grid, 
        n_iter = n_iter, 
        cv = k, 
        verbose = 2, 
        random_state = 0, 
        n_jobs = n_jobs,
        )

    # GridSearchCV
    #---
    if optimizer == "GridSearchCV":
        # Instantiate the grid search model
        opt_model = GridSearchCV(estimator=clf, 
        param_grid = param_grid, 
        cv = k,
        verbose = 2, 
        n_jobs = n_jobs, 
        )

    # Fit the random search model
    opt_model.fit(X_train, y_train)

    # Best Params
    #---

    return opt_model.best_params_

def model_training(
        X: np.array,
        y: np.array,
        clf: RandomForestClassifier,
        random_state: int,
        test_size: float,
        is_scaled: bool,
        optimizer: str,
        hparam_grid: dict,
        k: int,
        n_iter: int,
        n_jobs: int=-1,
):
    """
    Description:
    Parameters:
        clf: Classifier, e.g. RandomForestClassifier
    Returns:
    """

    #---
    # Apply Train-Test split 
    #---
    print("Apply train-test-split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        random_state=random_state, 
        test_size=test_size)

    #---
    # Scale data if they are on different scales
    #---
    X_test_unscaled = X_test

    if is_scaled:
        print("Scale training data")
        s = StandardScaler()
        s.fit(X_train)
        X_train = s.transform(X_train)
        X_test = s.transform(X_test)

    #---
    #  Optimize Hyperparameters
    #---
    print(f"Optimize Hyperparameters using {optimizer}")
    print(f"Tested Hyperparameters: {hparam_grid}")
    
    hparam_grid_opt = optimize_hyperparameter(
        X_train, 
        y_train, 
        clf(), 
        optimizer, 
        hparam_grid, 
        k, 
        n_iter, 
        n_jobs=n_jobs)

    #---
    # Fit the model
    #---
    print(f"Fit model with hyperparameters {hparam_grid_opt}")

    model = clf(**hparam_grid_opt) # One can set parameters afterwards via model.set_params() 

    model.fit(X_train, y_train)

    return model, hparam_grid_opt, X_train, X_test, y_train, y_test, X_test_unscaled, s