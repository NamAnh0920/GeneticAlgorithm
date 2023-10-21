import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import sklearn.tree

import neural


def create_model(args, seed, X, y, checkpoint_file=""):
    np.random.seed(seed) if seed else None

    # train a model
    if args.model == "linear":
        hyperparameter_sets = create_linear_hyperparameter_sets()
        model, hyperparameters, kfold_y_hat = learn_model(X, y, args.k, hyperparameter_sets, build_linear, seed)
    elif args.model == "forest":
        hyperparameter_sets = create_forest_hyperparameter_sets(X.shape[1])
        model, hyperparameters, kfold_y_hat = learn_model(X, y, args.k, hyperparameter_sets, build_forest, seed)
    elif args.model == "neural":
        hyperparameter_sets = neural.create_neural_hyperparameter_sets(X.shape[1])
        model, hyperparameters, kfold_y_hat = learn_model(X, y, args.k, hyperparameter_sets,
                                                          neural.build_neural, seed,
                                                          checkpoint_file =checkpoint_file)
    elif args.model == "svm":
        hyperparameter_sets = create_svm_hyperparameter_sets()
        model, hyperparameters, kfold_y_hat = learn_model(X, y, args.k, hyperparameter_sets, build_svm, seed)
    elif args.model == "tree":
        hyperparameter_sets = create_tree_hyperparameter_sets()
        model, hyperparameters, kfold_y_hat = learn_model(X, y, args.k, hyperparameter_sets, build_tree, seed)
    elif args.model == "boost":
        hyperparameter_sets = create_boost_hyperparameter_sets()
        model, hyperparameters, kfold_y_hat = learn_model(X, y, args.k, hyperparameter_sets, build_boost, seed)

    performance = calculate_mse(kfold_y_hat, y)
    # print("k-fold performance:", performance)

    return model, hyperparameters


def learn_model(X, y, k, hyperparameter_sets, model_builder, seed, checkpoint_file=""):
    # use grid search to find the best hyperparameters
    best_hyperparameters, kfold_y_hat = grid_search(X, y, k, hyperparameter_sets, model_builder, seed, checkpoint_file)

    # create the model using the best hyperparameters
    n_features = X.shape[1]
    model = model_builder(best_hyperparameters, seed, n_features)

    if checkpoint_file:
        neural.fit(model, X, y, checkpoint_file)
    else:
        model.fit(X, y)

    return model, best_hyperparameters, kfold_y_hat


def grid_search(X, y, k, hyperparameter_sets, model_builder, seed, checkpoint_file=""):
    # perform our grid search
    best_performance = float('inf')
    best_hyperparameters = None
    best_kfold_y_hat = None

    for hyperparameter_combo in hyperparameter_sets:
        performance, kfold_y_hat = test_hyperparameters(X, y, k, hyperparameter_combo, model_builder, seed, checkpoint_file)

        # print(hyperparameter_combo, performance)

        if performance < best_performance:
            best_performance = performance
            best_hyperparameters = hyperparameter_combo
            best_kfold_y_hat = kfold_y_hat

    return best_hyperparameters, best_kfold_y_hat


def test_hyperparameters(X, y, k, hyperparameter_combo, model_builder, seed, checkpoint_file=""):
    n_features = X.shape[1]

    # create the models for k-fold cross validation
    kfold_models = []
    for i in range(k):
        model = model_builder(hyperparameter_combo, seed, n_features)
        kfold_models.append(model)

    # perform k-fold cross validation
    kfold_y_hat = kfold_cv(kfold_models, X, y, checkpoint_file)
    performance = calculate_mse(kfold_y_hat, y)

    return performance, kfold_y_hat


def create_linear_hyperparameter_sets():
    hyperparameter_sets = list()

    hyperparameter_sets.append((None, None))
    hyperparameter_sets.append(("l1", None))
    hyperparameter_sets.append(("l2", None))
    hyperparameter_sets.append(("elasticnet", 0.25))
    hyperparameter_sets.append(("elasticnet", 0.5))
    hyperparameter_sets.append(("elasticnet", 0.75))

    return hyperparameter_sets


def build_linear(hyperparameter_combo, seed, n_features):
    if hyperparameter_combo[0] is None:
        return sklearn.linear_model.LinearRegression()
    elif hyperparameter_combo[0] == "l1":
        return sklearn.linear_model.Lasso(max_iter=10000,
                                          random_state=seed)
    elif hyperparameter_combo[0] == "l2":
        return sklearn.linear_model.Ridge(max_iter=10000,
                                          solver="saga",
                                          random_state=seed)
    elif hyperparameter_combo[0] == "elasticnet":
        return sklearn.linear_model.ElasticNet(l1_ratio=hyperparameter_combo[1],
                                               max_iter=10000,
                                               random_state=seed)


def create_forest_hyperparameter_sets(total_attributes):
    num_attributes_options = []
    sqrt_attributes = int(np.sqrt(total_attributes))
    for coeff in [0.5, 1, 2.0]:
        num_attributes = min(total_attributes, max(2, int(sqrt_attributes * coeff)))
        if num_attributes not in num_attributes_options:
            num_attributes_options.append(num_attributes)

    hyperparameter_sets = []
    for n_estimators in [100, 200, 500, 1000]:
        for max_features in num_attributes_options:
            hyperparameter_combo = (n_estimators, max_features)
            hyperparameter_sets.append(hyperparameter_combo)

    return hyperparameter_sets


def build_forest(hyperparameter_combo, seed, n_features):
    return sklearn.ensemble.RandomForestRegressor(n_estimators=hyperparameter_combo[0],
                                                  max_features=hyperparameter_combo[1],
                                                  oob_score=True,
                                                  random_state=seed)


def create_svm_hyperparameter_sets():
    inner_sets = []
    inner_sets.append(("linear", 3))
    inner_sets.append(("poly", 2))
    inner_sets.append(("poly", 3))
    inner_sets.append(("poly", 4))
    inner_sets.append(("poly", 5))
    inner_sets.append(("rbf", 3))
    inner_sets.append(("sigmoid", 3))

    hyperparameter_sets = []

    for combo in inner_sets:
        if combo[0] == "linear":
            cs = [0.25, 0.3333, 0.5, 1.0, 2.0, 3.0, 4.0]
        else:
            cs = [0.25, 0.3333, 0.5, 1.0, 2.0, 3.0, 4.0]

        for c in cs:
            hyperparameter_combo = (combo[0], combo[1], c)
            hyperparameter_sets.append(hyperparameter_combo)

    return hyperparameter_sets


def build_svm(hyperparameter_combo, seed, n_features):
    return sklearn.svm.SVR(kernel=hyperparameter_combo[0],
                           degree=hyperparameter_combo[1],
                           C=hyperparameter_combo[2])


def create_tree_hyperparameter_sets():
    hyperparameter_sets = []

    for min_samples_leaf in [0.02, 0.05, 0.1]:
        for ccp_alpha in [0.0, 0.001, 0.002, 0.005]:
            hyperparameter_sets.append((min_samples_leaf, ccp_alpha))

    return hyperparameter_sets


def build_tree(hyperparameter_combo, seed, n_features):
    return sklearn.tree.DecisionTreeRegressor(min_samples_leaf=hyperparameter_combo[0],
                                              ccp_alpha=hyperparameter_combo[1],
                                              random_state=seed)


def create_boost_hyperparameter_sets():
    hyperparameter_sets = []

    for n_estimator_values in [100, 200, 500, 1000]:
        for learning_rate_values in [0.01, 0.02, 0.05, 0.1, 0.2]:
            hyperparameter_sets.append((n_estimator_values, learning_rate_values))

    return hyperparameter_sets


def build_boost(hyperparameter_combo, seed, n_features):

    return sklearn.ensemble.GradientBoostingRegressor(n_estimators=hyperparameter_combo[0],
                                                      learning_rate=hyperparameter_combo[1],
                                                      random_state=seed)


def kfold_cv(models, X, y, checkpoint_file=""):
    # split the data into folds
    k = len(models)
    folds = create_folds(k, X, y)
    kfold_y_hat = []

    # perform each fold and aggregate the results
    matrix = {}
    for i in range(k):
        # grab what we need for this fold
        model = models[i]
        fold_train_X, fold_train_y, fold_test_X, fold_test_y = folds[i]

        # fit the model for this fold and predict on the validation fold
        if checkpoint_file:
            neural.fit(model, fold_train_X, fold_train_y, checkpoint_file)
            fold_y_hat = model.predict(fold_test_X)
        else:
            model.fit(fold_train_X, fold_train_y)
            fold_y_hat = model.predict(fold_test_X)

        kfold_y_hat.extend(fold_y_hat)

    return kfold_y_hat


def create_folds(k, X, y):
    # split the instances into folds
    n = len(X)
    n_per = n / k
    fold_indices = [[] for _ in range(k)]
    for index in range(n):
        i = int(index / n_per)
        fold_indices[i].append(index)

    folds = []
    for i in range(k):
        # get the test indices for this fold
        test_indices = fold_indices[i]
        train_indices = [index for index in range(test_indices[0])] + [index for index in
                                                                       range(test_indices[-1] + 1, n)]

        # splice the data
        fold_train_X = X.iloc[train_indices]
        fold_train_y = y[train_indices]
        fold_test_X = X.iloc[test_indices]
        fold_test_y = y[test_indices]

        # save the data subsets for this fold
        data = (fold_train_X, fold_train_y, fold_test_X, fold_test_y)
        folds.append(data)

    return folds


def calculate_results(y_hat, y):
    return str(calculate_mse(y_hat, y))


def calculate_mse(y_hat, y):
    sse = 0.0
    n = len(y_hat)

    for i in range(n):
        err = y_hat[i] - y[i]
        sse = sse + err * err

    return sse / n
