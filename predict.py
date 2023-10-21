import argparse
import graphviz
import numpy as np
import pandas
import sklearn.tree
import sys

import process_data
import regression
import neural


BASE_SEED = 12345
PREFIX = "covid"


def log_tree(tree, features, filename):
    # save the visualization as an image
    treeGraph = graphviz.Source(sklearn.tree.export_graphviz(tree, out_file=None,
                                                             feature_names=list(features),
                                                             class_names=["no", "yes"],
                                                             filled=True, rounded=True,
                                                             special_characters=True))

    # credit to @singer from https://stackoverflow.com/questions/27817994/visualizing-decision-tree-in-scikit-learn
    png_bytes = treeGraph.pipe(format='png')
    with open(filename + ".png", 'wb') as f:
        f.write(png_bytes)


def log_forest_importances(forest, features, filename):
    # get the importances
    importances = list()
    for i in range(len(features)):
        t = (forest.feature_importances_[i], features[i])
        importances.append(t)
    importances.sort(reverse=True)

    with open(filename + ".csv", 'w') as file:
        file.write("Feature,Importance\n")

        for t in importances:
            file.write(",".join([str(v) for v in t]) + "\n")


def log_predictions(model, week, lag, window, prior_weeks, best_hyperparameters, test_y_hat, test_counties):
    output_file = PREFIX + "_predictions_" + model + "_" + str(week) + "week_" + str(lag) + "lag_" \
                  + str(prior_weeks) + "prior_" + str(window) + "window.csv"
    header = "Model,Week,Lag,Window,PriorWeeks,BestHyper,FIPS,Prediction"

    with open(output_file, "w") as file:
        file.write(header + "\n")
        run_info = ",".join([model, str(week), str(lag), str(window), str(prior_weeks)])
        hyperparameter_info = "-".join([str(h).replace("-", "") for h in best_hyperparameters])
        run_info += "," + hyperparameter_info

        for i in range(len(test_counties)):
            county = test_counties[i]
            prediction = test_y_hat[i]

            file.write(run_info + "," + str(county) + "," + str(prediction) + "\n")


def validate_args(args):
    if args.week > 114 or args.week < -14:
        print("args.week must be between -14 and 114 (inclusive).  You provided", args.week)
        sys.exit(-1)

    # check how many prior weeks the user wants
    if args.prior_weeks == -1:
        args.prior_weeks = args.week - 1

    earliest = args.week - args.prior_weeks
    if earliest < 1:
        print("1 is the earliest week of data, so args.week - args.prior_weeks cannot be less than 1")
        sys.exit(-1)

    if args.prior_weeks < args.window:
        print("args.window must be less than or equal to args.prior_weeks")
        sys.exit(-1)


def main(args):
    validate_args(args)

    if args.model == "neural":
        neural.setup()

    # do we need to do any pre-processing?
    scale = args.model == "linear" or args.model == "neural" or args.model == "svm"
    one_hot = True
    remove_nas = True

    # grab all the data
    attributes, labels, all_categorical = process_data.read_data(remove_nas, args.outcome)

    # do we need to do any pre-processing?
    if one_hot:
        attributes, categorical_column_map = process_data.create_all_onehot(attributes, all_categorical)

    if scale:
        # NOTE: we assume we know max and min values in advance
        process_data.scale_features(attributes, all_categorical)

    '''
    window 

    left FIPS and years alone in the convertion
    convertion then scale again => fit on tree

    need to save test_y
    '''

    # split the data into train and test
    train_X, train_y, test_X, _, test_counties = process_data.split_data(attributes, labels, args.week, args.lag,
                                                                              args.window, args.prior_weeks)
    # set the random seed for this run
    np.random.seed(BASE_SEED)

    # shuffle the training data
    train_X, train_y = process_data.shuffle_instances(train_X, train_y)

    # find the best model using hyperparameter grid search
    checkpoint_file = ""
    if args.model == "neural":
        checkpoint_file = "checkpoints_" + str(args.week) + "week_" + str(args.lag) + "lag_" \
                          + str(args.prior_weeks) + "prior_" + str(args.window) + "window.h5"

    model, best_hyperparameters = regression.create_model(args, BASE_SEED, train_X, train_y, checkpoint_file)

    # save the models if possible
    if args.model == "forest":
        forest_filename = PREFIX + "_forest_" + str(args.week) + "week_" + str(args.lag) + "lag_" \
                          + str(args.prior_weeks) + "prior_" + str(args.window) + "window"
        log_forest_importances(model, test_X.columns, forest_filename)
    elif args.model == "tree":
        tree_filename = PREFIX + "_tree_" + str(args.week) + "week_" + str(args.lag) + "lag_" \
                        + str(args.prior_weeks) + "prior_" + str(args.window) + "window"
        log_tree(model, test_X.columns, tree_filename)

    # calculate the results on the test set
    test_y_hat = model.predict(test_X)
    if model == "neural":
        test_y_hat = test_y_hat.flatten()

    # log the results
    log_predictions(args.model, args.week, args.lag, args.window, args.prior_weeks, best_hyperparameters,
                    test_y_hat, test_counties)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Makes predictions for county-level COVID-19 outcomes")
    parser.add_argument("--week", metavar="week", type=int, default=2,
                        help='The week to use as the test set. Default is 2.')
    parser.add_argument("--lag", metavar="lag", type=int, default=1,
                        help='The number of weeks of lag to use for predictions (predictors come from earlier weeks).'
                             + ' Default is 1 week.')
    parser.add_argument("--window", metavar="window", type=int, default=0,
                        help='Number of weeks of autoregression to use in each windowed instance. Default is 0.')
    parser.add_argument("--prior_weeks", metavar="prior_weeks", type=int, default=-1,
                        help='Number of prior weeks to use in the training set.'
                             + ' Default is -1, which means use all available data.')
    parser.add_argument('--model', metavar='model', type=str, default="linear",
                        help='The type of model to learn (linear, tree, forest, boost, svm, neural).'
                             ' Default is linear.')
    parser.add_argument("--k", metavar="k", type=int, default=5,
                        help='Number of folds to use in k-fold cross validation. Default is 5.')
    parser.add_argument("--outcome", metavar="outcome", type=str, default="cases_inc",
                        help="The outcome to predict (cases_inc, deaths_inc, TODO). Default is cases_inc.")
    main(parser.parse_args())
