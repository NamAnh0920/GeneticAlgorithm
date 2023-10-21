import numpy as np
import pandas
import argparse
import regression

DATA_FOLDER = "../../../data/"
DATA_FILE = "features.csv"
LABELS_FILE = "outcomes.csv"
REMOVE_COLS = []  # empty for now, but could change, so leaving this here
CATEGORICAL = []  # empty for now, but could change for other data sets, so leaving this here

SCALE_TYPE_NONE = "none"
SCALE_TYPE_MINMAX = "minmax"
SCALE_TYPE_Z = "z"

def read_data(remove_nas, label_col):
    # read in the attributes
    attributes = pandas.read_csv(DATA_FOLDER + DATA_FILE, encoding="latin-1")
    if REMOVE_COLS:
        attributes = attributes.drop(REMOVE_COLS, axis=1)

    if remove_nas:
        attributes = attributes.dropna()

    # create the list of categorical features
    categorical = CATEGORICAL  # empty for now, but could change for other data sets, so leaving this here
    if categorical:
        categorical = [col for col in categorical if col not in REMOVE_COLS]

    # read in the labels
    labels = pandas.read_csv(DATA_FOLDER + LABELS_FILE, encoding="latin-1")
    all_label_cols = ["FIPS", "week_index", label_col]
    labels = labels[all_label_cols]

    return attributes, labels, categorical


def create_all_onehot(attributes, all_categorical):
    categorical_column_map = {}

    for col in attributes.columns:
        if col in all_categorical:
            attributes, map = create_onehot(attributes, col)
            categorical_column_map.update(map)

    return attributes, categorical_column_map


def create_onehot(dataset, column):
    # create the one hot encoding
    onehots = pandas.get_dummies(dataset[column], column, dummy_na=True, drop_first=True)

    # convert everything to NA if the original was NA
    na_column = column + "_nan"
    onehots[na_column] = onehots[na_column].replace(1, np.NaN)
    onehots.loc[onehots.isna().any(axis=1), :] = np.NaN
    onehots = onehots.drop(na_column, axis=1)

    # track what columns were just created
    map = {}
    map[column] = list(onehots.columns)

    # add the onehots to the original dataset
    return pandas.concat([dataset.drop(column, axis=1), onehots], axis=1), map


def scale_features(attributes, all_categorical, scale_type=SCALE_TYPE_MINMAX):
    for col in attributes.columns:
        if col not in all_categorical and col != "FIPS" and col != "Year":
            attributes[col] = scale(attributes[col], scale_type)

    return attributes


def scale_label(labels, label_col, scale_type=SCALE_TYPE_NONE):
    if scale_type == SCALE_TYPE_MINMAX or scale_type == SCALE_TYPE_Z:
        labels[label_col] = scale(labels[label_col], scale_type)

    return labels


def scale(column, scale_type):
    if scale_type == SCALE_TYPE_MINMAX:
        min_val = column.min()
        max_val = column.max()

        if min_val < max_val and (min_val != 0.0 or max_val != 1.0):
            column = (column -  min_val) / (max_val - min_val)
    elif scale_type == SCALE_TYPE_Z:
        mean = column.mean()
        sd = column.std()

        if sd > 0:
            column = (column - mean) / sd

    return column

def scale_test_train(trainset, testset, all_categorical, scale_type = SCALE_TYPE_MINMAX):
    for col in trainset.columns:
        if col not in all_categorical and col != "FIPS" and col != "Year":
            if scale_type == SCALE_TYPE_MINMAX:
                max_val = trainset[col].max()
                min_val = trainset[col].min()

                if min_val < max_val and (min_val != 0.0 or max_val != 1.0):
                    trainset[col] = (trainset[col] - min_val)/(max_val - min_val)
                    testset[col] = (testset[col] - min_val)/(max_val - min_val)

            elif scale_type == SCALE_TYPE_Z:
                mean = trainset[col].mean()
                sd = trainset[col].std()

                if sd > 0:
                    trainset[col] = (trainset[col] - mean) / sd
                    testset[col] = (testset[col] - mean) / sd

    return trainset, testset
    
def filter_features(attributes, features):
    all_features = list(attributes.columns)

    for feature in all_features:
        if feature not in features:
            attributes = attributes.drop(feature, axis=1)

    return X


def split_data(attributes, labels, week, lag, window, prior_weeks):
    # subset the test data
    test_X, test_y, test_counties = create_window(week, attributes, labels, lag, window)
    if test_y is not None:
        test_y = test_y.to_numpy().flatten()

    # get all the training years
    train_Xs = []
    train_ys = []
    for prev_week in range(week - prior_weeks + window, week):
        # print("Prev week:", prev_week)
        part_X, part_y, _ = create_window(prev_week, attributes, labels, lag, window)

        # in case we tried to include a week in the training set for which we do not have data
        if part_y is not None:
            train_Xs.append(part_X)
            train_ys.append(part_y)

    train_X = pandas.concat(train_Xs, axis=0)
    train_y = pandas.concat(train_ys, axis=0)
    train_y = train_y.to_numpy().flatten()

    return train_X, train_y, test_X, test_y, test_counties


def create_window(week, attributes, labels, lag, window):
    # start with just the attributes
    X = attributes.copy()

    # add autoregression if window isn't 0
    if window > 0:
        # get the last week used to predict for the given week
        last_week = week - lag

        # get the labels for the last year in the window
        y_prev = labels[labels["week_index"] == last_week]
        y_prev = y_prev.drop("week_index", axis=1)

        # merge in the label from the last year
        X = pandas.merge(X, y_prev, how="inner", on=["FIPS"])

        # add in the rest of the window
        print("-" + str(last_week))
        for prev in range(1, window):
            prev_week = last_week - prev
            print("-" + str(prev_week))

            # get the labels for the previous week  in the window and rename to show the week offset
            y_prev = labels[labels["week_index"] == prev_week]
            y_prev = y_prev.drop("week_index", axis=1)
            y_prev = y_prev.rename(columns={name: (name + "_-" + str(prev)) for name in y_prev.columns[1:]})

            # merge in the previous year's data
            X = pandas.merge(X, y_prev, how="inner", on=["FIPS"])

    if week in labels["week_index"].unique():
        # get the labels for the correct year
        y = labels[labels["week_index"] == week]
        y = y.drop("week_index", axis=1)

        # save only the counties in both X and y
        y = y[y["FIPS"].isin(X["FIPS"])]
        X = X[X["FIPS"].isin(y["FIPS"])]

        # sort y by FIPS, then drop FIPS
        y = y.sort_values(by=["FIPS"]).drop("FIPS", axis=1)
    else:
        y = None

    # sort X by FIPS, then drop FIPS
    X = X.sort_values(by=["FIPS"])
    counties = X["FIPS"].tolist()
    X = X.drop("FIPS", axis=1)

    return X, y, counties


def shuffle_instances(X, y):
    # randomize the indices of the instances
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    return X.iloc[indices], y[indices]

def process_chromosome(attributes, chromosome):

    linear_cols = np.where(chromosome == 1)[0]

    log_cols = np.where(chromosome == 2)[0]

    exp_cols = np.where(chromosome == 3)[0]

    log_res = attributes.iloc[:, log_cols].apply(lambda att: np.log(att.replace(0, 0.0001)))

    exp_res = attributes.iloc[:, exp_cols].apply(lambda att: np.exp(att))

    linear_res = attributes.iloc[:, linear_cols]

    X = pandas.concat([linear_res, exp_res, log_res], axis = 1)

    return X

def main(args):
    attributes, labels, all_categorical = read_data(True, "cases_inc")
    attributes, categorical_column_map = create_all_onehot(attributes, all_categorical)
    scale_features(attributes, all_categorical)

    chrome = np.random.randint(0, 4, size=(len(attributes.columns)))
    print(len(chrome))

    X = process_chromosome(attributes, chrome)
    train_X, train_y, test_X, test_y, counties = split_data(X, labels, args.week, args.lag, args.window, args.prior_weeks)
    train_X, train_y = shuffle_instances(train_X, train_y)
    scale_features(train_X, all_categorical)
    scale_features(test_X, all_categorical)
    model, parameters = regression.create_model(args, 4444, train_X, train_y)
    y_hat = model.predict(test_X)
    performance = regression.calculate_mse(y_hat, test_y)
    print("test performance:", performance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Makes predictions for county-level COVID-19 outcomes")
    parser.add_argument("--week", metavar="week", type=int, default=2,
                        help='The week to use as the test set. Default is 2.')
    parser.add_argument("--lag", metavar="lag", type=int, default=1,
                        help='The number of weeks of lag to use for predictions (predictors come from earlier weeks).'
                             + ' Default is 1 week.')
    parser.add_argument("--window", metavar="window", type=int, default=0,
                        help='Number of weeks of autoregression to use in each windowed instance. Default is 0.')
    parser.add_argument("--prior_weeks", metavar="prior_weeks", type=int, default=1,
                        help='Number of pri1or weeks to use in the training set.'
                             + ' Default is -1, which means use all available data.')
    parser.add_argument('--model', metavar='model', type=str, default="tree",
                        help='The type of model to learn (linear, tree, forest, boost, svm, neural).'
                             ' Default is linear.')
    parser.add_argument("--k", metavar="k", type=int, default=5,
                        help='Number of folds to use in k-fold cross validation. Default is 5.')
    parser.add_argument("--outcome", metavar="outcome", type=str, default="cases_inc",
                        help="The outcome to predict (cases_inc, deaths_inc, TODO). Default is cases_inc.")
    main(parser.parse_args())