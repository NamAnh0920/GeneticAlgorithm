import numpy as np
import tensorflow as tf

CHECKPOINTS_FOLDER = "/work/CovidOutcomes/checkpoints/"

def setup():
    # from https://www.tensorflow.org/guide/gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def create_neural_hyperparameter_sets(n_features):
    hyperparameter_sets = []

    for learning_rate in [0.0001, 0.001, 0.01, 0.1]:
        for layers in [1, 2, 3]:
            for neurons in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                if neurons > n_features // 8 and neurons < n_features * 3:
                    hyperparameter_combo = (layers, neurons, learning_rate)
                    hyperparameter_sets.append(hyperparameter_combo)

    return hyperparameter_sets

def build_neural(hyperparameter_combo, seed, n_features):
    # set the random seed
    tf.random.set_seed(seed)

    num_layers = hyperparameter_combo[0]
    num_neurons = hyperparameter_combo[1]
    learning_rate = hyperparameter_combo[2]

    # create the hidden layers
    layers = []
    for layer_i in range(num_layers):
        if layer_i == 0:
            layer = tf.keras.layers.Dense(units=num_neurons, input_dim=n_features, activation="relu")
        else:
            layer = tf.keras.layers.Dense(units=num_neurons, activation="relu")
        layers.append(layer)

        layer = tf.keras.layers.Dropout(0.5, seed=seed)
        layers.append(layer)

    # add the output layer
    layer = tf.keras.layers.Dense(units=1)
    layers.append(layer)

    # create the model
    model = tf.keras.Sequential(layers)

    # setup the training
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=["mean_squared_error"])

    return model


def fit(model, X, y, checkpoint_file):
    checkpoint_path = CHECKPOINTS_FOLDER + checkpoint_file

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=25)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", verbose=0,
                                                     save_best_only=True)
    training_history = model.fit(X, y, validation_split=0.2, epochs=1000, verbose=0, #2,
                                 callbacks=[early_stop, checkpoints])

    return training_history
