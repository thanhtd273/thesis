import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import random
import os
from pickle import dump
import matplotlib.pyplot as plt
import glob
import gc
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, LSTM, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model

# # Suppress TensorFlow warnings
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# # Disable GPU to avoid CUDA errors
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.append("../")
sys.path.append("../../")
sys.path.append("./source_code/")

import project_config as CONFIG

def prepare_output_directory(output_path):
    """Prepare the output directory."""
    dir_name = str(os.path.dirname(output_path))
    os.system("rm -rf " + dir_name)
    os.system("mkdir -p " + dir_name)

def load_dataset(path, chunksize=10000):
    """Load dataset in chunks to reduce memory usage."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunks.append(chunk)
    data = pd.concat(chunks)
    del chunks
    gc.collect()
    return data

def upsample_dataset(X, y, num_labels):
    """Upsample minority class to balance dataset."""
    df = pd.DataFrame(X)
    df["ATTACKED"] = y
    print(df["ATTACKED"].value_counts())
    attacked_data = df.loc[df["ATTACKED"] == 1]
    not_attacked_data = df.loc[df["ATTACKED"] == 0]
    attacked_data = resample(
        attacked_data,
        replace=True,
        n_samples=not_attacked_data.shape[0],
        random_state=10,
    )
    df = pd.concat([not_attacked_data, attacked_data])
    print(df["ATTACKED"].value_counts())
    X = np.array(df.iloc[:, 0:-num_labels])
    y = np.array(df.iloc[:, -num_labels:])
    del df, attacked_data, not_attacked_data
    gc.collect()
    return X, y

def calculate_dataset_step(dataset):
    """Calculate step size for dataset processing."""
    attack_ratios = dataset["ATTACK_RATIO"].unique()
    attack_start_times = dataset["ATTACK_START_TIME"].unique()
    attack_durations = dataset["ATTACK_DURATION"].unique()
    k_list = dataset["ATTACK_PARAMETER"].unique()
    nodes = dataset["NODE"].unique()
    dataset = dataset.sort_values(
        by=[
            "NODE", "ATTACK_RATIO", "ATTACK_START_TIME", "ATTACK_DURATION", "ATTACK_PARAMETER", "TIME",
        ]
    ).reset_index(drop=True)
    for node in nodes:
        for k in k_list:
            for attack_ratio in attack_ratios:
                for attack_start_time in attack_start_times:
                    for attack_duration in attack_durations:
                        temp = dataset.loc[
                            (dataset["NODE"] == node)
                            & (dataset["ATTACK_RATIO"] == attack_ratio)
                            & (dataset["ATTACK_START_TIME"] == attack_start_time)
                            & (dataset["ATTACK_DURATION"] == attack_duration)
                            & (dataset["ATTACK_PARAMETER"] == k)
                        ]
                        if temp.shape[0] == 0:
                            continue
                        return temp.shape[0]

def create_tf_dataset(X, y, batch_size):
    """Create a TensorFlow dataset for efficient data streaming."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def get_dataset_input_output(
    nn_model, architecture, data_type, dataset, selected_nodes_for_correlation, num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled,
):
    """Prepare input and output data for training."""
    temp_columns = []
    if use_time_hour is True:
        temp_columns.append("TIME_HOUR")
    if architecture == ["multiple_models_with_correlation", "one_model_with_correlation"]:
        for node in selected_nodes_for_correlation:
            temp_columns.append("PACKET_" + str(node))
    elif architecture in ["multiple_models_without_correlation", "one_model_without_correlation"]:
        temp_columns.append("PACKET")
    if architecture in ["one_model_with_correlation", "one_model_without_correlation"] :
        if use_onehot is True:
            for node in selected_nodes_for_correlation:
                temp_columns.append("NODE_" + str(node))
        else:
            temp_columns.append("NODE")
    temp_columns.append("ATTACKED")

    temp = dataset[temp_columns]
    X = temp.iloc[:, 0:-num_labels]
    X = np.asarray(X).astype(float)
    if data_type == "train":
        scaler = StandardScaler()
        scaler.fit_transform(X)
        dump(scaler, open(scaler_save_path, "wb"))
    del temp
    gc.collect()

    X_out = []
    y_out = []
    dataset = dataset.sort_values(
        by=[
            "NODE", "ATTACK_RATIO", "ATTACK_START_TIME", "ATTACK_DURATION", "ATTACK_PARAMETER", "TIME",
        ]
    ).reset_index(drop=True)
    dataset_step = calculate_dataset_step(dataset)

    for index_start in range(0, dataset.shape[0], dataset_step):
        temp = dataset.iloc[index_start : index_start + dataset_step, :]
        temp = temp[temp_columns]
        X = temp.iloc[:, 0:-num_labels]
        y = temp.iloc[:, -num_labels:]
        X = np.asarray(X).astype(float)
        y = np.asarray(y).astype(float)
        X = scaler.transform(X)
        for i in range(X.shape[0] - time_window + 1):
            X_out.append(X[i : i + time_window])
            y_out.append(y[i + time_window - 1])
        del temp, X, y
        gc.collect()

    X_out, y_out = np.array(X_out), np.array(y_out)
    if nn_model == "dense" or nn_model == "aen":
        X_out = X_out.reshape((X_out.shape[0], X_out.shape[1] * X_out.shape[2]))
    if (data_type == "train") and (upsample_enabled is True):
        X_out, y_out = upsample_dataset(X_out, y_out, num_labels)
    return X_out, y_out, scaler

def autoencoder(input_shape):
    """
    Generate the neural network model
    Args:
        input_shape: the input shape of the dataset given to the model

    Returns:
        The neural network model

    """
    # encoder
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(input_shape,), activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(16, activation="relu", trainable=False))
    model.add(tf.keras.layers.BatchNormalization())

    # decoder
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(input_shape, activation="relu"))

    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=[tf.keras.metrics.Accuracy()],
        run_eagerly=True,
    )
    model.summary()
    return model

def setup_aen_model(
    nn_model, architecture, train_dataset, validation_dataset, selected_nodes_for_correlation, num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled,
):
    """Setup and train autoencoder for AEN model."""
    train_dataset_benign = train_dataset.loc[(train_dataset["ATTACK_RATIO"] == 0)]
    validation_dataset_benign = validation_dataset.loc[
        (validation_dataset["ATTACK_RATIO"] == 0)
    ]
    X_train, y_train, scaler = get_dataset_input_output(
        nn_model, architecture, "train", train_dataset_benign, selected_nodes_for_correlation, num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled,
    )
    X_validation, y_validation, _ = get_dataset_input_output(
        nn_model, architecture, "validation", validation_dataset_benign, selected_nodes_for_correlation, num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled,
    )

    tf.keras.backend.clear_session()
    aen_model = autoencoder(X_train.shape[1])
    epochs_aen = 10
    batch_size_aen = 8

    train_dataset_tf = create_tf_dataset(X_train, X_train, batch_size_aen)
    validation_dataset_tf = create_tf_dataset(X_validation, X_validation, batch_size_aen)
    
    history = aen_model.fit(
        train_dataset_tf,
        validation_data=validation_dataset_tf,
        epochs=epochs_aen,
        verbose=1,
    )
    del X_train, y_train, X_validation, y_validation, train_dataset_tf, validation_dataset_tf
    gc.collect()
    return aen_model

def trans_encoder(inputs, head_size, n_heads, ff_dim, drop=0.0):
    """Transformer encoder layer."""
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=n_heads, dropout=drop
    )(x, x)
    x = Dropout(drop)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(drop)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def create_nn_model(
    nn_model, architecture, input_shape, output_shape, output_bias=None
):
    """Create neural network model."""
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    l2_regularizer_val = 0
    model = Sequential()
    optimizer = "adam"
    match nn_model:
        case "dense":
            if architecture in ["multiple_models_with_correlation", "one_model_with_correlation"]:
                l2_regularizer_val = 0.0  # dense use to have 4 neurons in previous runs
                dropout_ratio = 0.3
            else:
                l2_regularizer_val = 0.0
                dropout_ratio = 0.0
            model.add(Input(shape=(input_shape,)))
            model.add(Dense(5, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer_val)))
            model.add(Dropout(dropout_ratio))
            model.add(Dense(output_shape, activation="sigmoid", bias_initializer=output_bias
                )
            )
        case "cnn":
            if architecture in ["multiple_models_with_correlation", "one_model_with_correlation"]:
                l2_regularizer_val = 0.0
                dropout_ratio = 0.3
            else:
                l2_regularizer_val = 0.0
                dropout_ratio = 0.0
            model.add(Input(shape=input_shape))
            model.add(Conv1D( filters=5, kernel_size=3, activation="relu", 
                             kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer_val)))
            model.add(Dropout(dropout_ratio))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(output_shape, activation="sigmoid", bias_initializer=output_bias))
        case "lstm":
            if architecture in ["multiple_models_with_correlation", "one_model_with_correlation"]:
                l2_regularizer_val = 0.3
            else:
                l2_regularizer_val = 0.0
            model.add(Input(shape=input_shape))
            model.add(LSTM(4, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer_val)))
            model.add(Dense(output_shape, activation="sigmoid", bias_initializer=output_bias))
        case "aen":
            model.add(Input(shape=(input_shape,)))
            model.add(
                Dense(64, activation="relu", trainable=False)
            )
            model.add(tf.keras.layers.BatchNormalization(trainable=False))
            model.add(Dense(32, activation="relu", trainable=False))
            model.add(tf.keras.layers.BatchNormalization(trainable=False))
            model.add(Dense(16, activation="relu", trainable=False))
            model.add(tf.keras.layers.BatchNormalization(trainable=False))
            model.add(Dense(8, activation="relu"))
            model.add(Dense(output_shape, activation="sigmoid", bias_initializer=output_bias))
        case "trans":
            if architecture in ["multiple_models_with_correlation", "one_model_with_correlation"]:
                l2_regularizer_val = 0.01
            else:
                l2_regularizer_val = 0.0
            head_size = 1
            n_heads = 1
            ff_dim = 2
            drop = 0.0
            mlp_drop = 0.4
            inpts = Input(shape=input_shape)
            x = inpts
            x = MultiHeadAttention(
                key_dim=head_size,
                num_heads=n_heads,
                dropout=drop,
                kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer_val),
            )(x, x)
            x = GlobalAveragePooling1D(data_format="channels_first")(x)
            oupts = Dense(
                1, activation="sigmoid", bias_initializer=output_bias
            )(x)
            model = Model(inpts, oupts)

    metrics = [
        tf.keras.metrics.Accuracy(),
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.BinaryCrossentropy(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives(),
    ]

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    model.summary()
    return model

class ClearMemory(tf.keras.callbacks.Callback):
    """Clear memory after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def setup_callbacks(saved_model_path):
    """Reduced checkpoint frequency to save only the best model."""
    checkpoint_path = saved_model_path + "checkpoints/best/weights.weights.h5"
    prepare_output_directory(checkpoint_path)
    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
    log_path = saved_model_path + "logs/logs.csv"
    prepare_output_directory(log_path)
    csv_logger = tf.keras.callbacks.CSVLogger(log_path, separator=",", append=False)
    callbacks = [cp, csv_logger, ClearMemory()]
    return callbacks

def plot_logs(logs_path, output_path):
    """Plot training metrics."""
    logs = pd.read_csv(logs_path)
    metrics = logs.columns.values
    new_metrics = {}
    for metric in metrics:
        if metric[-2] == "_":
            new_metrics[metric] = metric[:-2]
        elif metric[-3] == "_":
            new_metrics[metric] = metric[:-3]
    logs = logs.rename(new_metrics, axis="columns")
    metrics = logs.columns.values
    for metric in metrics:
        if metric == "epoch" or "val" in metric:
            continue
        plt.clf()
        plt.plot(logs["epoch"], logs[metric], label="Train")
        plt.plot(logs["epoch"], logs["val_" + metric], label="Validation")
        plt.xlabel("Epoch Number")
        plt.ylabel(metric)
        plt.title(metric + " vs epoch")
        plt.legend()
        plt.savefig(output_path + metric + ".png")

def main_plot_logs(metadata_metric, nn_model, architecture, run_number, group_number):
    """Plot logs for all saved models."""
    all_saved_models_path = (
        CONFIG.OUTPUT_DIRECTORY + "nn_training/group_" + str(group_number) + "/metadata_" + metadata_metric + "/" + nn_model + "/" + architecture + "/" + "run_" + str(run_number) + "/saved_model/*"
    )
    for directory in glob.glob(all_saved_models_path):
        print(directory)
        logs_path = directory + "/logs/logs.csv"
        output_path = directory + "/logs/pics/"
        prepare_output_directory(output_path)
        plot_logs(logs_path, output_path)

def main_train_model(
    nn_model, architecture, use_time_hour, use_onehot, upsample_enabled, train_dataset_path, validation_dataset_path, time_window, num_labels, epochs, batch_size, use_metadata, metadata_path, metadata_metric, num_selected_nodes_for_correlation, run_number, group_number,
):
    """Train model for a specific configuration."""
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    train_dataset_all = load_dataset(train_dataset_path)
    validation_dataset_all = load_dataset(validation_dataset_path)

    model_output_path = (
        CONFIG.OUTPUT_DIRECTORY + "nn_training/group_" + str(group_number) + "/metadata_" + metadata_metric + "/" + nn_model + "/" + architecture + "/" + "run_" + str(run_number) + "/saved_model/"
    )
    prepare_output_directory(model_output_path)

    nodes = []
    if architecture in ["multiple_models_with_correlation", "multiple_models_without_correlation"]:
        nodes = list(train_dataset_all["NODE"].unique())
    elif architecture in ["one_model_with_correlation", "one_model_without_correlation"]:
        nodes = ["one_model"]

    for node in nodes:
        saved_model_path = model_output_path + str(node) + "/"
        scaler_save_path = model_output_path + str(node) + "/scaler.pkl"
        prepare_output_directory(saved_model_path)

        train_dataset = pd.DataFrame()
        validation_dataset = pd.DataFrame()
        if architecture in ["multiple_models_with_correlation", "multiple_models_without_correlation"]:
            train_dataset = train_dataset_all.loc[(train_dataset_all["NODE"] == node)]
            validation_dataset = validation_dataset_all.loc[
                (validation_dataset_all["NODE"] == node)
            ]
        elif architecture in ["one_model_with_correlation", "one_model_without_correlation"]:
            train_dataset = train_dataset_all
            validation_dataset = validation_dataset_all

        selected_nodes_for_correlation = list(train_dataset_all["NODE"].unique())
        if (architecture == "multiple_models_with_correlation" and use_metadata is True 
            and num_selected_nodes_for_correlation < len(selected_nodes_for_correlation)):
            selected_nodes_for_correlation = [node]
            match metadata_metric:
                case "RANDOM":
                    rest_of_the_nodes = list(train_dataset_all["NODE"].unique())
                    rest_of_the_nodes.remove(node)
                    selected_nodes_for_correlation.extend(
                        random.choices(
                            rest_of_the_nodes, k=num_selected_nodes_for_correlation - 1
                        )
                    )
                
                case "SHAP":
                    metadata_dataset_all = load_dataset(metadata_path)
                    metadata_dataset = metadata_dataset_all.loc[
                        (metadata_dataset_all["node"] == node)
                    ]
                    selected_nodes_for_correlation = list(
                        metadata_dataset["feature_node"][
                            0:num_selected_nodes_for_correlation
                        ].values
                    )
                case "DISTANCE":
                    metadata_dataset_all = load_dataset(metadata_path)
                    metadata_dataset = metadata_dataset_all.loc[
                        (metadata_dataset_all["NODE_1"] == node)
                    ]
                    metadata_dataset = metadata_dataset.sort_values(
                        by=[metadata_metric], ascending=True
                    )
                    selected_nodes_for_correlation.extend(
                        list(
                            metadata_dataset["NODE_2"][
                                0 : num_selected_nodes_for_correlation - 1
                            ].values
                        )
                    )
                case "CORRELATION":
                    metadata_dataset_all = load_dataset(metadata_path)
                    metadata_dataset = metadata_dataset_all.loc[
                        (metadata_dataset_all["NODE_1"] == node)
                    ]
                    metadata_dataset = metadata_dataset.sort_values(
                        by=[metadata_metric], ascending=False
                    )
                    selected_nodes_for_correlation.extend(
                        list(
                            metadata_dataset["NODE_2"][
                                0 : num_selected_nodes_for_correlation - 1
                            ].values
                        )
                    )
        selected_nodes_save_path = (
            model_output_path + str(node) + "/selected_nodes_for_correlation.pkl"
        )
        dump(selected_nodes_for_correlation, open(selected_nodes_save_path, "wb"))

        scaler = StandardScaler()
        X_train, y_train, scaler = get_dataset_input_output(
            nn_model, architecture, "train", train_dataset, selected_nodes_for_correlation, num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled,
        )
        X_validation, y_validation, _ = get_dataset_input_output(
            nn_model, architecture, "validation", validation_dataset, selected_nodes_for_correlation, num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled,
        )

        train_dataset_tf = create_tf_dataset(X_train, y_train, batch_size)
        validation_dataset_tf = create_tf_dataset(X_validation, y_validation, batch_size)

        input_shape = 0
        output_shape = 0
        if nn_model in ["dense", "aen"]:
            input_shape = X_train.shape[1]
            output_shape = y_train.shape[1]
        elif nn_model in ["cnn", "lstm", "trans"]:
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_shape = y_train.shape[1]

        neg, pos = np.bincount(train_dataset["ATTACKED"].astype(int))
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        initial_bias = np.log([pos / neg])

        if nn_model == "aen":
            aen_model = setup_aen_model(
                nn_model, architecture, train_dataset, validation_dataset, selected_nodes_for_correlation, num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled,
            )

        tf.keras.backend.clear_session()
        model = create_nn_model(
            nn_model, architecture, input_shape, output_shape, initial_bias
        )
        if nn_model == "aen" and aen_model is not None:
            for l1, l2 in zip(model.layers[0:6], aen_model.layers[0:6]):
                l1.set_weights(l2.get_weights())

        callbacks_list = setup_callbacks(saved_model_path)
        model.fit(
            train_dataset_tf,
            validation_data=validation_dataset_tf,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks_list,
            class_weight=class_weight,
        )
        model.save(saved_model_path + "final_model.keras") 
        print(f"Model saved to {saved_model_path}final_model.keras")
        del X_train, y_train, X_validation, y_validation, train_dataset_tf, validation_dataset_tf, model
        gc.collect()

def main_train(group_number):
    """Train models for a group."""
    use_time_hour = False
    use_onehot = True
    upsample_enabled = False
    epochs = 3
    batch_size = 32
    num_labels = 1
    time_window = 10
    train_dataset_path = f"{CONFIG.OUTPUT_DIRECTORY}pre_process/Output/group_{str(group_number)}/train_data/train_data.csv"
    validation_dataset_path = f"{CONFIG.OUTPUT_DIRECTORY}pre_process/Output/group_{str(group_number)}/validation_data/validation_data.csv"
    print(sys.argv)

    if len(sys.argv) > 1:
        # Extract all arguments
        nn_model = sys.argv[1]
        architecture = sys.argv[2]
        use_metadata = sys.argv[3] == "True"
        metadata_path = sys.argv[4]
        metadata_metric = sys.argv[5]
        num_selected_nodes_for_correlation = int(sys.argv[6])
        run_number = int(sys.argv[7])
        use_time_hour = sys.argv[8] == "True"
        use_onehot = sys.argv[9] == "True"
        upsample_enabled = sys.argv[10] == "True"
        epochs = int(sys.argv[11])
        batch_size = int(sys.argv[12])
        num_labels = int(sys.argv[13])
        time_window = int(sys.argv[14])
        if metadata_metric == "SHAP":
            metadata_path = f"{CONFIG.OUTPUT_DIRECTORY}nn_training/group_{str(group_number)}/metadata_NOT_USED/{nn_model}/{architecture}/run_{str(run_number)}/report/shap/feature_importance/feature_importance.csv"
        else:
            metadata_path = metadata_path.replace("Output/metadata", f"Output/group_{str(group_number)}/metadata")
        
        main_train_model(
            nn_model, architecture, use_time_hour, use_onehot, upsample_enabled, train_dataset_path, 
            validation_dataset_path, time_window, num_labels, epochs, batch_size, use_metadata, metadata_path, 
            metadata_metric, num_selected_nodes_for_correlation, run_number, group_number,
        )
        main_plot_logs(
            metadata_metric, nn_model, architecture, run_number, group_number
        )
    else:
        nn_model_list = ["dense"]
        architecture_list = ["multiple_models_with_correlation"]
        use_metadata = False
        metadata_path = CONFIG.OUTPUT_DIRECTORY
        metadata_metric = "NOT_USED"
        num_selected_nodes_for_correlation = 0
        run_number = 0
        for nn_model in nn_model_list:
            for architecture in architecture_list:
                main_train_model(
                    nn_model, architecture, use_time_hour, use_onehot, upsample_enabled, train_dataset_path, 
                    validation_dataset_path, time_window, num_labels, epochs, batch_size, use_metadata, metadata_path, 
                    metadata_metric, num_selected_nodes_for_correlation, run_number, group_number,
                )
                main_plot_logs(metadata_metric, nn_model, architecture, run_number, group_number)
        sys.exit()

def main(num_groups):
    """Main function to train models for all groups."""
    # for group_number in range(num_groups):
    main_train(80)

if __name__ == "__main__":
    main(CONFIG.NUM_GROUPS)