import pandas as pd
import tensorflow as tf
import keras
from keras import Model
from keras.layers import Conv2D, Masking, BatchNormalization, Activation, Dense
from keras import Sequential
from matplotlib import pyplot as plt
from IPython.display import clear_output
from utils.GetData import *
from utils.GetDataUtils import *
from utils.PostProcessing import *
from utils.PostProcessingUtils import *

# Hyperparameters
batch_size = 32
epochs = 100

# File paths
model_filepath = None
trainlog_filepath = None

"""
This class the custom CNN.

Attributes:
  input_shape ((n, n, 8)): The shape of the inputs.

Methods:
  call(self, inputs): Forward pass.
"""
class CustomCNN(Model):
    def __init__(self, input_shape):
        super(CustomCNN, self).__init__()

        self.masking = Masking(mask_value=-1, input_shape=input_shape)
        self.conv1 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            use_bias=False,
        )
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            use_bias=False,
        )
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(
            filters=2,
            kernel_size=(1, 1),
            activation="relu",
            padding="same",
            use_bias=False,
        )
        self.conv4 = Conv2D(
            filters=1, kernel_size=(1, 1), padding="same", use_bias=False
        )
        self.activation = Activation(tf.nn.softmax)

    def call(self, inputs):
        x = self.masking(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.activation(x)


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if "val" not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), self.metrics[metric], label=metric)
            if logs["val_" + metric]:
                axs[i].plot(
                    range(1, epoch + 2),
                    self.metrics["val_" + metric],
                    label="val_" + metric,
                )

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()

def get_data_from_file(data_path):
    data_path = 'CODE/MODELS/JET-RNA/bprna_df.csv'
    bprna_df = pd.read_csv(data_path)
    X, y = get_data(data=data_path, num_samples=5000, max_len=200, random_state=7, create_tensor_sequence=True, create_tensor_structure=True)
    X_test, y_test, test_lengths = get_data(data=data_path, columns=['length'], num_samples=10, random_state=4, max_len=200, create_tensor_sequence=True, create_tensor_structure=True)
    
    return X, y, X_test, y_test, test_lengths


def train_custom_cnn(
    X_train,
    y_train,
    X_test,
    y_test,
):
    input_shape = X_train[0, :, :, :].shape
    model = CustomCNN(input_shape=input_shape)

    ### Callbacks for monitoring and managing the training process
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10, min_delta=0.001, restore_best_weights=True
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        f"/model_checkpoints/valloss-{epoch}.h5",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )

    csv_logger = keras.callbacks.CSVLogger(
        f"/trainlogs/trainlog-{epoch}.h5", separator=",", append=False
    )

    # Compile and train the model
    model.compile(optimizer="adam", loss="mae", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=10)

    callbacks_list = [PlotLearning(), early_stopping]

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=True,
        # class_weight=class_weight,
        callbacks=callbacks_list,
    )

    history_df = pd.DataFrame(history.history)

    history_df.head()


def test_custom_cnn(X_test, y_test, model):
    loss, metric_values = model.evaluate(X_test, y_test)

    print("Loss:", loss)
    print("Metrics:", metric_values)

if __name__ == "__main__":
    