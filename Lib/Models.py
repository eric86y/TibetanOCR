import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import (
    Add,
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    Multiply,
    RandomRotation,
    RandomZoom
)

from keras.optimizers import Adam

"""
Vanilla Easter2 from: https://github.com/kartikgill/Easter2/blob/main/src/easter_model.py
"""


BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997


data_augmentation = keras.Sequential(
    [
        RandomRotation(factor=(0.01, 0.02)),
        RandomZoom(0.1, 0.3),
    ]
)


def ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_custom(args):
    """
    custom CTC loss
    """
    y_pred, labels, input_length, label_length = args
    ctc_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    p = tf.exp(-ctc_loss)
    gamma = 0.5
    alpha = 0.25
    return alpha * (K.pow((1 - p), gamma)) * ctc_loss


def batch_norm(inputs):
    return BatchNormalization(
        momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON
    )(inputs)


def add_global_context(data, filters):
    pool = GlobalAveragePooling1D()(data)
    pool = Dense(filters // 8, activation="relu")(pool)
    pool = Dense(filters, activation="sigmoid")(pool)
    final = Multiply()([data, pool])
    return final


def easter_unit(old, data, filters, kernel, stride, dropouts):
    old = Conv1D(filters=filters, kernel_size=(1), strides=(1), padding="same")(old)
    old = batch_norm(old)

    this = Conv1D(filters=filters, kernel_size=(1), strides=(1), padding="same")(data)
    this = batch_norm(this)

    old = Add()([old, this])


    data = Conv1D(
        filters=filters, kernel_size=(kernel), strides=(stride), padding="same"
    )(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(dropouts)(data)


    data = Conv1D(
        filters=filters, kernel_size=(kernel), strides=(stride), padding="same"
    )(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(dropouts)(data)

    data = Conv1D(
        filters=filters, kernel_size=(kernel), strides=(stride), padding="same"
    )(data)

    data = batch_norm(data)
    data = add_global_context(data, filters)

    final = Add()([old, data])

    data = Activation("relu")(final)
    data = Dropout(dropouts)(data)

    return data, old


def Easter2(
    weights: str = "None",
    inptu_width: int = 2000,
    input_height: int = 80,
    vocab_size: int = 64,
    learning_rate: float = 0.001,
    max_sequence_length: int = 500,
):
    input_data = Input(name="the_input", shape=(inptu_width, input_height))

    data = data_augmentation(input_data)
    data = Conv1D(filters=128, kernel_size=(3), strides=(2), padding="same")(input_data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)

    data = Conv1D(filters=128, kernel_size=(3), strides=(2), padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)

    old = data

    data, old = easter_unit(old, data, 128, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 7, 1, 0.2)
    data, old = easter_unit(old, data, 256, 9, 1, 0.3)

    data = Conv1D(
        filters=512, kernel_size=(11), strides=(1), padding="same", dilation_rate=2
    )(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)

    data = Conv1D(filters=512, kernel_size=(1), strides=(1), padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)

    data = Conv1D(filters=vocab_size, kernel_size=(1), strides=(1), padding="same")(
        data
    )

    y_pred = Activation("softmax", name="Final")(data)

    Optimizer = Adam(learning_rate=learning_rate)

    labels = Input(name="the_labels", shape=[max_sequence_length], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    output = Lambda(ctc_custom, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(
        inputs=[input_data, labels, input_length, label_length], outputs=output
    )

    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=Optimizer)

    if weights != "None":
        model.load_weights(weights)

    return model
