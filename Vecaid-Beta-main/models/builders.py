"""
models/builders.py — Keras model definitions.

Each builder accepts input_dim and returns a compiled Model.
All models expect a flat (batch, input_dim) input — reshaping to sequences
is done internally where required.
"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GRU,
    Input,
    Lambda,
    LSTM,
    MaxPooling1D,
    MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def _compile(model: Model, lr: float = 1e-3) -> Model:
    model.compile(optimizer=Adam(lr), loss="mean_squared_error")
    return model


def build_gru(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,), name="gru_in")
    x = Reshape((1, input_dim))(inp)
    x = GRU(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    return _compile(Model(inp, out, name="gru"))


def build_cnn_lstm(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,), name="cnn_lstm_in")
    x = Reshape((input_dim, 1))(inp)
    x = Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    return _compile(Model(inp, out, name="cnn_lstm"))


def build_cnn_lstm_v2(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,), name="cnn_lstm_v2_in")
    x = Reshape((input_dim, 1))(inp)
    x = Conv1D(32, kernel_size=5, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    return _compile(Model(inp, out, name="cnn_lstm_v2"))


def build_bilstm(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,), name="bilstm_in")
    x = Reshape((1, input_dim))(inp)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    return _compile(Model(inp, out, name="bilstm"))


def build_dnn(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,), name="dnn_in")
    x = Dense(128, activation="relu")(inp)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    return _compile(Model(inp, out, name="dnn"))


def build_transformer(input_dim: int) -> Model:
    key_dim = max(1, input_dim // 2)
    inp = Input(shape=(input_dim,), name="transformer_in")
    x = Lambda(lambda t: tf.expand_dims(t, axis=1))(inp)
    x = MultiHeadAttention(num_heads=2, key_dim=key_dim)(x, x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)
    return _compile(Model(inp, out, name="transformer"))


# ---------------------------------------------------------------------------
# Registry — iterable so ensemble code can loop over all base models
# ---------------------------------------------------------------------------
BASE_MODEL_BUILDERS: dict = {
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
    "cnn_lstm_v2": build_cnn_lstm_v2,
    "bilstm": build_bilstm,
    "dnn": build_dnn,
    "transformer": build_transformer,
}
