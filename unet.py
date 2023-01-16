#!/usr/bin/env python3
import os
import datetime

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPool2D,
)
from tensorflow.keras.metrics import BinaryIoU, MeanIoU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from utils import prep_data

# Config
BASE_LOGS_DIR = "logs"
DATA_DIR = "../../data/"
SET_NAMES = ["simp_imp"]
MASK_VALUE = -1

# Hyperparams
N_FOLDS = 10
EPOCHS = 50
ETA = 1e-2
BATCH_SIZE = 16

tf.get_logger().setLevel("ERROR")


# Construct the U-Net
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def data_augmentation():
    augs = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.5),
            # keras.layers.RandomContrast(0.1),
            # keras.layers.RandomBrightness(0.1),
        ]
    )
    return augs


def build_unet(input_shape, aug=False):
    inputs = Input(input_shape)
    if aug:
        inputs = data_augmentation()(inputs)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="U-Net")
    return model


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, MASK_VALUE), K.floatx())
    y_true = K.cast(y_true, K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def callbacks(name):
    cb = []
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOGS_DIR = os.path.join(BASE_LOGS_DIR, f"{now}_{name}")
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1,
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(LOGS_DIR, f"{name}_best_wg.h5"),
        save_best_only=True,
        mode="min",
        monitor="val_loss",
        save_weights_only=True,
        verbose=1,
    )
    lr_decay = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=1,
        min_lr=0.00001,
    )
    log = keras.callbacks.CSVLogger(os.path.join(LOGS_DIR, f"{now}_{name}_history.csv"))
    cb.append(tensorboard)
    cb.append(checkpoint)
    cb.append(lr_decay)
    cb.append(log)
    return cb


def train_unet(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size=16,
    epochs=100,
    eta=1e-2,
    cb=None,
    masked_loss=False,
):
    input_shape = X_train.shape[1:]
    model = build_unet(input_shape)
    batch_size = batch_size
    epochs = epochs
    if masked_loss:
        loss = masked_loss_function
        metrics = MeanIoU(num_classes=3, ignore_class=MASK_VALUE)
    else:
        loss = "binary_crossentropy"
        metrics = BinaryIoU(target_class_ids=[1], threshold=0.5)
    model.compile(
        optimizer=Adam(learning_rate=eta),
        loss=loss,
        metrics=[metrics],
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=cb,
    )
    return model, history


def train_set(set_name, ws_rgb, ws_label):
    X_train, y_train, X_test, y_test = prep_data(DATA_DIR, "hand", ws_rgb, ws_label)

    # MODEL
    # Data structure
    stats = ["loss", "val_loss", "iou", "val_iou", "mean_biou", "tree_biou", "bg_biou"]

    data = np.zeros((N_FOLDS, len(stats)), dtype=object)

    # Initialize the KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=7)

    for i, (itrain, itest) in enumerate(
        tqdm(
            kf.split(
                X_train,
                y_train,
            ),
            desc="K-Folds",
            position=0,
            leave=True,
            total=N_FOLDS,
        )
    ):
        X_train_fold = X_train[itrain]
        y_train_fold = y_train[itrain]
        X_test_fold = X_train[itest]
        y_test_fold = y_train[itest]

        # Set callbacks each iteration so that logs are stored in new
        # directory
        cb = callbacks(f"KF{i+1}of{N_FOLDS}_{set_name}")
        model, history = train_unet(
            X_train_fold,
            y_train_fold,
            X_test_fold,
            y_test_fold,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            eta=ETA,
            cb=cb,
        )

        # Loss and accuracies from each epoch
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        iou = history.history[list(history.history.keys())[1]]
        val_iou = history.history[list(history.history.keys())[3]]

        # Test the model on the preserved test data
        y_pred = model.predict(X_test)

        # Get the IoU for the test data
        biou = BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
        biou.update_state(y_pred=y_pred, y_true=y_test)
        pred_biou = biou.result().numpy()

        # only for trees
        tree_biou = BinaryIoU(target_class_ids=[1], threshold=0.5)
        tree_biou.update_state(y_pred=y_pred, y_true=y_test)
        pred_tree_biou = tree_biou.result().numpy()

        # only for non-tree pixel (background)
        bg_biou = BinaryIoU(target_class_ids=[0], threshold=0.5)
        bg_biou.update_state(y_pred=y_pred, y_true=y_test)
        pred_bg_biou = bg_biou.result().numpy()

        # Log the five stats according to their K-Fold and parameter iteration
        stats = [
            loss,
            val_loss,
            iou,
            val_iou,
            pred_biou,
            pred_tree_biou,
            pred_bg_biou,
        ]

        for s, stat in enumerate(stats):
            data[i, s] = stat

        # Save the updated array each iteration
        np.save(f"stats/{N_FOLDS}folds_CV_{set_name}.npy", data)


if __name__ == "__main__":
    for set_name in tqdm(SET_NAMES, desc="Set", total=len(SET_NAMES)):
        train_set(set_name, ws_rgb="strict/simple_imp", ws_label="strict")
