import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
import tensorflow as tf
import torch
from keras import callbacks, layers, losses, models, optimizers
from tqdm import tqdm

import tensorflow_datasets as tfds

(train_dataset, test_dataset), info = tfds.load(
    "cats_vs_dogs",
    split=("train[:80%]", "train[80%:]"),
    with_info=True,
    as_supervised=True,
)  # pyright:ignore

# print(info)


def normalize(
    image: jt.Float[tf.Tensor, "h w c"], label: jt.Int[tf.Tensor, ""]
) -> tuple[jt.Float[tf.Tensor, "h w c"], jt.Int[tf.Tensor, ""]]:
    return (tf.cast(image, tf.float32) / 255.0, label)  # pyright: ignore


def resize(
    image: jt.Float[tf.Tensor, "h w c"], label: jt.Int[tf.Tensor, ""]
) -> tuple[jt.Float[tf.Tensor, "c h w"], jt.Int[tf.Tensor, ""]]:
    img = tf.image.resize(image, (224, 224))
    # img = tf.transpose(img, perm=[2, 0, 1])
    return img, label


train_dataset = train_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

SHUFFLE_VAL = len(train_dataset) // 1000
BATCH_SIZE = 4

train_dataset = train_dataset.shuffle(SHUFFLE_VAL)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


test_dataset = test_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# train_dataset = tfds.as_numpy(train_dataset)
# test_dataset = tfds.as_numpy(test_dataset)


class LRN(layers.Layer):
    def call(self, x):
        return tf.nn.local_response_normalization(x, 2, bias=2, alpha=1e-4, beta=0.75)


def alexnet():
    inp = layers.Input((224, 224, 3))
    x = layers.Conv2D(96, 11, 4, activation="relu")(inp)
    x = LRN()(x)
    # x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(3, 2)(x)
    x = layers.Conv2D(256, 5, 1, activation="relu")(x)
    x = LRN()(x)
    # x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(3, 2)(x)

    x = layers.Conv2D(384, 3, 1, activation="relu")(x)
    x = layers.Conv2D(384, 3, 1, activation="relu")(x)
    x = layers.Conv2D(256, 3, 1, activation="relu")(x)

    x = layers.MaxPool2D(3, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inp, outputs=x)
    return model


model = alexnet()

model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(learning_rate=0.0001),  # pyright: ignore
    metrics=["accuracy"],
)
es = callbacks.EarlyStopping(patience=5, monitor="loss")


model.fit(train_dataset, epochs=100, validation_data=test_dataset, callbacks=[es])
