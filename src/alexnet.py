import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
import tensorflow as tf
import torch
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
    image = tf.divide(tf.cast(image, tf.float32), 255.0)
    assert isinstance(image, tf.Tensor)
    return image, label


def resize(
    image: jt.Float[tf.Tensor, "h w c"], label: jt.Int[tf.Tensor, ""]
) -> tuple[jt.Float[tf.Tensor, "c h w"], jt.Int[tf.Tensor, ""]]:
    img = tf.image.resize(image, (224, 224))
    img = tf.transpose(img, perm=[2, 0, 1])
    return img, label


train_dataset = train_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)

SHUFFLE_VAL = len(train_dataset) // 1000
BATCH_SIZE = 4

train_dataset = train_dataset.shuffle(SHUFFLE_VAL)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


test_dataset = test_dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


train_dataset = tfds.as_numpy(train_dataset)
test_dataset = tfds.as_numpy(test_dataset)


class LocalResponseNormalization(eqx.Module):
    k: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, k=2, n=5, alpha=1e-4, beta=0.75) -> None:
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: jt.Float[jt.Array, "c h w"]) -> jt.Float[jt.Array, "c h w"]:
        c, _, _ = x.shape
        p = jnp.pad(x, pad_width=[(self.n // 2, self.n // 2), (0, 0), (0, 0)])

        # def _body(_, i):
        #     window = jax.lax.dynamic_slice_in_dim(p, i, self.n) ** 2
        #     d = (jnp.einsum("ijk->jk", window) * self.alpha + self.k) ** self.beta
        #     b = x[i] / d
        #     return _, b

        # _, ys = jax.lax.scan(_body, None, jnp.arange(c))
        # return ys

        def _body(i):
            window = jax.lax.dynamic_slice_in_dim(p, i, self.n) ** 2
            d = (jnp.einsum("ijk->jk", window) * self.alpha + self.k) ** self.beta
            b = x[i] / d
            return b

        ys = eqx.filter_vmap(_body)(jnp.arange(c))
        return ys


class AlexNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    conv5: eqx.nn.Conv2d
    lrn1: LocalResponseNormalization
    lrn2: LocalResponseNormalization
    max_pool_1: eqx.nn.MaxPool2d
    max_pool_2: eqx.nn.MaxPool2d
    max_pool_3: eqx.nn.MaxPool2d

    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear

    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    final: eqx.nn.Linear

    def __init__(self, *, key: jt.PRNGKeyArray):
        _, *subkeys = jax.random.split(key, 10)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            key=subkeys[0],
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, key=subkeys[1]
        )
        self.conv3 = eqx.nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, key=subkeys[2]
        )
        self.conv4 = eqx.nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, key=subkeys[3]
        )
        self.conv5 = eqx.nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, key=subkeys[4]
        )

        self.lrn1 = LocalResponseNormalization()
        self.lrn2 = LocalResponseNormalization()
        self.max_pool_1 = eqx.nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool_2 = eqx.nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool_3 = eqx.nn.MaxPool2d(kernel_size=3, stride=2)

        self.dense1 = eqx.nn.Linear(in_features=256, out_features=4096, key=subkeys[5])
        self.dense2 = eqx.nn.Linear(in_features=4096, out_features=4096, key=subkeys[6])

        self.dropout1 = eqx.nn.Dropout()
        self.dropout2 = eqx.nn.Dropout()

        self.final = eqx.nn.Linear(in_features=4096, out_features=1, key=subkeys[7])

    def __call__(
        self, x: jt.Float[jt.Array, "3 224 224"], key: jt.PRNGKeyArray
    ) -> jt.Array:
        key, subkey = jax.random.split(key)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.lrn1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = self.lrn2(x)
        x = self.max_pool_2(x)

        x = self.conv3(x)
        x = jax.nn.relu(x)
        x = self.conv4(x)
        x = jax.nn.relu(x)
        x = self.conv5(x)
        x = jax.nn.relu(x)

        x = self.max_pool_3(x)
        x = jnp.ravel(x)

        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dropout1(x, key=key)
        x = self.dense2(x)
        x = jax.nn.relu(x)
        x = self.dropout2(x, key=subkey)

        x = self.final(x)
        return x


def loss_fn(
    alexnet: AlexNet,
    x: jt.Float[jt.Array, "batch_size 3 224 224"],
    y: jt.Float[jt.Array, "batch_size 1"],
    key: jt.PRNGKeyArray,
) -> tuple[jt.Array, jt.Array]:
    k, _ = jax.random.split(key)
    logits = eqx.filter_vmap(alexnet, in_axes=(0, None))(x, k)
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    return jnp.mean(loss), logits


@eqx.filter_jit
def step(
    alexnet,
    x: jt.Float[jt.Array, "batch_size 3 224 224"],
    y: jt.Float[jt.Array, "batch_size 1"],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jt.PRNGKeyArray,
):
    (loss_value, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        alexnet, x, y, key
    )
    updates, opt_state = optimizer.update(grads, opt_state, alexnet)
    alexnet = eqx.apply_updates(alexnet, updates)
    return alexnet, opt_state, loss_value, logits


def eval(alexnet: AlexNet, test_dataset, key: jt.PRNGKeyArray):
    """
    Evaluate model on test dataset, returning average loss and accuracy
    """
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for x, y in test_dataset:
        y = y.reshape(-1, 1)
        key, subkey = jax.random.split(key)

        # Get loss
        loss, _ = loss_fn(alexnet, x, y, subkey)

        # Get predictions
        preds = eqx.filter_vmap(alexnet, in_axes=(0, None))(x, subkey)
        pred_classes = (jax.nn.sigmoid(preds) > 0.5).astype(jnp.int32)
        correct = jnp.sum(pred_classes == y)

        # Accumulate batch stats
        batch_size = len(y)
        total_loss += loss * batch_size
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


alexnet = AlexNet(key=jax.random.key(42))
learning_rate = 0.0001
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(alexnet, eqx.is_array))

key = jax.random.key(99)
n_epochs = 10
for epoch in range(n_epochs):
    batch_count = len(train_dataset)
    avg_train_loss = 0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(enumerate(train_dataset), total=batch_count, desc=f"Epoch {epoch}")
    for i, (x, y) in pbar:
        y = y.reshape(-1, 1)
        key, subkey = jax.random.split(key)
        alexnet, opt_state, loss, logits = step(
            alexnet, x, y, optimizer, opt_state, key
        )

        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
        correct_predictions += jnp.sum(preds == y)
        total_samples += len(y)
        avg_train_loss += loss

        current_accuracy = correct_predictions / total_samples
        pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{current_accuracy:.4f}"})

    avg_train_loss /= len(train_dataset)
    train_accuracy = correct_predictions / total_samples

    key, subkey = jax.random.split(key)
    test_loss, test_acc = eval(alexnet, test_dataset, subkey)
    print(
        f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, train_acc={train_accuracy:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
    )
