from collections.abc import Callable
from typing import Any, ClassVar, Hashable, Sequence, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import optax
import tensorflow as tf
import torch
import torch.nn as nn
from clu import metrics
from equinox.nn import State
from jaxtyping import Array, Float, PRNGKeyArray
from torch import Tensor
from tqdm import tqdm

import tensorflow_datasets as tfds


class BatchNorm(eqx.Module):
    state_index: eqx.nn.StateIndex

    gamma: Float[Array, "size"] | None
    beta: Float[Array, "size"] | None

    inference: bool
    axis_name: Hashable | Sequence[Hashable]

    size: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)

    def __init__(
        self,
        size: int,
        axis_name: Hashable | Sequence[Hashable],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        inference: bool = False,
    ):
        self.size = size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.inference = inference
        self.axis_name = axis_name

        self.gamma = jnp.ones(self.size) if self.affine else None
        self.beta = jnp.zeros(self.size) if self.affine else None

        self.state_index = eqx.nn.StateIndex((jnp.zeros(size), jnp.ones(size)))

    def __call__(
        self,
        x: Array,
        state: State,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Array, State]:
        if inference is None:
            inference = self.inference

        running_mean, running_var = state.get(self.state_index)

        batch_mean = jax.lax.pmean(x, axis_name=self.axis_name)
        batch_size = jax.lax.psum(1, axis_name=self.axis_name)

        if inference:
            x_normalized = (x - running_mean) / jnp.sqrt(running_var + self.eps)
        else:
            xmu = x - batch_mean
            sq = xmu**2
            batch_var = jax.lax.pmean(sq, axis_name=self.axis_name)
            std = jnp.sqrt(batch_var + self.eps)
            x_normalized = xmu / std

            correction_factor = batch_size / jnp.maximum(batch_size - 1, 1)
            running_mean = (
                1 - self.momentum
            ) * running_mean + self.momentum * batch_mean
            running_var = (1 - self.momentum) * running_var + self.momentum * (
                batch_var * correction_factor
            )

            state = state.set(self.state_index, (running_mean, running_var))

        out = x_normalized
        if self.affine and self.gamma is not None and self.beta is not None:
            out = self.gamma * x_normalized + self.beta

        return out, state


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d
    bn: BatchNorm

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        key: jt.PRNGKeyArray,
    ):
        _, subkey = jax.random.split(key)
        self.conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            use_bias=False,
            key=subkey,
        )

        self.bn = BatchNorm(out_channels, axis_name="batch")

    def __call__(
        self, x: jt.Float[jt.Array, "c_in h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, "c_out*e h/s w/s"], eqx.nn.State]:
        x = self.conv(x)
        x, state = self.bn(x, state)

        return x, state


class BasicBlock(eqx.Module):
    downsample: Downsample | None

    conv1: eqx.nn.Conv2d
    bn1: BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: BatchNorm

    expansion: int = eqx.field(static=True, default=1)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: Downsample | None,
        groups: int,
        base_width: int,
        dilation: int,
        key: jt.PRNGKeyArray,
    ):
        key, *subkeys = jax.random.split(key, 3)

        self.conv1 = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            use_bias=False,
            key=subkeys[0],
        )
        self.bn1 = BatchNorm(size=out_channels, axis_name="batch")

        self.conv2 = eqx.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=subkeys[1],
        )
        self.bn2 = BatchNorm(size=out_channels, axis_name="batch")

        self.downsample = downsample

    def __call__(self, x: jt.Float[jt.Array, "c h w"], state: eqx.nn.State):
        i = x

        x = self.conv1(x)
        x, state = self.bn1(x, state)

        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state)

        if self.downsample:
            i, state = self.downsample(i, state)

        x += i
        x = jax.nn.relu(x)

        return x, state


class Bottleneck(eqx.Module):
    downsample: Downsample | None

    conv1: eqx.nn.Conv2d
    bn1: BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: BatchNorm

    conv3: eqx.nn.Conv2d
    bn3: BatchNorm

    expansion: int = eqx.field(static=True, default=4)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: Downsample | None,
        groups: int,
        base_width: int,
        dilation: int,
        key: jt.PRNGKeyArray,
    ):
        _, *subkeys = jax.random.split(key, 4)

        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = eqx.nn.Conv2d(
            in_channels, width, kernel_size=1, use_bias=False, key=subkeys[0]
        )
        self.bn1 = BatchNorm(width, axis_name="batch")

        self.conv2 = eqx.nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            dilation=dilation,
            padding=dilation,
            use_bias=False,
            key=subkeys[1],
        )

        self.bn2 = BatchNorm(width, axis_name="batch")

        self.conv3 = eqx.nn.Conv2d(
            width,
            out_channels * self.expansion,
            kernel_size=1,
            key=subkeys[2],
            use_bias=False,
        )

        self.bn3 = BatchNorm(out_channels * self.expansion, axis_name="batch")

        self.downsample = downsample

    def __call__(
        self, x: jt.Float[jt.Array, "c_in h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, "c_out*e h/s w/s"], eqx.nn.State]:
        i = x

        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)

        x = self.conv3(x)
        x, state = self.bn3(x, state)

        if self.downsample:
            i, state = self.downsample(i, state)

        x += i
        x = jax.nn.relu(x)
        return x, state


class ResNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn: BatchNorm
    mp: eqx.nn.MaxPool2d

    layer1: list[BasicBlock | Bottleneck]
    layer2: list[BasicBlock | Bottleneck]
    layer3: list[BasicBlock | Bottleneck]
    layer4: list[BasicBlock | Bottleneck]

    avg: eqx.nn.AdaptiveAvgPool2d
    fc: eqx.nn.Linear

    running_internal_channels: int = eqx.field(static=True, default=64)
    dilation: int = eqx.field(static=True, default=1)

    def __init__(
        self,
        block: Type[BasicBlock | Bottleneck],
        layers: list[int],
        n_classes: int,
        zero_init_residual: bool,
        groups: int,
        width_per_group: int,
        replace_stride_with_dilation: list[bool] | None,
        key: jt.PRNGKeyArray,
        input_channels: int = 3,
    ):
        key, *subkeys = jax.random.split(key, 10)

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"`replace_stride_with_dilation` should either be `None` or have a length of 3, got {replace_stride_with_dilation} instead."
            )

        self.conv1 = eqx.nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.running_internal_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=subkeys[0],
        )

        self.bn = BatchNorm(self.running_internal_channels, axis_name="batch")
        self.mp = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=1,
            dilate=False,
            groups=groups,
            base_width=width_per_group,
            key=subkeys[1],
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            groups=groups,
            base_width=width_per_group,
            key=subkeys[2],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            groups=groups,
            base_width=width_per_group,
            key=subkeys[3],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            groups=groups,
            base_width=width_per_group,
            key=subkeys[4],
        )

        self.avg = eqx.nn.AdaptiveAvgPool2d(target_shape=(1, 1))
        self.fc = eqx.nn.Linear(512 * block.expansion, n_classes, key=subkeys[-1])

        if zero_init_residual:
            # todo: init last bn layer with zero weights
            pass

    def _make_layer(
        self,
        block: Type[BasicBlock | Bottleneck],
        out_channels: int,
        blocks: int,
        stride: int,
        dilate: bool,
        groups: int,
        base_width: int,
        key: jt.PRNGKeyArray,
    ) -> list[BasicBlock | Bottleneck]:
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if (
            stride != 1
            or self.running_internal_channels != out_channels * block.expansion
        ):
            key, subkey = jax.random.split(key)
            downsample = Downsample(
                self.running_internal_channels,
                out_channels * block.expansion,
                stride,
                subkey,
            )
        layers = []

        key, subkey = jax.random.split(key)
        layers.append(
            block(
                in_channels=self.running_internal_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                groups=groups,
                base_width=base_width,
                dilation=previous_dilation,
                key=subkey,
            )
        )

        self.running_internal_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            key, subkey = jax.random.split(key)
            layers.append(
                block(
                    in_channels=self.running_internal_channels,
                    out_channels=out_channels,
                    groups=groups,
                    base_width=base_width,
                    dilation=self.dilation,
                    stride=1,
                    downsample=None,
                    key=subkey,
                )
            )

        return layers

    def __call__(
        self, x: jt.Float[jt.Array, "c h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, " n_classes"], eqx.nn.State]:
        x = self.conv1(x)
        x, state = self.bn(x, state)
        x = jax.nn.relu(x)
        x = self.mp(x)

        for layer in self.layer1:
            x, state = layer(x, state)

        for layer in self.layer2:
            x, state = layer(x, state)

        for layer in self.layer3:
            x, state = layer(x, state)

        for layer in self.layer4:
            x, state = layer(x, state)

        x = self.avg(x)
        x = jnp.ravel(x)

        x = self.fc(x)

        return x, state


def resnet18(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        BasicBlock,
        [2, 2, 2, 2],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )

    # initializer = jax.nn.initializers.he_normal()
    # is_conv2d = lambda x: isinstance(x, eqx.nn.Conv2d)
    # get_weights = lambda m: [
    #     x.weight for x in jax.tree.leaves(m, is_leaf=is_conv2d) if is_conv2d(x)
    # ]
    # weights = get_weights(resnet)
    # new_weights = [
    #     initializer(subkey, weight.shape, jnp.float32)
    #     for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    # ]
    # resnet = eqx.tree_at(get_weights, resnet, new_weights)

    return resnet, state


def resnet34(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        BasicBlock,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnet50(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnet101(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnet152(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 8, 36, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnext50_32x4d(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=32,
        width_per_group=4,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnext101_32x8d(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=32,
        width_per_group=8,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnext101_64x4d(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=64,
        width_per_group=4,
        replace_stride_with_dilation=None,
        key=key,
    )


def wide_resnet50_2(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64 * 2,
        replace_stride_with_dilation=None,
        key=key,
    )


def wide_resnet101_2(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64 * 2,
        replace_stride_with_dilation=None,
        key=key,
    )


(train, test), info = tfds.load(
    "cifar10", split=["train", "test"], with_info=True, as_supervised=True
)


def preprocess(
    img: jt.Float[tf.Tensor, "h w c"], label: jt.Int[tf.Tensor, ""]
) -> tuple[jt.Float[tf.Tensor, "h w c"], jt.Int[tf.Tensor, "1 n_classes"]]:
    # Convert to float and normalize to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0  # pyright: ignore

    mean = tf.constant([0.4914, 0.4822, 0.4465])
    std = tf.constant([0.2470, 0.2435, 0.2616])
    img = (img - mean) / std  # pyright: ignore

    img = tf.transpose(img, perm=[2, 0, 1])

    label = tf.one_hot(label, depth=10)

    return img, label


# For training data, add data augmentation
def preprocess_train(
    img: jt.Float[tf.Tensor, "h w c"], label: jt.Int[tf.Tensor, ""]
) -> tuple[jt.Float[tf.Tensor, "h w c"], jt.Int[tf.Tensor, "1 n_classes"]]:
    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, [32, 32, 3])
    img = tf.image.random_flip_left_right(img)  # pyright: ignore

    return preprocess(img, label)


train_dataset = train.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
SHUFFLE_VAL = len(train_dataset) // 1000
BATCH_SIZE = 128
train_dataset = train_dataset.shuffle(SHUFFLE_VAL)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

train_dataset = tfds.as_numpy(train_dataset)
test_dataset = tfds.as_numpy(test_dataset)


def loss_fn(
    resnet: ResNet,
    x: jt.Float[jt.Array, "batch_size 3 32 32"],
    y: jt.Float[jt.Array, "batch_size 10"],
    state: eqx.nn.State,
) -> tuple[jt.Array, tuple[jt.Array, eqx.nn.State]]:
    logits, state = eqx.filter_vmap(
        resnet, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(x, state)
    loss = optax.softmax_cross_entropy(logits, y)
    return jnp.mean(loss), (logits, state)


@eqx.filter_jit
def step(
    resnet: jt.PyTree,
    state: eqx.nn.State,
    x: jt.Float[jt.Array, "batch_size 3 32 32"],
    y: jt.Float[jt.Array, "batch_size 10"],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    (loss_value, (logits, state)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(resnet, x, y, state)
    updates, opt_state = optimizer.update(grads, opt_state, resnet)
    resnet = eqx.apply_updates(resnet, updates)
    return resnet, state, opt_state, loss_value, logits


class TrainMetrics(eqx.Module, metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy


def eval(
    resnet: ResNet, state: eqx.nn.State, test_dataset, key: jt.PRNGKeyArray
) -> tuple[TrainMetrics, eqx.nn.State]:
    eval_metrics = TrainMetrics.empty()
    for x, y in test_dataset:
        y = jnp.array(y, dtype=jnp.int32)
        loss, (logits, state) = loss_fn(resnet, x, y, state)
        eval_metrics = eval_metrics.merge(
            TrainMetrics.single_from_model_output(
                logits=logits, labels=jnp.argmax(y, axis=1), loss=loss
            )
        )

    return eval_metrics, state


train_metrics = TrainMetrics.empty()

resnet, state = resnet18(key=jax.random.key(0), n_classes=10)

resnet, state = resnet18(key=jax.random.key(0), n_classes=10)

learning_rate = 0.001
weight_decay = 5e-4
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.add_decayed_weights(weight_decay),
    optax.sgd(learning_rate, momentum=0.9),
)

# Add learning rate scheduler
scheduler = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=2000,  # Adjust based on dataset size
    decay_rate=0.1,
    end_value=1e-6,
)
optimizer = optax.chain(
    optax.add_decayed_weights(weight_decay), optax.sgd(scheduler, momentum=0.9)
)

opt_state = optimizer.init(eqx.filter(resnet, eqx.is_array))

key = jax.random.key(99)
n_epochs = 100


for epoch in range(n_epochs):
    batch_count = len(train_dataset)

    pbar = tqdm(enumerate(train_dataset), total=batch_count, desc=f"Epoch {epoch}")
    for i, (x, y) in pbar:
        y = jnp.array(y, dtype=jnp.int32)
        resnet, state, opt_state, loss, logits = step(
            resnet, state, x, y, optimizer, opt_state
        )
        train_metrics = train_metrics.merge(
            TrainMetrics.single_from_model_output(
                logits=logits, labels=jnp.argmax(y, axis=1), loss=loss
            )
        )

        vals = train_metrics.compute()
        pbar.set_postfix(
            {"loss": f"{vals['loss']:.4f}", "acc": f"{vals['accuracy']:.4f}"}
        )
    key, subkey = jax.random.split(key)
    eval_metrics, state = eval(resnet, state, test_dataset, subkey)
    evals = eval_metrics.compute()
    print(
        f"Epoch {epoch}: "
        f"test_loss={evals['loss']:.4f}, "
        f"test_acc={evals['accuracy']:.4f}"
    )
