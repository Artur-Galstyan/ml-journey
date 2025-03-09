from collections.abc import Callable
from typing import Any, ClassVar, Type

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
from torch import Tensor
from tqdm import tqdm

import tensorflow_datasets as tfds


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm

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

        self.bn = eqx.nn.BatchNorm(out_channels, axis_name="batch")

    def __call__(
        self, x: jt.Float[jt.Array, "c_in h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, "c_out*e h/s w/s"], eqx.nn.State]:
        x = self.conv(x)
        x, state = self.bn(x, state)

        return x, state


class BasicBlock(eqx.Module):
    expansion: int = eqx.field(static=True, default=1)
    downsample: Downsample | None

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

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
        self.bn1 = eqx.nn.BatchNorm(input_size=out_channels, axis_name="batch")

        self.conv2 = eqx.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=subkeys[1],
        )
        self.bn2 = eqx.nn.BatchNorm(input_size=out_channels, axis_name="batch")

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
    expansion: int = eqx.field(static=True, default=4)
    downsample: Downsample | None

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm

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
        self.bn1 = eqx.nn.BatchNorm(width, axis_name="batch")

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

        self.bn2 = eqx.nn.BatchNorm(width, axis_name="batch")

        self.conv3 = eqx.nn.Conv2d(
            width,
            out_channels * self.expansion,
            kernel_size=1,
            key=subkeys[2],
            use_bias=False,
        )

        self.bn3 = eqx.nn.BatchNorm(out_channels * self.expansion, axis_name="batch")

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
    running_internal_channels: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    bn: eqx.nn.BatchNorm
    mp: eqx.nn.MaxPool2d

    layer1: list[BasicBlock | Bottleneck]
    layer2: list[BasicBlock | Bottleneck]
    layer3: list[BasicBlock | Bottleneck]
    layer4: list[BasicBlock | Bottleneck]

    avg: eqx.nn.AdaptiveAvgPool2d
    fc: eqx.nn.Linear

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
        self.running_internal_channels = 64
        self.dilation = 1

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

        self.bn = eqx.nn.BatchNorm(self.running_internal_channels, axis_name="batch")
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

        # todo: init weights using kaiming normal

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
    return eqx.nn.make_with_state(ResNet)(
        BasicBlock,
        [2, 2, 2, 2],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


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


image = jnp.ones(shape=(1, 3, 32, 32))

r, state = resnet18(key=jax.random.key(0), n_classes=10)
o, state = eqx.filter_vmap(r, axis_name="batch", in_axes=(0, None))(image, state)
print(o.shape)

(train, test), info = tfds.load(
    "cifar10", split=["train", "test"], with_info=True, as_supervised=True
)


def preprocess(
    img: jt.Float[tf.Tensor, "h w c"], label: jt.Int[tf.Tensor, ""]
) -> tuple[jt.Float[tf.Tensor, "h w c"], jt.Int[tf.Tensor, ""]]:
    img = tf.divide(tf.cast(img, tf.float32), 255.0)
    img = tf.transpose(img, perm=[2, 0, 1])
    return img, label


train_dataset = train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
SHUFFLE_VAL = len(train_dataset) // 1000
BATCH_SIZE = 4
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
        y = y.reshape(-1)  # Make 1D for CLU
        y = jnp.array(y, dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        loss, (logits, state) = loss_fn(resnet, x, y.reshape(-1, 1), subkey)
        logits = jnp.concatenate([-logits, logits], axis=1)
        eval_metrics = eval_metrics.merge(
            TrainMetrics.single_from_model_output(logits=logits, labels=y, loss=loss)
        )

    return eval_metrics, state


train_metrics = TrainMetrics.empty()

resnet, state = resnet18(key=jax.random.key(0), n_classes=10)
learning_rate = 0.0001
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(resnet, eqx.is_array))

key = jax.random.key(99)
n_epochs = 10


for epoch in range(n_epochs):
    batch_count = len(train_dataset)

    pbar = tqdm(enumerate(train_dataset), total=batch_count, desc=f"Epoch {epoch}")
    for i, (x, y) in pbar:
        y = y.reshape(-1, 1)
        y = jnp.array(y, dtype=jnp.int32)
        resnet, state, opt_state, loss, logits = step(
            resnet, state, x, y, optimizer, opt_state
        )
        logits = jnp.concatenate([-logits, logits], axis=1)
        train_metrics = train_metrics.merge(
            TrainMetrics.single_from_model_output(
                logits=logits, labels=y.reshape(-1), loss=loss
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
