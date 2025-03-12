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
import treescope as ts
from clu import metrics
from equinox.nn import State
from torch import Tensor
from tqdm import tqdm

import tensorflow_datasets as tfds


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

        self.bn = eqx.nn.BatchNorm(out_channels, axis_name="batch", momentum=0.1)

    def __call__(
        self, x: jt.Float[jt.Array, "c_in h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, "c_out*e h/s w/s"], eqx.nn.State]:
        x = self.conv(x)
        x, state = self.bn(x, state)

        return x, state


class JAXBasicBlock(eqx.Module):
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
        self.bn1 = eqx.nn.BatchNorm(
            input_size=out_channels, axis_name="batch", momentum=0.1
        )

        self.conv2 = eqx.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=subkeys[1],
        )
        self.bn2 = eqx.nn.BatchNorm(
            input_size=out_channels, axis_name="batch", momentum=0.1
        )

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


class TORCHBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        print("INITIALISING BASIC BLOCK WITH THESE PARAMETERS:")
        print("inplanes:", inplanes)
        print("planes:", planes)
        print("stride:", stride)
        print("downsample:", downsample)
        print("groups:", groups)
        print("base_width:", base_width)
        print("dilation:", dilation)
        print("norm_layer:", norm_layer)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        print(self.__class__.__name__, f"I: {x.shape}")
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        print(self.__class__.__name__, f"O: {out.shape}")
        return out


inplanes = 64
planes = 64
stride = 1
downsample = None
groups = 1
base_width = 64
dilation = 1

t_bb = TORCHBasicBlock(
    inplanes,
    planes,
    stride,
    downsample,
    groups,
    base_width,
    dilation,
)
print(t_bb.conv1.weight.shape)
print(t_bb.conv2.weight.shape)
print(t_bb.bn1.weight.shape)
print(t_bb.bn2.weight.shape)
j_bb, state = eqx.nn.make_with_state(JAXBasicBlock)(
    inplanes,
    planes,
    stride,
    downsample,
    groups,
    base_width,
    dilation,
    key=jax.random.key(0),
)

new_conv1_weight = np.array(
    np.random.uniform(size=j_bb.conv1.weight.shape), dtype=np.float32
)
print(new_conv1_weight.shape, t_bb.conv1.weight.shape)
assert new_conv1_weight.shape == t_bb.conv1.weight.shape
where = lambda l: l.conv1.weight
j_bb = eqx.tree_at(where, j_bb, new_conv1_weight)
t_bb.conv1.weight.data = torch.from_numpy(new_conv1_weight)
assert np.allclose(np.array(j_bb.conv1.weight), t_bb.conv1.weight.data.numpy())

new_bn1_weight = np.array(
    np.random.uniform(size=j_bb.bn1.weight.shape), dtype=np.float32
)
print(new_bn1_weight.shape, t_bb.bn1.weight.shape)
assert new_bn1_weight.shape == t_bb.bn1.weight.shape
where = lambda l: l.bn1.weight
j_bb = eqx.tree_at(where, j_bb, new_bn1_weight)
t_bb.bn1.weight.data = torch.from_numpy(new_bn1_weight)
assert np.allclose(np.array(j_bb.bn1.weight), t_bb.bn1.weight.data.numpy())

new_conv2_weight = np.array(
    np.random.uniform(size=j_bb.conv2.weight.shape), dtype=np.float32
)
print(new_conv2_weight.shape, t_bb.conv2.weight.shape)
assert new_conv2_weight.shape == t_bb.conv2.weight.shape
where = lambda l: l.conv2.weight
j_bb = eqx.tree_at(where, j_bb, new_conv2_weight)
t_bb.conv2.weight.data = torch.from_numpy(new_conv2_weight)
assert np.allclose(np.array(j_bb.conv2.weight), t_bb.conv2.weight.data.numpy())

new_bn2_weight = np.array(
    np.random.uniform(size=j_bb.bn2.weight.shape), dtype=np.float32
)
print(new_bn2_weight.shape, t_bb.bn2.weight.shape)
assert new_bn2_weight.shape == t_bb.bn2.weight.shape
where = lambda l: l.bn2.weight
j_bb = eqx.tree_at(where, j_bb, new_bn2_weight)
t_bb.bn2.weight.data = torch.from_numpy(new_bn2_weight)
assert np.allclose(np.array(j_bb.bn2.weight), t_bb.bn2.weight.data.numpy())


test_array = np.ones(shape=(4, 64, 8, 8), dtype=np.float32)
o_t = t_bb(torch.from_numpy(test_array))
o_j, _ = eqx.filter_vmap(
    j_bb, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(jnp.array(test_array), state)
assert np.allclose(o_j, o_t.detach().numpy(), atol=1e-4)


# Test with new parameters
inplanes = 64
planes = 128
stride = 2
groups = 1
base_width = 64
dilation = 1

# Create the PyTorch downsample module
t_downsample = nn.Sequential(conv1x1(inplanes, planes, stride), nn.BatchNorm2d(planes))

# Create the JAX downsample module
j_downsample = Downsample(inplanes, planes, stride, key=jax.random.key(1))

# Create the PyTorch and JAX BasicBlocks
t_bb2 = TORCHBasicBlock(
    inplanes,
    planes,
    stride,
    t_downsample,
    groups,
    base_width,
    dilation,
)

j_bb2, state2 = eqx.nn.make_with_state(JAXBasicBlock)(
    inplanes,
    planes,
    stride,
    j_downsample,
    groups,
    base_width,
    dilation,
    key=jax.random.key(2),
)

# Transfer weights from PyTorch downsample to JAX
# Downsample conv weight
new_ds_conv_weight = np.array(t_downsample[0].weight.data.numpy(), dtype=np.float32)
where = lambda l: l.downsample.conv.weight
j_bb2 = eqx.tree_at(where, j_bb2, new_ds_conv_weight)
assert np.allclose(
    np.array(j_bb2.downsample.conv.weight), t_downsample[0].weight.data.numpy()
)

# Downsample batchnorm weight
new_ds_bn_weight = np.array(t_downsample[1].weight.data.numpy(), dtype=np.float32)
where = lambda l: l.downsample.bn.weight
j_bb2 = eqx.tree_at(where, j_bb2, new_ds_bn_weight)
assert np.allclose(
    np.array(j_bb2.downsample.bn.weight), t_downsample[1].weight.data.numpy()
)

# Transfer weights for conv1, bn1, conv2, bn2 as before
new_conv1_weight = np.array(t_bb2.conv1.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.conv1.weight
j_bb2 = eqx.tree_at(where, j_bb2, new_conv1_weight)
assert np.allclose(np.array(j_bb2.conv1.weight), t_bb2.conv1.weight.data.numpy())

new_bn1_weight = np.array(t_bb2.bn1.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.bn1.weight
j_bb2 = eqx.tree_at(where, j_bb2, new_bn1_weight)
assert np.allclose(np.array(j_bb2.bn1.weight), t_bb2.bn1.weight.data.numpy())

new_conv2_weight = np.array(t_bb2.conv2.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.conv2.weight
j_bb2 = eqx.tree_at(where, j_bb2, new_conv2_weight)
assert np.allclose(np.array(j_bb2.conv2.weight), t_bb2.conv2.weight.data.numpy())

new_bn2_weight = np.array(t_bb2.bn2.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.bn2.weight
j_bb2 = eqx.tree_at(where, j_bb2, new_bn2_weight)
assert np.allclose(np.array(j_bb2.bn2.weight), t_bb2.bn2.weight.data.numpy())

# Test with input data
test_array2 = np.ones(shape=(4, 64, 8, 8), dtype=np.float32)
o_t2 = t_bb2(torch.from_numpy(test_array2))
o_j2, _ = eqx.filter_vmap(
    j_bb2, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)(jnp.array(test_array2), state2)
assert np.allclose(o_j2, o_t2.detach().numpy(), atol=1e-4)
print("All tests passed for configuration with downsample!")


class TORCHBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        print("INITIALISING BOTTLENECK BLOCK WITH THESE PARAMETERS:")
        print("inplanes:", inplanes)
        print("planes:", planes)
        print("stride:", stride)
        print("downsample:", downsample)
        print("groups:", groups)
        print("base_width:", base_width)
        print("dilation:", dilation)
        print("norm_layer:", norm_layer)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        print(self.__class__.__name__, f"I: {x.shape}")
        intermediates = {}
        identity = x
        intermediates["input"] = x

        out = self.conv1(x)
        intermediates["conv1"] = out
        out = self.bn1(out)
        intermediates["bn1"] = out
        out = self.relu(out)
        intermediates["relu1"] = out

        out = self.conv2(out)
        intermediates["conv2"] = out
        out = self.bn2(out)
        intermediates["bn2"] = out
        out = self.relu(out)
        intermediates["relu2"] = out

        out = self.conv3(out)
        intermediates["conv3"] = out
        out = self.bn3(out)
        intermediates["bn3"] = out

        if self.downsample is not None:
            identity = self.downsample(x)
            intermediates["downsample"] = identity

        out += identity
        intermediates["skip_connection"] = out
        out = self.relu(out)
        intermediates["final"] = out

        print(self.__class__.__name__, f"O: {out.shape}")
        return out, intermediates


class JAXBottleneck(eqx.Module):
    downsample: Downsample | None

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm

    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm

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
        self.bn1 = eqx.nn.BatchNorm(width, axis_name="batch", momentum=0.1)

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

        self.bn2 = eqx.nn.BatchNorm(width, axis_name="batch", momentum=0.1)

        self.conv3 = eqx.nn.Conv2d(
            width,
            out_channels * self.expansion,
            kernel_size=1,
            key=subkeys[2],
            use_bias=False,
        )

        self.bn3 = eqx.nn.BatchNorm(
            out_channels * self.expansion, axis_name="batch", momentum=0.1
        )

        self.downsample = downsample

    def __call__(
        self, x: jt.Float[jt.Array, "c_in h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, "c_out*e h/s w/s"], eqx.nn.State, dict]:
        intermediates = {}
        i = x
        intermediates["input"] = x

        x = self.conv1(x)
        intermediates["conv1"] = x
        x, state = self.bn1(x, state)
        intermediates["bn1"] = x
        x = jax.nn.relu(x)
        intermediates["relu1"] = x

        x = self.conv2(x)
        intermediates["conv2"] = x
        x, state = self.bn2(x, state)
        intermediates["bn2"] = x
        x = jax.nn.relu(x)
        intermediates["relu2"] = x

        x = self.conv3(x)
        intermediates["conv3"] = x
        x, state = self.bn3(x, state)
        intermediates["bn3"] = x

        if self.downsample:
            i, state = self.downsample(i, state)
            intermediates["downsample"] = i

        x += i
        intermediates["skip_connection"] = x
        x = jax.nn.relu(x)
        intermediates["final"] = x

        return x, state, intermediates


# Test Bottleneck layer
inplanes = 64
planes = 64
stride = 1
groups = 1
base_width = 64
dilation = 1
expansion = 4

# Create the PyTorch downsample module for Bottleneck
t_downsample_bn = nn.Sequential(
    conv1x1(inplanes, planes * 4, stride),  # Note the expansion factor of 4
    nn.BatchNorm2d(planes * 4),
)

# Create the JAX downsample module
j_downsample_bn = Downsample(
    inplanes,
    planes * 4,  # Note the expansion factor of 4
    stride,
    key=jax.random.key(3),
)

# Create the PyTorch and JAX Bottleneck blocks
t_bottleneck = TORCHBottleneck(
    inplanes,
    planes,
    stride,
    t_downsample_bn,
    groups,
    base_width,
    dilation,
)

j_bottleneck, bn_state = eqx.nn.make_with_state(JAXBottleneck)(
    inplanes,
    planes,
    stride,
    j_downsample_bn,
    groups,
    base_width,
    dilation,
    key=jax.random.key(4),
)

print("The state is: ", bn_state)
print(bn_state.get(j_bottleneck.bn1.state_index))
bn_state = bn_state.set(
    j_bottleneck.bn1.state_index,
    (
        jnp.ones_like(bn_state.get(j_bottleneck.bn1.state_index)[0]),
        jnp.zeros_like(bn_state.get(j_bottleneck.bn1.state_index)[1]),
    ),
)
print(bn_state.get(j_bottleneck.bn1.state_index))
print(t_bottleneck.bn1.running_var, t_bottleneck.bn1.running_mean)

print(j_bottleneck.bn1.bias)
print(t_bottleneck.bn1.bias)

# Transfer weights from PyTorch downsample to JAX
# Downsample conv weight
new_ds_conv_weight = np.array(t_downsample_bn[0].weight.data.numpy(), dtype=np.float32)
where = lambda l: l.downsample.conv.weight
j_bottleneck = eqx.tree_at(where, j_bottleneck, new_ds_conv_weight)
assert np.allclose(
    np.array(j_bottleneck.downsample.conv.weight),
    t_downsample_bn[0].weight.data.numpy(),
)

# Downsample batchnorm weight
new_ds_bn_weight = np.array(t_downsample_bn[1].weight.data.numpy(), dtype=np.float32)
where = lambda l: l.downsample.bn.weight
j_bottleneck = eqx.tree_at(where, j_bottleneck, new_ds_bn_weight)
assert np.allclose(
    np.array(j_bottleneck.downsample.bn.weight), t_downsample_bn[1].weight.data.numpy()
)

# Transfer weights for conv1, bn1
new_conv1_weight = np.array(t_bottleneck.conv1.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.conv1.weight
j_bottleneck = eqx.tree_at(where, j_bottleneck, new_conv1_weight)
assert np.allclose(
    np.array(j_bottleneck.conv1.weight), t_bottleneck.conv1.weight.data.numpy()
)

new_bn1_weight = np.array(t_bottleneck.bn1.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.bn1.weight
j_bottleneck = eqx.tree_at(where, j_bottleneck, new_bn1_weight)
assert np.allclose(
    np.array(j_bottleneck.bn1.weight), t_bottleneck.bn1.weight.data.numpy()
)

# Transfer weights for conv2, bn2
new_conv2_weight = np.array(t_bottleneck.conv2.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.conv2.weight
j_bottleneck = eqx.tree_at(where, j_bottleneck, new_conv2_weight)
assert np.allclose(
    np.array(j_bottleneck.conv2.weight), t_bottleneck.conv2.weight.data.numpy()
)

new_bn2_weight = np.array(t_bottleneck.bn2.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.bn2.weight
j_bottleneck = eqx.tree_at(where, j_bottleneck, new_bn2_weight)
assert np.allclose(
    np.array(j_bottleneck.bn2.weight), t_bottleneck.bn2.weight.data.numpy()
)

# Transfer weights for conv3, bn3 (new for Bottleneck)
new_conv3_weight = np.array(t_bottleneck.conv3.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.conv3.weight
j_bottleneck = eqx.tree_at(where, j_bottleneck, new_conv3_weight)
assert np.allclose(
    np.array(j_bottleneck.conv3.weight), t_bottleneck.conv3.weight.data.numpy()
)

new_bn3_weight = np.array(t_bottleneck.bn3.weight.data.numpy(), dtype=np.float32)
where = lambda l: l.bn3.weight
j_bottleneck = eqx.tree_at(where, j_bottleneck, new_bn3_weight)
assert np.allclose(
    np.array(j_bottleneck.bn3.weight), t_bottleneck.bn3.weight.data.numpy()
)

# print(j_bottleneck.bn1.weight, j_bottleneck.bn1.bias)
# print(t_bottleneck.bn1.weight, t_bottleneck.bn1.bias)

# print(j_bottleneck.bn2.weight, j_bottleneck.bn2.bias)
# print(t_bottleneck.bn2.weight, t_bottleneck.bn2.bias)

# print(j_bottleneck.bn3.weight, j_bottleneck.bn3.bias)
# print(t_bottleneck.bn3.weight, t_bottleneck.bn3.bias)

# Test with input data
test_array_bn = np.ones(shape=(4, 64, 8, 8), dtype=np.float32)
o_t_bn, t_intermediates = t_bottleneck(torch.from_numpy(test_array_bn))
o_j_bn, bn_state, j_intermediates = eqx.filter_vmap(
    j_bottleneck, in_axes=(0, None), out_axes=(0, None, 0), axis_name="batch"
)(jnp.array(test_array_bn), bn_state)
print(type(j_intermediates), j_intermediates.keys())
print(type(t_intermediates), t_intermediates.keys())

keys = [
    "input",
    "conv1",
    "bn1",
    "relu1",
    "conv2",
    "bn2",
    "relu2",
    "conv3",
    "bn3",
    "downsample",
    "skip_connection",
    "final",
]

for k in keys:
    j, t = j_intermediates[k], t_intermediates[k]
    print(j.shape, t.shape)
    assert j.shape == t.shape

    atols_to_check = [
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
    ]

    for a in atols_to_check:
        res = np.allclose(np.array(j), t.detach().numpy())
        print(f"Checking key {k} and atol {a}. Got {res}")
        if res:
            break

assert np.allclose(o_j_bn, o_t_bn.detach().numpy(), atol=1e-2)
print("All tests passed for Bottleneck with downsample!")
