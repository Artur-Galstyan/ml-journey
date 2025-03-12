import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
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
    import treescope as ts
    return (
        Any,
        Callable,
        ClassVar,
        State,
        Tensor,
        Type,
        eqx,
        jax,
        jnp,
        jt,
        metrics,
        nn,
        np,
        optax,
        tf,
        tfds,
        torch,
        tqdm,
        ts,
    )


@app.cell(hide_code=True)
def _(nn):
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
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )
    return conv1x1, conv3x3


@app.cell
def _(eqx, jax, jt):
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
    return Downsample, JAXBasicBlock


@app.cell
def _(Callable, Tensor, conv3x3, nn):
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
                raise ValueError(
                    "BasicBlock only supports groups=1 and base_width=64"
                )
            if dilation > 1:
                raise NotImplementedError(
                    "Dilation > 1 not supported in BasicBlock"
                )
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
    return (TORCHBasicBlock,)


@app.cell
def _(JAXBasicBlock, TORCHBasicBlock, eqx, jax):
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
    return (
        base_width,
        dilation,
        downsample,
        groups,
        inplanes,
        j_bb,
        planes,
        state,
        stride,
        t_bb,
    )


@app.cell
def _(eqx, j_bb, np, t_bb, torch):
    new_conv1_weight = np.random.uniform(size=j_bb.conv1.weight.shape)
    print(new_conv1_weight.shape, t_bb.conv1.weight.shape)
    assert new_conv1_weight.shape == t_bb.conv1.weight.shape
    where = lambda l: l.conv1.weight
    j_bb_new = eqx.tree_at(where, j_bb, new_conv1_weight)
    t_bb.conv1.weight.data = torch.from_numpy(new_conv1_weight)
    assert np.allclose(np.array(j_bb_new.conv1.weight), t_bb.conv1.weight.data.numpy())

    new_bn1_weight = np.random.uniform(size=j_bb_new.bn1.weight.shape)
    print(new_bn1_weight.shape, t_bb.bn1.weight.shape)
    assert new_bn1_weight.shape == t_bb.bn1.weight.shape
    where = lambda l: l.bn1.weight
    j_bb_new = eqx.tree_at(where, j_bb_new, new_bn1_weight)
    t_bb.bn1.weight.data = torch.from_numpy(new_bn1_weight)
    assert np.allclose(np.array(j_bb_new.bn1.weight), t_bb.bn1.weight.data.numpy())

    new_conv2_weight = np.random.uniform(size=j_bb_new.conv2.weight.shape)
    print(new_conv2_weight.shape, t_bb.conv2.weight.shape)
    assert new_conv2_weight.shape == t_bb.conv2.weight.shape
    where = lambda l: l.conv2.weight
    j_bb_new = eqx.tree_at(where, j_bb_new, new_conv2_weight)
    t_bb.conv2.weight.data = torch.from_numpy(new_conv2_weight)
    assert np.allclose(np.array(j_bb_new.conv2.weight), t_bb.conv2.weight.data.numpy())

    new_bn2_weight = np.random.uniform(size=j_bb_new.bn2.weight.shape)
    print(new_bn2_weight.shape, t_bb.bn2.weight.shape)
    assert new_bn2_weight.shape == t_bb.bn2.weight.shape
    where = lambda l: l.bn2.weight
    j_bb_new = eqx.tree_at(where, j_bb_new, new_bn2_weight)
    t_bb.bn2.weight.data = torch.from_numpy(new_bn2_weight)
    assert np.allclose(np.array(j_bb_new.bn2.weight), t_bb.bn2.weight.data.numpy())
    return (
        j_bb_new,
        new_bn1_weight,
        new_bn2_weight,
        new_conv1_weight,
        new_conv2_weight,
        where,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
