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
import torch.nn.functional as F
from clu import metrics
from equinox.nn import State
from jaxtyping import Array, Float, PRNGKeyArray
from torch import Tensor
from tqdm import tqdm

import tensorflow_datasets as tfds


class CustomBatchNorm(eqx.Module):
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

        input_shape = x.shape
        ndim = len(input_shape)

        if ndim == 1:
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
        else:
            spatial_axes = tuple(range(1, ndim))  # All dims except channel dim (0)

            if inference:
                x_normalized = (
                    x - running_mean.reshape((-1,) + (1,) * (ndim - 1))
                ) / jnp.sqrt(running_var.reshape((-1,) + (1,) * (ndim - 1)) + self.eps)
            else:
                spatial_mean = jnp.mean(x, axis=spatial_axes)

                batch_mean = jax.lax.pmean(spatial_mean, axis_name=self.axis_name)
                batch_size = jax.lax.psum(1, axis_name=self.axis_name)

                broadcast_shape = (-1,) + (1,) * (ndim - 1)
                batch_mean_broadcasted = batch_mean.reshape(broadcast_shape)

                xmu = x - batch_mean_broadcasted
                sq = xmu**2

                spatial_var = jnp.mean(sq, axis=spatial_axes)
                batch_var = jax.lax.pmean(spatial_var, axis_name=self.axis_name)

                batch_var_broadcasted = batch_var.reshape(broadcast_shape)
                std = jnp.sqrt(batch_var_broadcasted + self.eps)

                x_normalized = xmu / std

                spatial_size = 1
                for dim in spatial_axes:
                    spatial_size *= x.shape[dim]
                total_size = batch_size * spatial_size

                correction_factor = total_size / jnp.maximum(total_size - 1, 1)
                running_mean = (
                    1 - self.momentum
                ) * running_mean + self.momentum * batch_mean
                running_var = (1 - self.momentum) * running_var + self.momentum * (
                    batch_var * correction_factor
                )

                state = state.set(self.state_index, (running_mean, running_var))

        out = x_normalized
        if self.affine and self.gamma is not None and self.beta is not None:
            if ndim > 1:
                broadcast_shape = (-1,) + (1,) * (ndim - 1)
                gamma_broadcasted = self.gamma.reshape(broadcast_shape)
                beta_broadcasted = self.beta.reshape(broadcast_shape)
                out = gamma_broadcasted * x_normalized + beta_broadcasted
            else:
                out = self.gamma * x_normalized + self.beta

        return out, state


def _batch_norm():
    pass


def batch_norm(
    x,
    gamma,  # scaling factor
    beta,  # offset factor
    eps=1e-5,
    momentum=0.1,
    running_mean=None,
    running_var=None,
    training=True,
):
    N, D = x.shape
    if running_mean is None:
        running_mean = jnp.zeros(D, dtype=x.dtype)
    if running_var is None:
        running_var = jnp.ones(D, dtype=x.dtype)

    running_mean_copy = running_mean.copy()
    running_var_copy = running_var.copy()

    if training:
        batch_mean = jnp.mean(x, axis=0)
        xmu = x - batch_mean
        sq = xmu**2
        batch_var = jnp.mean(sq, axis=0)
        std = jnp.sqrt(batch_var + eps)
        x_normalized = xmu / std

        out = gamma * x_normalized + beta

        correction_factor = N / (N - 1)
        running_mean_copy = (1 - momentum) * running_mean_copy + momentum * batch_mean
        running_var_copy = (1 - momentum) * running_var_copy + momentum * (
            batch_var * correction_factor
        )

    else:
        x_normalized = (x - running_mean_copy) / jnp.sqrt(running_var_copy + eps)
        out = gamma * x_normalized + beta

    return out, running_mean_copy, running_var_copy


def test_my_batchnorm():
    feature_size = 4
    batch_size = 8
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize parameters
    gamma = np.ones(feature_size, dtype=np.float32)
    beta = np.zeros(feature_size, dtype=np.float32)
    eps = 1e-5
    momentum = 0.1
    running_mean_our = np.zeros(feature_size, dtype=np.float32)
    running_var_our = np.ones(feature_size, dtype=np.float32)

    my_batch_norm, state = eqx.nn.make_with_state(CustomBatchNorm)(
        size=feature_size, axis_name="batch", eps=eps, momentum=momentum
    )
    x_np = np.random.randn(batch_size, feature_size).astype(np.float32)
    x_jax = jnp.array(x_np)

    batch_norm(
        x_np,
        gamma,
        beta,
        eps,
        momentum,
        running_mean_our,
        running_var_our,
        training=True,
    )

    o, state = eqx.filter_vmap(
        my_batch_norm, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(x_jax, state)


def test_multiple_batches(num_batches=5, batch_size=64, feature_size=10):
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize parameters
    gamma = np.ones(feature_size, dtype=np.float32)
    beta = np.zeros(feature_size, dtype=np.float32)
    eps = 1e-5
    momentum = 0.1

    # Initialize running statistics
    running_mean_our = np.zeros(feature_size, dtype=np.float32)
    running_var_our = np.ones(feature_size, dtype=np.float32)

    # Convert to PyTorch tensors for PyTorch implementation
    gamma_torch = torch.from_numpy(gamma)
    beta_torch = torch.from_numpy(beta)
    running_mean_torch = torch.from_numpy(running_mean_our.copy())
    running_var_torch = torch.from_numpy(running_var_our.copy())

    # Create PyTorch BatchNorm layer
    bn_torch = nn.BatchNorm1d(feature_size, eps=eps, momentum=momentum)
    bn_torch.weight.data = gamma_torch
    bn_torch.bias.data = beta_torch
    bn_torch.running_mean = running_mean_torch
    bn_torch.running_var = running_var_torch
    bn_torch.train()  # Set to training mode

    # Initialize Equinox BatchNorm
    def setup_equinox_batchnorm():
        bn, bs = eqx.nn.make_with_state(eqx.nn.BatchNorm)(
            input_size=feature_size,
            axis_name="batch",
            momentum=1
            - momentum,  # Note: Equinox momentum is inverted compared to PyTorch
            eps=eps,
            channelwise_affine=True,
            inference=False,
        )

        # Set parameters
        bs = bs.set(
            bn.state_index,
            (
                jnp.array(running_mean_our.copy()),
                jnp.array(running_var_our.copy()),
            ),
        )
        bn = eqx.tree_at(lambda l: l.weight, bn, jnp.array(gamma))
        bn = eqx.tree_at(lambda l: l.bias, bn, jnp.array(beta))

        # Make sure first_time flag is True
        bs = bs.set(bn.first_time_index, jnp.array(True))

        return bn, bs

    # Initialize Custom BatchNorm
    def setup_custom_batchnorm():
        bn, bs = eqx.nn.make_with_state(CustomBatchNorm)(
            size=feature_size,
            axis_name="batch",
            eps=eps,
            momentum=momentum,
            affine=True,
            inference=False,
        )

        # Set parameters
        bn = eqx.tree_at(lambda b: b.gamma, bn, jnp.array(gamma))
        bn = eqx.tree_at(lambda b: b.beta, bn, jnp.array(beta))

        return bn, bs

    bn_eqx, bs_eqx = setup_equinox_batchnorm()
    bn_custom, bs_custom = setup_custom_batchnorm()

    print("=== Testing BatchNorm over multiple batches ===")
    print("-" * 70)
    print(
        "Batch | PyTorch Mean | Our Mean | Equinox Mean | Custom Mean | PyTorch Var | Our Var | Equinox Var | Custom Var"
    )
    print("-" * 120)

    # Process multiple batches
    for batch_idx in range(num_batches):
        # Generate new batch of data
        x_np = np.random.randn(batch_size, feature_size).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_jax = jnp.array(x_np)

        # Our implementation
        out_our, running_mean_our, running_var_our = batch_norm(
            x_np,
            gamma,
            beta,
            eps,
            momentum,
            running_mean_our,
            running_var_our,
            training=True,
        )

        # PyTorch implementation
        out_torch = bn_torch(x_torch)

        # Equinox implementation
        vmap_fn = eqx.filter_vmap(
            bn_eqx, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
        out_eqx, bs_eqx = vmap_fn(x_jax, bs_eqx)

        # Custom Equinox implementation
        vmap_fn_custom = eqx.filter_vmap(
            bn_custom, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
        out_custom, bs_custom = vmap_fn_custom(x_jax, bs_custom)

        # Get running stats from each implementation
        running_mean_torch_np = bn_torch.running_mean.detach().numpy()
        running_var_torch_np = bn_torch.running_var.detach().numpy()

        running_mean_eqx = bs_eqx.get(bn_eqx.state_index)[0]
        running_var_eqx = bs_eqx.get(bn_eqx.state_index)[1]

        running_mean_custom = bs_custom.get(bn_custom.state_index)[0]
        running_var_custom = bs_custom.get(bn_custom.state_index)[1]

        # Print mean values across features (for brevity)
        print(
            f"{batch_idx:5d} | {running_mean_torch_np.mean():11.6f} | {running_mean_our.mean():8.6f} | "
            f"{running_mean_eqx.mean():11.6f} | {running_mean_custom.mean():11.6f} | "
            f"{running_var_torch_np.mean():10.6f} | {running_var_our.mean():7.6f} | "
            f"{running_var_eqx.mean():10.6f} | {running_var_custom.mean():10.6f}"
        )

        # Compare outputs for this batch
        if batch_idx == 0:
            print("\nInitial batch output comparison:")
            print(
                f"PyTorch vs Our: {np.max(np.abs(out_our - out_torch.detach().numpy())):.8f}"
            )
            print(
                f"PyTorch vs Equinox: {np.max(np.abs(out_eqx - out_torch.detach().numpy())):.8f}"
            )
            print(
                f"PyTorch vs Custom: {np.max(np.abs(out_custom - out_torch.detach().numpy())):.8f}"
            )

        if batch_idx == num_batches - 1:
            print("\nFinal batch output comparison:")
            print(
                f"PyTorch vs Our: {np.max(np.abs(out_our - out_torch.detach().numpy())):.8f}"
            )
            print(
                f"PyTorch vs Equinox: {np.max(np.abs(out_eqx - out_torch.detach().numpy())):.8f}"
            )
            print(
                f"PyTorch vs Custom: {np.max(np.abs(out_custom - out_torch.detach().numpy())):.8f}"
            )

            print("\nFinal running stats comparison:")
            print(
                f"PyTorch vs Our - Mean: {np.max(np.abs(running_mean_our - running_mean_torch_np)):.8f}"
            )
            print(
                f"PyTorch vs Our - Var: {np.max(np.abs(running_var_our - running_var_torch_np)):.8f}"
            )
            print(
                f"PyTorch vs Equinox - Mean: {np.max(np.abs(running_mean_eqx - running_mean_torch_np)):.8f}"
            )
            print(
                f"PyTorch vs Equinox - Var: {np.max(np.abs(running_var_eqx - running_var_torch_np)):.8f}"
            )
            print(
                f"PyTorch vs Custom - Mean: {np.max(np.abs(running_mean_custom - running_mean_torch_np)):.8f}"
            )
            print(
                f"PyTorch vs Custom - Var: {np.max(np.abs(running_var_custom - running_var_torch_np)):.8f}"
            )

    # Test inference mode with final stats
    bn_torch.eval()

    x_test = np.random.randn(batch_size, feature_size).astype(np.float32)
    x_test_torch = torch.from_numpy(x_test)
    x_test_jax = jnp.array(x_test)

    # Our implementation in eval mode
    out_our_test, _, _ = batch_norm(
        x_test,
        gamma,
        beta,
        eps,
        momentum,
        running_mean_our,
        running_var_our,
        training=False,
    )

    # PyTorch in eval mode
    out_torch_test = bn_torch(x_test_torch)

    # Equinox in inference mode
    bn_eqx_eval = eqx.tree_at(lambda b: b.inference, bn_eqx, True)
    vmap_fn_eval = eqx.filter_vmap(
        bn_eqx_eval, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )
    out_eqx_test, _ = vmap_fn_eval(x_test_jax, bs_eqx)

    # Custom in inference mode
    bn_custom_eval = eqx.tree_at(lambda b: b.inference, bn_custom, True)
    vmap_fn_custom_eval = eqx.filter_vmap(
        bn_custom_eval, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )
    out_custom_test, _ = vmap_fn_custom_eval(x_test_jax, bs_custom)

    print("\nInference mode output comparison:")
    print(
        f"PyTorch vs Our: {np.max(np.abs(out_our_test - out_torch_test.detach().numpy())):.8f}"
    )
    print(
        f"PyTorch vs Equinox: {np.max(np.abs(out_eqx_test - out_torch_test.detach().numpy())):.8f}"
    )
    print(
        f"PyTorch vs Custom: {np.max(np.abs(out_custom_test - out_torch_test.detach().numpy())):.8f}"
    )


def test_multidim_batchnorm(num_batches=5, batch_size=32):
    # Test with high-dimensional data
    # Using a 5D tensor: batch, channels, d1, d2, d3
    feature_size = (4, 4, 4, 4, 4)  # 3D feature map
    channels = 3  # RGB channels

    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize parameters (one per channel)
    gamma = np.ones(channels, dtype=np.float32)
    beta = np.zeros(channels, dtype=np.float32)
    eps = 1e-5
    momentum = 0.1

    # Initialize running statistics
    running_mean_torch = np.zeros(channels, dtype=np.float32)
    running_var_torch = np.ones(channels, dtype=np.float32)

    # Convert to PyTorch tensors
    gamma_torch = torch.from_numpy(gamma)
    beta_torch = torch.from_numpy(beta)
    running_mean_torch_tensor = torch.from_numpy(running_mean_torch.copy())
    running_var_torch_tensor = torch.from_numpy(running_var_torch.copy())

    # Initialize Equinox BatchNorm
    def setup_equinox_batchnorm():
        bn, bs = eqx.nn.make_with_state(eqx.nn.BatchNorm)(
            input_size=channels,
            axis_name="batch",
            momentum=1 - momentum,  # Equinox momentum is inverted compared to PyTorch
            eps=eps,
            channelwise_affine=True,
            inference=False,
        )

        # Set parameters
        bs = bs.set(
            bn.state_index,
            (
                jnp.array(running_mean_torch.copy()),
                jnp.array(running_var_torch.copy()),
            ),
        )
        bn = eqx.tree_at(lambda l: l.weight, bn, jnp.array(gamma))
        bn = eqx.tree_at(lambda l: l.bias, bn, jnp.array(beta))

        # Make sure first_time flag is True
        bs = bs.set(bn.first_time_index, jnp.array(True))

        return bn, bs

    # Initialize Custom BatchNorm
    def setup_custom_batchnorm():
        bn, bs = eqx.nn.make_with_state(CustomBatchNorm)(
            size=channels,  # Just channels, not spatial dimensions
            axis_name="batch",
            eps=eps,
            momentum=momentum,
            affine=True,
            inference=False,
        )

        # Set parameters
        bn = eqx.tree_at(lambda b: b.gamma, bn, jnp.array(gamma))
        bn = eqx.tree_at(lambda b: b.beta, bn, jnp.array(beta))

        return bn, bs

    bn_eqx, bs_eqx = setup_equinox_batchnorm()
    bn_custom, bs_custom = setup_custom_batchnorm()

    print("=== Testing BatchNorm with Multi-dimensional inputs ===")
    print(
        f"Input shape: ({batch_size}, {channels}, {feature_size[0]}, {feature_size[1]}, {feature_size[2]})"
    )
    print("-" * 90)
    print(
        "Batch | PyTorch Mean | Equinox Mean | Custom Mean | PyTorch Var | Equinox Var | Custom Var"
    )
    print("-" * 90)

    training = True  # Start in training mode

    for batch_idx in range(num_batches):
        # Generate high-dimensional data: [batch, channels, d1, d2, d3]
        x_np = np.random.randn(batch_size, channels, *feature_size).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        x_jax = jnp.array(x_np)

        # PyTorch implementation using F.batch_norm directly
        # First, compute mean and var along the right dimensions (all except channels)
        if training:
            # Reshape to (batch, channels, -1) to compute statistics over all other dimensions
            x_reshaped = x_torch.reshape(batch_size, channels, -1)
            batch_mean = x_reshaped.mean(
                dim=[0, 2]
            )  # Mean over batch and all spatial dims
            batch_var = x_reshaped.var(dim=[0, 2], unbiased=True)  # Unbiased variance

            # Update running statistics
            running_mean_torch_tensor = (
                1 - momentum
            ) * running_mean_torch_tensor + momentum * batch_mean
            running_var_torch_tensor = (
                1 - momentum
            ) * running_var_torch_tensor + momentum * batch_var

            # Use F.batch_norm with the computed statistics
            out_torch = F.batch_norm(
                x_torch,
                running_mean=None,  # Don't use running stats in training
                running_var=None,
                weight=gamma_torch,
                bias=beta_torch,
                training=True,
                momentum=momentum,
                eps=eps,
            )
        else:
            # In eval mode, use running statistics
            out_torch = F.batch_norm(
                x_torch,
                running_mean=running_mean_torch_tensor,
                running_var=running_var_torch_tensor,
                weight=gamma_torch,
                bias=beta_torch,
                training=False,
                momentum=momentum,
                eps=eps,
            )

        # Equinox implementation
        vmap_fn_eqx = eqx.filter_vmap(
            lambda x, s: bn_eqx(x, s),
            axis_name="batch",
            in_axes=(0, None),
            out_axes=(0, None),
        )
        out_eqx, bs_eqx = vmap_fn_eqx(x_jax, bs_eqx)

        # Custom Equinox implementation
        vmap_fn_custom = eqx.filter_vmap(
            lambda x, s: bn_custom(x, s),
            axis_name="batch",
            in_axes=(0, None),
            out_axes=(0, None),
        )
        out_custom, bs_custom = vmap_fn_custom(x_jax, bs_custom)

        # Get running stats
        running_mean_torch_np = running_mean_torch_tensor.detach().numpy()
        running_var_torch_np = running_var_torch_tensor.detach().numpy()

        running_mean_eqx = bs_eqx.get(bn_eqx.state_index)[0]
        running_var_eqx = bs_eqx.get(bn_eqx.state_index)[1]

        running_mean_custom = bs_custom.get(bn_custom.state_index)[0]
        running_var_custom = bs_custom.get(bn_custom.state_index)[1]

        # Print mean values
        print(
            f"{batch_idx:5d} | {running_mean_torch_np.mean():11.6f} | "
            f"{running_mean_eqx.mean():11.6f} | {running_mean_custom.mean():11.6f} | "
            f"{running_var_torch_np.mean():10.6f} | {running_var_eqx.mean():10.6f} | "
            f"{running_var_custom.mean():10.6f}"
        )

        # Output comparison for first batch
        if batch_idx == 0:
            print("\nInitial batch output comparison:")
            print(
                f"PyTorch vs Equinox: {np.max(np.abs(out_eqx - out_torch.detach().numpy())):.8f}"
            )
            print(
                f"PyTorch vs Custom: {np.max(np.abs(out_custom - out_torch.detach().numpy())):.8f}"
            )
            print(f"Custom vs Equinox: {np.max(np.abs(out_custom - out_eqx)):.8f}")

        if batch_idx == num_batches - 1:
            print("\nFinal running stats comparison:")
            print(
                f"PyTorch vs Equinox - Mean: {np.max(np.abs(running_mean_eqx - running_mean_torch_np)):.8f}"
            )
            print(
                f"PyTorch vs Equinox - Var: {np.max(np.abs(running_var_eqx - running_var_torch_np)):.8f}"
            )
            print(
                f"PyTorch vs Custom - Mean: {np.max(np.abs(running_mean_custom - running_mean_torch_np)):.8f}"
            )
            print(
                f"PyTorch vs Custom - Var: {np.max(np.abs(running_var_custom - running_var_torch_np)):.8f}"
            )

    # Test inference mode
    training = False  # Switch to evaluation mode

    x_test = np.random.randn(batch_size, channels, *feature_size).astype(np.float32)
    x_test_torch = torch.from_numpy(x_test)
    x_test_jax = jnp.array(x_test)

    # PyTorch in eval mode using F.batch_norm
    out_torch_test = F.batch_norm(
        x_test_torch,
        running_mean=running_mean_torch_tensor,
        running_var=running_var_torch_tensor,
        weight=gamma_torch,
        bias=beta_torch,
        training=False,
        momentum=momentum,
        eps=eps,
    )

    # Equinox in inference mode
    bn_eqx_eval = eqx.tree_at(lambda b: b.inference, bn_eqx, True)
    vmap_fn_eqx_eval = eqx.filter_vmap(
        bn_eqx_eval, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )
    out_eqx_test, _ = vmap_fn_eqx_eval(x_test_jax, bs_eqx)

    # Custom in inference mode
    bn_custom_eval = eqx.tree_at(lambda b: b.inference, bn_custom, True)
    vmap_fn_custom_eval = eqx.filter_vmap(
        bn_custom_eval, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )
    out_custom_test, _ = vmap_fn_custom_eval(x_test_jax, bs_custom)

    print("\nInference mode output comparison:")
    print(
        f"PyTorch vs Equinox: {np.max(np.abs(out_eqx_test - out_torch_test.detach().numpy())):.8f}"
    )
    print(
        f"PyTorch vs Custom: {np.max(np.abs(out_custom_test - out_torch_test.detach().numpy())):.8f}"
    )
    print(f"Custom vs Equinox: {np.max(np.abs(out_custom_test - out_eqx_test)):.8f}")


if __name__ == "__main__":
    # test_multiple_batches(num_batches=50)
    # test_my_batchnorm()
    test_multidim_batchnorm(num_batches=128)
