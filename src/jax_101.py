import time
from collections.abc import Callable
from functools import partial

import equinox as eqx
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import make_jaxpr
from jaxtyping import Array, Float, Int, PRNGKeyArray

# What is JAX
## Origin: autograd (for numpy), then grew into JAX
## (relatively new) ML library by Google;
## handles AD, GPUs, TPUs, CPUs
## functional programming paradigm

# JAX vs. Numpy ✅
# Indexing ✅
# RNG ✅
# jax.vmap ✅
# jax.grad (and jax.value_and_grad) ✅
# jax.jit ✅
# jax.lax.scan ✅
# while/for loops ✅


# Equinox 101


class Model(eqx.Module):
    lin: eqx.nn.Linear
    relu: Callable

    def __init__(self):
        self.lin = eqx.nn.Linear(in_features=3, out_features=3, key=jax.random.key(0))
        self.relu = jax.nn.relu

    @eqx.filter_jit
    def __call__(self, x: Array) -> Array:
        return self.relu(self.lin(x))


m = Model()
o = m(jnp.array([1, 2, 3]))
print(o)


# Flax 101

# Optax 101
