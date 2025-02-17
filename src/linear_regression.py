import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
from beartype import beartype as typechecker
from jaxtyping import Array, Float, PyTree, jaxtyped
from tqdm import tqdm

# 1. This is not a Python tutorial
# 2. This is not an in-depth PyTorch tutorial -> will come later :)
# 3. This is not an in-depth JAX tutorial -> will come later
# 4. This is more of an overview of PyTorch & JAX


def main():
    def target_function(x, noise=True):
        return 2 * x + 5 + np.random.normal(0, 1, x.shape) if noise else 2 * x + 5

    learning_rate = 0.01
    n_epochs = 50

    train_x = np.linspace(0, 10, 128)
    train_y = target_function(train_x)

    test_x = np.linspace(0, 10, 32)
    test_y = target_function(test_x)

    # plt.scatter(train_x, train_y)
    # plt.plot(train_x, target_function(train_x, noise=False), color="red")
    # plt.show()

    # JAX

    @jaxtyped(typechecker=typechecker)
    class LinearModelJAX(eqx.Module):
        slope: Float[Array, ""]
        intercept: Float[Array, ""]

        def __init__(self):
            self.slope = jnp.array(0.0)
            self.intercept = jnp.array(0.0)

        def __call__(self, x: Float[Array, ""]) -> Float[Array, ""]:
            return self.slope * x + self.intercept

    linear_model_jax = LinearModelJAX()

    def loss_fn_jax(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    model_params = eqx.filter(linear_model_jax, eqx.is_array)
    loss, grads = eqx.filter_value_and_grad(loss_fn_jax)(
        model_params, jnp.array(train_x[0]), jnp.array(train_y[0])
    )
    print(loss)
    print(model_params.slope, model_params.intercept)
    print(grads.slope, grads.intercept)
    update = jax.tree.map(lambda p, g: p - learning_rate * g, model_params, grads)
    print(update.slope, update.intercept)

    @eqx.filter_jit
    def step_fn(
        model: PyTree,
        x: Array,
        y: Array,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
    ) -> tuple[PyTree, optax.OptState, Array]:
        loss, grad = eqx.filter_value_and_grad(loss_fn_jax)(model, x, y)
        updates, opt_state = optimizer.update(grad, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def eval(model, test_x, test_y):
        preds = eqx.filter_vmap(model)(test_x)
        diff = jnp.mean(preds - test_y) ** 2
        return diff

    jax_optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = jax_optimizer.init(eqx.filter(linear_model_jax, eqx.is_inexact_array))

    for epoch in range(n_epochs):
        train_loss = 0
        for x, y in zip(train_x, train_y):
            linear_model_jax, opt_state, loss = step_fn(
                linear_model_jax, jnp.array(x), jnp.array(y), opt_state, jax_optimizer
            )
            train_loss += loss

        print(
            f"Loss for epoch {epoch} = {eval(linear_model_jax, test_x, test_y)}, train_loss = {train_loss / len(train_x)}"
        )

    # PyTorch
    class LinearModelPyTorch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slope = torch.nn.Parameter(torch.tensor(0.0))
            self.intercept = torch.nn.Parameter(torch.tensor(0.0))

        @jaxtyped(typechecker=typechecker)
        def forward(self, x: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
            return self.slope * x + self.intercept

    linear_model_torch = LinearModelPyTorch()

    @torch.no_grad
    def eval_torch(
        model: LinearModelPyTorch, test_x: torch.Tensor, test_y: torch.Tensor
    ):
        preds = torch.vmap(model.forward)(test_x)
        diffs = torch.mean((preds - test_y) ** 2)
        return diffs

    def loss_fn_torch(model, x, y_expected):
        return torch.mean((model.forward(x) - y_expected) ** 2)

    torch_optimizer = torch.optim.Adam(
        linear_model_torch.parameters(), lr=learning_rate
    )
    linear_model_torch.train()  # training mode!
    for epoch in range(n_epochs):
        train_loss = 0
        for x, y in zip(train_x, train_y):
            torch_optimizer.zero_grad()  # always reset the gradients!
            loss = loss_fn_torch(linear_model_torch, torch.tensor(x), y)

            loss.backward()  # this computes the gradients
            torch_optimizer.step()  # this steps the optimizer and updates the model parameters
            train_loss += loss

        train_loss /= len(train_x)
        eval_loss = eval_torch(
            linear_model_torch, torch.tensor(test_x), torch.tensor(test_y)
        )

        print(f"Loss for epoch {epoch} = {eval_loss}, train_loss = {train_loss}")

    print(f"{linear_model_jax.slope=}, {linear_model_jax.intercept=}")
    print(f"{linear_model_torch.slope=}, {linear_model_torch.intercept=}")
