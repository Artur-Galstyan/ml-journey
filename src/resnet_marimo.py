import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jaxtyping as jt
    import numpy as np
    import optax
    import tensorflow as tf
    from clu import metrics
    from tqdm import tqdm

    import tensorflow_datasets as tfds
    return eqx, jax, jnp, jt, metrics, np, optax, tf, tfds, tqdm


@app.cell
def _(tfds):
    (train, test), info = tfds.load(
        "cifar10", split=["train", "test"], with_info=True, as_supervised=True
    )

    # print(info, len(train), len(test))
    return info, test, train


@app.cell
def _(train):
    image, label = list(train.take(1))[0]
    print(image.shape)
    return image, label


@app.cell
def _(image, info, label, mo):
    mo.image(image, width=320, height=320, caption=info.features['label'].int2str(label))
    return


@app.cell
def _(eqx):
    class Block(eqx.Module):
        expansion: int = eqx.field(static=True)


    class BasicBlock(Block):
        pass
    return BasicBlock, Block


if __name__ == "__main__":
    app.run()
