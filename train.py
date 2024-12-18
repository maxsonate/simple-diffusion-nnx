"""Training file for simple diffusion model."""
from typing import Tuple
from flax import nnx
import jax
import jax.numpy as jnp
from jax import random
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import numpy as np
from modules import UNet


# Prevent TFDS from using GPU
tf.config.experimental.set_visible_devices([], 'GPU')

# Defining some hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_STEPS_PER_EPOCH = 60000//BATCH_SIZE # MNIST has 60,000 training samples

##################### CLEAN THIS SECTION #####################:
timesteps = 200
beta = jnp.linspace(0.0001, 0.02, timesteps)
alpha = 1 - beta
alpha_bar = jnp.cumprod(alpha, 0)
alpha_bar = jnp.concatenate((jnp.array([1.]), alpha_bar[:-1]), axis=0)
sqrt_alpha_bar = jnp.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = jnp.sqrt(1 - alpha_bar)
###############################################################

 # Load MNIST dataset

def get_datasets(batch_size: int):
    """Load the MNIST dataset"""

    train_ds = tfds.load('mnist', as_supervised=True, split="train")

    # Normalization helper
    def preprocess(x, y):
        return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))

    # Normalize to [-1, 1], shuffle and batch
    train_ds = train_ds.map(preprocess, tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Return numpy arrays instead of TF tensors while iterating
    return tfds.as_numpy(train_ds)

def forward_noising_1(key, x, t):
    """
    Applies forward noising to the input image `x` based
    on the diffusion coefficient `alpha_hat_t` at time `t`.

    Args:
    key: A JAX random key for generating noise.
    x: The input image.
    t: The time step.

    Returns:
    A tuple containing the noisy image and the generated noise.
    """
    noise = jax.random.normal(key, x.shape)
    alpha_hat_t = jnp.take(alpha_bar, t)
    alpha_hat_t = jnp.reshape(alpha_hat_t, (-1, 1, 1, 1))
    noisy_img = noise * jnp.sqrt(1 - alpha_hat_t) + x * jnp.sqrt(alpha_hat_t)
    return noisy_img, noise

def forward_noising_2(key, x_0, t):
    """
    Applies forward noising to the input image.

    Args:
    key: The random key used for generating noise.
    x_0: The input image.
    t: The time step.

    Returns:
    noisy_image: The image with forward noising applied.
    noise: The generated noise.
    """
    noise = random.normal(key, x_0.shape)
    reshaped_sqrt_alpha_bar_t = jnp.reshape(jnp.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = jnp.reshape(
                                                      jnp.take(one_minus_sqrt_alpha_bar, t), 
                                                      (-1, 1, 1, 1)
                                                      )
    noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
    return noisy_image, noise

# Train step:
def loss_fn(model: UNet, noisy_image: jax.Array, noise: jax.Array, timestep: int):
    """
    Calculates the loss between the predicted noise and the actual noise.

    Args:
    model (UNet): The UNet model used for prediction.
    noisy_image (jax.Array): The input noisy image.
    noise (jax.Array): The actual noise.
    timestep (int): The timestep of the prediction.

    Returns:
    float: The calculated loss.
    """
    pred_noise = model([noisy_image, timestep])
    loss = jnp.mean((noise - pred_noise) ** 2)
    return loss


@nnx.jit
def train_step(model: UNet, 
               optimizer: nnx.Optimizer,  
               noisy_images: jax.Array,
               noise: jax.Array, 
               timestep: jax.Array):
    """
    Train for a single step.

    Args:
    model (UNet): The UNet model.
    optimizer (nnx.Optimizer): The optimizer.
    noisy_images (jax.Array): The noisy images.
    noise (jax.Array): The noise.
    timestep (jax.Array): The timestep.

    Returns:
    Tuple: The gradients and the loss.
    """
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, noisy_images, noise, timestep)
    optimizer.update(grads)
    return grads, loss


# Define the training epoch
def train_epoch(model: UNet, optimizer:nnx.Optimizer, train_ds, rng, timesteps=200):
    """
    Trains the model for one epoch.

    Args:
        model (UNet): The model to be trained.
        optimizer (nnx.Optimizer): The optimizer used for training.
        train_ds: The training dataset.
        rng: The random number generator.

    Returns:
        float: The average training loss for the epoch.
    """

    epoch_loss = []

    for index, batch_images in enumerate(tqdm(train_ds)):
        rng, tsrng = random.split(rng)

        # Generate timestamps for this batch
        timestamps = random.randint(tsrng,
                                    shape=(batch_images.shape[0]),
                                    minval = 0,
                                    maxval=timesteps
                                    )

        # Generating the noise and noisy image for this batch:
        noisy_images, noise = forward_noising_1(rng, batch_images, timestamps)

        # Get loss and gradients:
        _, loss = train_step(model,
                                    optimizer,
                                    noisy_images=noisy_images,
                                    noise=noise,
                                    timestep=timestamps
                                )

        # Update the model:
        epoch_loss.append(loss)
        if index % 10 == 0:
            print(f'loss after step {index} : {loss}')

    train_loss = np.mean(epoch_loss)
    return train_loss


def train(train_ds,
          model:UNet,
          optimizer:nnx.Optimizer,
          ckpt_manager: ocp.CheckpointManager,
          init_epoch: int = 0,
          timesteps: int = 200):
    """
    Trains the model on the given dataset for a specified number of epochs.

    Args:
        train_ds: The training dataset.
        model: The UNet model to train.
        optimizer: The optimizer used for training.
        ckpt_manager: The checkpoint manager for saving model checkpoints.
        init_epoch: The initial epoch number (default is 0).

    Returns:
        The final training loss.

    """
    rng = jax.random.PRNGKey(0)
    train_loss = 0
    for i in range(init_epoch, NUM_EPOCHS):
        rng, input_rng = jax.random.split(rng)
        train_loss = train_epoch(model, optimizer, train_ds, input_rng, timesteps)
        print(f'Train loss after epoch {i} :{train_loss}')
        # saving checkpoint:
        _, state = nnx.split(model)
        metadata = {'epoch': i}
        ckpt_manager.save(i, 
                          args=ocp.args.Composite(state=ocp.args.StandardSave(state), 
                                                 extra_metadata=ocp.args.JsonSave(metadata)))
        ckpt_manager.wait_until_finished()

    return train_loss


def load_checkpoint(ckpt_manager: ocp.CheckpointManager, model: UNet) -> Tuple[UNet, int]:
    """Loads the latest checkpoint from the checkpoint manager and restores the model.

    Args:
        ckpt_manager (ocp.CheckpointManager): The checkpoint manager.
        model (UNet): The UNet model.

    Returns:
        Tuple[UNet, int]: The loaded model and the epoch number.
    """
    latest_step = ckpt_manager.latest_step()
    print(f'Found checkpoint at step {latest_step}')
    abstract_model = nnx.eval_shape(lambda: model)
    graphdef, abstract_state = nnx.split(abstract_model)

    restored = ckpt_manager.restore(latest_step,
                                    args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_state),
                                                            extra_metadata=ocp.args.JsonRestore()))
    restored_state = restored.state
    metadata = restored.extra_metadata
    epoch = metadata['epoch'] + 1
    loaded_model = nnx.merge(graphdef, restored_state)

    print(f'Loaded from the latest step: {latest_step}')

    return loaded_model, epoch
