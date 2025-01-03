"""Sample from the model."""
import os
import random as r
from tqdm import tqdm
import matplotlib.pyplot as plt

from flax import nnx

import jax
from jax import numpy as jnp

import orbax.checkpoint as ocp

from train import load_checkpoint
from modules import UNet


def backward_denoising_ddpm(x_t, pred_noise, t, alpha, alpha_bar, beta):
    """Applies backward denoising for DDPM (Diffusion Models with Denoising Prior).

    Args:
        x_t (ndarray): The input tensor of shape (batch_size, ...).
        pred_noise (ndarray): The predicted noise tensor of shape (batch_size, ...).
        t (int): The time step.
        alpha (ndarray): The alpha tensor of shape (num_steps,).
        alpha_bar (ndarray): The alpha_bar tensor of shape (num_steps,).
        beta (ndarray): The beta tensor of shape (num_steps,).

    Returns:
        ndarray: The denoised output tensor of shape (batch_size, ...).
    """
    alpha_t = jnp.take(alpha, t)
    alpha_bar_t = jnp.take(alpha_bar, t)

    z = jax.random.normal(jax.random.PRNGKey(r.randint(1, 100)), shape=x_t.shape)
    var = jnp.take(beta, t)

    mean = (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t) * pred_noise

    ret = 1 / jnp.sqrt(alpha_t) * (x_t - mean) + (var ** 0.5) * z
    return ret

#TODO: Add argparse
#TODO: Add ddim


def main():
    """Main function for running the sample code."""
    ## STEP 1: Load the model
    checkpoint_dir = '/tmp/checkpoints'
    samples_dir = "/tmp/samples"
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir)

    assert checkpoint_manager.latest_step() is not None, 'No checkpoint found.'

    model, epoch = load_checkpoint(
        checkpoint_manager,
        UNet(out_features=32, rngs=nnx.Rngs(0), num_channels=1)
    )
    print(f'Loaded from the epoch: {epoch}')


    ##################### CLEAN THIS SECTION #####################:
    timesteps = 200
    beta = jnp.linspace(0.0001, 0.02, timesteps)
    alpha = 1 - beta
    alpha_bar = jnp.cumprod(alpha, 0)
    alpha_bar = jnp.concatenate((jnp.array([1.]), alpha_bar[:-1]), axis=0)
    ###############################################################

    # Generate random Gaussian Noise

    x = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32, 1))

    # Create a list to store output images
    img_list_ddpm = []
    img_list_ddpm.append(jnp.squeeze(x, (0, -1)))
    # Need to stop at t = 1, because at t=0 the alpha_bar is 1, and denom will be 0.
    for t in tqdm(range(timesteps - 1)):
        step_from_last = jnp.expand_dims(jnp.array(timesteps - t -1, jnp.int32), 0)

        # Predict noise using U-Net
        pred_noise = model((x, step_from_last))

        # Obtain the output from the noise using the formula seen before
        x = backward_denoising_ddpm(x, pred_noise, step_from_last, alpha, alpha_bar, beta)

        # Log the image after every 25 iterations
        if t % 25 == 0:
            img_list_ddpm.append(jnp.squeeze(x, (0, -1)))
            plt.imshow(jnp.squeeze(x, (0, -1)), cmap = 'gray')
            plt.savefig(os.path.join(samples_dir, f'ddpm_sample_{t}.png'))

    # Display the final generated image:
    plt.imshow(jnp.squeeze(x, (0, -1)), cmap = 'gray')
    plt.savefig(os.path.join(samples_dir, 'ddpm_sample_final.png'))
