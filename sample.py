"""Sample from the model."""
import argparse
from enum import Enum
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


class SamplingType(Enum):
    """Enum for the sampling type."""
    DDPM = 'ddpm'
    DDIM = 'ddim'


def backward_denoising_ddpm(x_t, pred_noise, t, alpha, alpha_bar, beta):
    """Applies backward denoising for DDPM.

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

def sample_ddpm(model: nnx.Module, 
                timesteps:int, 
                alpha:jnp.array, 
                alpha_bar:jnp.array, 
                beta:jnp.array, 
                samples_dir:str)->list[jnp.array]:
    """Sample using DDPM.

    Args:
        model (flax.nn.base.Model): The model to sample from.
        timesteps (int): The number of timesteps to sample for.
        alpha (ndarray): The alpha tensor of shape (num_steps,).
        alpha_bar (ndarray): The alpha_bar tensor of shape (num_steps,).
        beta (ndarray): The beta tensor of shape (num_steps,).
        samples_dir (str): The directory to save the samples to.
    """
    # Generate random Gaussian Noise
    x = jax.random.normal(jax.random.PRNGKey(r.randint(0, 100)), (1, 32, 32, 1))

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

    return img_list_ddpm


def backward_denoising_ddim(x_t:jnp.array, 
                            t:int, 
                            pred_noise:jnp.array, 
                            sigma_t:float, 
                            alpha:jnp.array, 
                            alpha_bar:jnp.array)->jnp.array:
    """
    Performs backward denoising with the DDIM method.

    Args:
        x_t (float): The value of x at time t.
        t (int): The time step.
        pred_noise (float): The predicted noise value.
        sigma_t (float): The standard deviation at time t.
        alpha (ndarray): The array of alpha values.
        alpha_bar (ndarray): The array of alpha_bar values.

    Returns:
        float: The denoised value of x at time t.

    """
    alpha_t_1 = jnp.take(alpha, t-1)
    alpha_bar_t = jnp.take(alpha_bar, t)

    pred_x_0 = (x_t - jnp.sqrt(1 - alpha_bar_t) * pred_noise) / (jnp.sqrt(alpha_bar_t))
    point_to_xt = jnp.sqrt(1 - alpha_t_1 - sigma_t ** 2 ) * pred_noise * x_t
    random_noise = jax.random.normal(jax.random.PRNGKey(r.randint(1, 100)))

    ret = jnp.sqrt(alpha_t_1) * pred_x_0 + point_to_xt + sigma_t * random_noise

    return ret

def sample_ddim(model: nnx.Module, 
                timesteps:int, 
                alpha:jnp.array, 
                alpha_bar:jnp.array, 
                samples_dir:str) -> list[jnp.array]:
    """
    Runs the diffusion model for a given number of timesteps using DDIM sampling.

    Args:
        model (nnx.Module): The diffusion model.
        timesteps (int): The total number of timesteps to run the diffusion model.
        alpha (jnp.array): The alpha parameter.
        alpha_bar (jnp.array): The alpha_bar parameter.
        samples_dir (str): The directory to save the generated images.

    Returns:
        list[jnp.array]: A list of intermediate images generated during the diffusion process.
    """

    img_list_ddim = []

    x = jax.random.normal(jax.random.PRNGKey(r.randint(1, 100)), (1, 32, 32, 1))

    # Add the first noise to the list:
    img_list_ddim.append(jnp.squeeze(x, (0, -1)))

    # Define number of inference loops to run:
    inference_timesteps = 10

    inference_range = range(0, timesteps, timesteps // inference_timesteps)

    for _, i in tqdm(enumerate(reversed(range(inference_timesteps))), total=inference_timesteps):

        t = jnp.expand_dims(inference_range[i], 0)
        print(f'current t:{t[0]}')

        pred_noise = model((x, t))
        x = backward_denoising_ddim(x, t, pred_noise, 0, alpha, alpha_bar)

        plt.imshow(jnp.squeeze(x, (0, -1)), cmap='gray')
        plt.savefig(os.path.join(samples_dir, f'ddim_sample_{t[0]}.png'))
        img_list_ddim.append(jnp.squeeze(x, (0, -1)))

    # Display the final image:
    plt.imshow(jnp.squeeze(x, (0, -1)), cmap='gray')
    plt.savefig(os.path.join(samples_dir, 'ddim_sample_final.png'))

    return img_list_ddim


def main(checkpoint_dir:str, samples_dir:str, sampling_type:SamplingType=SamplingType.DDPM):
    """Main function for running the sample code."""
    ## STEP 1: Load the model from the checkpoint
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir)

    assert checkpoint_manager.latest_step() is not None, 'No checkpoint found.'

    model, _ = load_checkpoint(
        checkpoint_manager,
        UNet(out_features=32, rngs=nnx.Rngs(0), num_channels=1)
    )

    # STEP 2: Ensure the samples diretory exists:
    os.makedirs(samples_dir, exist_ok=True)

    # STEP 3: Run the sampling code
    # Define the alpha, alpha_bar and beta values:
    # TODO: Make these values configurable
    timesteps = 200
    beta = jnp.linspace(0.0001, 0.02, timesteps)
    alpha = 1 - beta
    alpha_bar = jnp.cumprod(alpha, 0)
    alpha_bar = jnp.concatenate((jnp.array([1.]), alpha_bar[:-1]), axis=0)

    if sampling_type == SamplingType.DDPM:
        sample_ddpm(model, timesteps, alpha, alpha_bar, beta, samples_dir)
    elif sampling_type == SamplingType.DDIM:
        print('ddim sampling!')
        sample_ddim(model, timesteps, alpha, alpha_bar, samples_dir)
    else:
        raise ValueError('Invalid sampling type.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='/tmp/checkpoints')
    parser.add_argument('--samples_dir', type=str, default='/tmp/samples')
    parser.add_argument(
        '--sampling_type',
        type=SamplingType,
        default=SamplingType.DDPM,
        choices=list(SamplingType)
    )
    args = parser.parse_args()

    main(args.checkpoint_dir, args.samples_dir, args.sampling_type)
