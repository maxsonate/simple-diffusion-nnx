### NNX modules for simple diffusion.

import jax
import optax
import math
from functools import partial
import os
from typing import Any, NamedTuple, Mapping
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random as r
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from flax import nnx
from flax.nnx import Optimizer
from typing import Callable
from tqdm.notebook import tqdm
from PIL import Image
from IPython import display

import orbax.checkpoint as ocp
import penzai
from penzai import pz
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)



 # Load MNIST dataset

def get_datasets(batch_size: int):
  # Load the MNIST dataset
  train_ds = tfds.load('mnist', as_supervised=True, split="train")

  # Normalization helper
  def preprocess(x, y):
    return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))

  # Normalize to [-1, 1], shuffle and batch
  train_ds = train_ds.map(preprocess, tf.data.AUTOTUNE)
  train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

  # Return numpy arrays instead of TF tensors while iterating
  return tfds.as_numpy(train_ds)


class SinusoidalEmbedding_(nnx.Module):

  def __init__(self, dim:int = 32):
    self.dim = dim

  def __call__(self, inputs: jax.Array):
    half_dim = self.dim // 2
    emb = jnp.log(10000) / half_dim
    emb = jnp.exp(jnp.arange(half_dim) * -emb)

    emb = inputs[..., None] * emb[None, :]
    ret = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis = -1)

    return ret


class TimeEmbedding_(nnx.Module):

  def __init__(self, rngs:nnx.Rngs, dim:int = 32):
    self.dim = dim

    self.linear_1 = nnx.Linear(self.dim, self.dim * 4, rngs=rngs)
    self.linear_2 = nnx.Linear(in_features=self.dim * 4,
                               out_features=self.dim * 4,
                               rngs=rngs
                               )


  def __call__(self, inputs: jax.Array):
    time_dim = 4 * self.dim # Why this is multiplied by 4 here?

    se = SinusoidalEmbedding_(self.dim)(inputs)

    x = self.linear_1(se)
    x = nnx.gelu(x)
    ret = self.linear_2(x)

    return ret


class Attn(nnx.Module):

  def __init__(self, in_features:int, dim:int, num_heads:int, rngs:nnx.Rngs,kernel_init: Callable=nnx.initializers.xavier_uniform(), bidir:bool=False):

    self.in_features = in_features
    self.dim = dim
    self.num_heads = num_heads
    self.kernel_init = kernel_init
    self.bidir = bidir
    self.Dh = self.dim // self.num_heads
    self.q = nnx.LinearGeneral(in_features=in_features,
                               out_features=(self.num_heads, self.Dh),
                               rngs=rngs)

    self.k = nnx.LinearGeneral(in_features=in_features,
                               out_features=(self.num_heads, self.Dh),
                               rngs=rngs)

    self.v = nnx.LinearGeneral(in_features=in_features,
                               out_features=(self.num_heads, self.Dh),
                               rngs=rngs)

    self.out = nnx.LinearGeneral(in_features=(self.num_heads, self.Dh), out_features=in_features, axis=(-2, -1), rngs=rngs)

  def __call__(self, inputs:jnp.ndarray, rngs=nnx.Rngs):


    q_BxLxHxDh = self.q(inputs)
    k_BxLxHxDh = self.k(inputs)
    v_BxLxHxDh = self.v(inputs)

    q_BxLxHxDh = q_BxLxHxDh // self.Dh ** 0.5
    attn_BxHxLxL = jnp.einsum('...lhd,...nhd->...hln', q_BxLxHxDh, k_BxLxHxDh)

    # create causal attn mask:
    L = inputs.shape[1]
    mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, L, L)))
    _NEG_INF = jnp.finfo(jnp.float32).min

    attn_BxHxLxL = jnp.where(mask_1x1xLxL + jnp.astype(self.bidir, jnp.int32),
                             attn_BxHxLxL,
                             _NEG_INF)

    attn_BxHxLxL = jax.nn.softmax(attn_BxHxLxL, axis=-1)
    attn_BxLxHxDh = jnp.einsum(
        '...hln,...nhd->...lhd',
        attn_BxHxLxL,
        v_BxLxHxDh
        )

    ret = self.out(attn_BxLxHxDh)

    return ret


class Block(nnx.Module):
  def __init__(self, in_features:int, out_features:int, rngs:nnx.Rngs, num_groups:int = 8):

    self.conv = nnx.Conv(in_features, out_features, (3, 3), rngs=rngs)
    self.norm = nnx.GroupNorm(out_features, num_groups = num_groups, rngs=rngs)

  def __call__(self, inputs: jax.Array):

    conv = self.conv(inputs)
    norm = self.norm(conv)
    activation = nnx.silu(norm)

    return activation


class ResNetBlock(nnx.Module):

  def __init__(self, in_features:int,
               out_features:int,
               time_emb_dim:int,
               rngs:nnx.Rngs,
               groups:int=8
               ):
    self.conv = nnx.Conv(in_features=in_features,
                         out_features=out_features,
                         kernel_size=(1, 1),
                         rngs=rngs
                         )

    self.block = Block(in_features=in_features,
                       out_features=out_features,
                       rngs=rngs,
                       num_groups=groups
                       )
    self.time_linear = nnx.Linear(in_features=time_emb_dim,
                                  out_features=out_features,
                                  rngs=rngs
                                  )


  def __call__(self, inputs:jax.Array, time_embed:jax.Array | None=None):

    x = self.block(inputs)

    if time_embed is not None:
      time_embed = nnx.silu(time_embed)
      time_embed = self.time_linear(time_embed)
      x = time_embed[:, None, None, :] + x # This is different than the main code, double check.


    x = self.conv(inputs) + x

    return x


class UNet(nnx.Module):

  def __init__(self,
               rngs:nnx.Rngs,
               num_channels: int,
               out_features:int = 8,
               num_groups:int = 8,
               num_heads: int = 8,
               dim_scale_factor = (1, 2, 4, 8)
               ):

    self.conv = nnx.Conv(in_features=num_channels,
                         out_features=out_features, # TBD: changed this from // 3 * 2
                         kernel_size=(7, 7),
                         padding=((3, 3), (3, 3)),
                         rngs=rngs
                         )

    self.time_emb = TimeEmbedding_(dim = out_features, rngs=rngs)

    dims = [out_features * i for i in dim_scale_factor]


    # Build the downsampling module list:

    self.downsampling_resnets = []
    self.downsampling_attn = []
    self.downsampling_convs = []

    in_features = out_features

    for index, dim in enumerate(dims):
      self.downsampling_resnets.append(
          nnx.Sequential(
          ResNetBlock(in_features=in_features,
                      out_features=dim,
                      time_emb_dim=out_features * 4,
                      rngs=rngs,
                      groups=num_groups),
          ResNetBlock(in_features=dim,
                      out_features=dim,
                      time_emb_dim=out_features * 4,
                      rngs=rngs,
                      groups=num_groups)
          )
      )
      self.downsampling_attn.append(
          nnx.Sequential(
          Attn(in_features=dim, dim=dim, num_heads = num_heads, rngs=rngs),
          nnx.GroupNorm(num_features=dim, num_groups=num_groups, rngs=rngs)
          )
      )
      if index != len(dims) - 1 :
        self.downsampling_convs.append(
            nnx.Conv(in_features=dim,
                    out_features=dim,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    rngs=rngs)
            )
      in_features=dim


    # Mid block modules

    self.mid_block_resnets = [ResNetBlock(in_features=dims[-1], out_features=dims[-1], groups=num_groups, time_emb_dim=out_features * 4, rngs=rngs)] * 2
    self.mid_block_norm = nnx.GroupNorm(num_features=dims[-1], num_groups=num_groups, rngs=rngs)
    self.mid_block_attn = Attn(in_features=dims[-1], dim=dims[-1], num_heads=num_heads, rngs=rngs)

    # Upsampling modules:

    self.upsampling_resnets = []
    self.upsampling_attns = []
    self.upsampling_convs = []

    in_features = dims[-1]
    for index, dim in enumerate(reversed(dims)):
      self.upsampling_resnets.append(
          nnx.Sequential(
          ResNetBlock(in_features=in_features + dim, out_features=dim, time_emb_dim=out_features * 4, rngs=rngs, groups=num_groups),
          ResNetBlock(in_features=dim, out_features=dim, time_emb_dim=out_features, rngs=rngs, groups=num_groups),
          ))

      self.upsampling_attns.append(nnx.Sequential(
          Attn(in_features=dim, dim=dim, num_heads = num_heads, rngs=rngs),
          nnx.GroupNorm(num_features=dim, num_groups=num_groups, rngs=rngs))
      )

      if index != len(dims) - 1 :
        self.upsampling_convs.append(
            nnx.ConvTranspose(in_features=dim,
                    out_features=dim,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    rngs=rngs)
            )
      in_features=dim


    # Final Resnet block and output Convolution layer
    self.final_block = nnx.Sequential(
        ResNetBlock(in_features=dims[0],
                    out_features=dims[0],
                    time_emb_dim=out_features * 4,
                    rngs=rngs,
                    groups=num_groups),
        nnx.Conv(
            in_features=dims[0],
            out_features=num_channels,
            kernel_size=(1, 1),
            rngs=rngs)
    )


  def __call__(self, inputs):
    inputs, time = inputs
    channels = inputs.shape[-1]

    x = self.conv(inputs)
    time_emb = self.time_emb(time)

    pre_downsampling = []

    # downsampling phase
    for index in range(len(self.downsampling_resnets)):
      x = self.downsampling_resnets[index](x, time_emb)
      attn = self.downsampling_attn[index](x, time_emb)
      x = attn + x

      pre_downsampling.append(x)

      # Saving this output for residual connection with the upsampling layer
      if index != len(self.downsampling_resnets) - 1:
        x = self.downsampling_convs[index](x)

    # Middle block

    x = self.mid_block_resnets[0](x, time_emb)
    attn = self.mid_block_attn(x)

    x = x + attn
    x = self.mid_block_resnets[1](x, time_emb)


    # Upsampling phase
    for index in range(len(self.upsampling_resnets)):

      x = jnp.concatenate([pre_downsampling.pop(), x], axis=-1) #TODO: I don't understand how this happens

      x = self.upsampling_resnets[index](x)
      attn = self.upsampling_attns[index](x)
      x = attn + x

      if index != len(self.upsampling_resnets) - 1:
        x = self.upsampling_convs[index](x)

      # Final Resnet block and output convolutional layer
    x = self.final_block(x)

    return x

