"""Training file for simple diffusion model."""

import jax
from flax import nnx
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import tensorflow_datasets as tfds
from modules import UNet
from tqdm import tqdm
import jax.random as random

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
  # Load the MNIST dataset
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
  noise = jax.random.normal(key, x.shape)
  alpha_hat_t = jnp.take(alpha_bar, t)
  alpha_hat_t = jnp.reshape(alpha_hat_t, (-1, 1, 1, 1))
  noisy_img = noise * jnp.sqrt(1 - alpha_hat_t) + x * jnp.sqrt(alpha_hat_t)
  return noisy_img, noise

def forward_noising_2(key, x_0, t):
  noise = random.normal(key, x_0.shape)
  reshaped_sqrt_alpha_bar_t = jnp.reshape(jnp.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
  reshaped_one_minus_sqrt_alpha_bar_t = jnp.reshape(jnp.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
  noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
  return noisy_image, noise

# Train step:
def loss_fn(model: UNet, noisy_image: jax.Array, noise:jax.Array, timestep: int):
  pred_noise = model([noisy_image, timestep])
  loss = jnp.mean((noise - pred_noise) ** 2)
  return loss

@nnx.jit
def train_step(model: UNet, optimizer: nnx.Optimizer,  noisy_images: jax.Array, noise:jax.Array, timestep: jax.Array):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(model, noisy_images, noise, timestep)
  optimizer.update(grads)
  return grads, loss


# Define the training epoch
def train_epoch(epoch_num, model: UNet, optimizer:nnx.Optimizer, train_ds, batch_size, rng):

  epoch_loss = []

  for index, batch_images in enumerate(tqdm(train_ds)):
    rng, tsrng = random.split(rng)
    # print(batch_images.shape)

    # Generate timestamps for this batch
    timestamps = random.randint(tsrng,
                                shape=(batch_images.shape[0]),
                                minval = 0,
                                maxval=timesteps
                                )

    # Generating the noise and noisy image for this batch:
    noisy_images, noise = forward_noising_1(rng, batch_images, timestamps)
    # print(noisy_images.shape)
    # print(f'noise shape : {noise.shape}')
    # Get loss and gradients:
    grads, loss = train_step(model,
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
          init_epoch: int = 0):

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)


  # if resume_state is not None:
  #   state = resume_state
  #   print('resuming from a previously trained state.')
  # else:
  #   state = create_train_state(init_rng)
  # print(jax.tree.map(jnp.shape, state.params))
  train_loss = 0
  for i in range(init_epoch, NUM_EPOCHS):

    rng, input_rng = jax.random.split(rng)

    train_loss = train_epoch(i, model, optimizer, train_ds, BATCH_SIZE, input_rng)
    print(f'Train loss after epoch {i} :{train_loss}')
    # log_states.append(state)
    # saving checkpoint:
    _,state = nnx.split(model)
    metadata = {'epoch':i}
    ckpt_manager.save(i, args=ocp.args.Composite(state=ocp.args.StandardSave(state), extra_metadata=ocp.args.JsonSave(metadata)))


  return train_loss


# Enable checkpointing:
ckpt_dir = '~/checkpoints'
options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
ckpt_manager = ocp.CheckpointManager(ckpt_dir, options=options)
# TBD: Fix loading from saved ckpt
restored_state = None
epoch = 0

if ckpt_manager.latest_step() is not None:
  # load from the latest checkpoint
  latest_step = ckpt_manager.latest_step()
  print(f'found checkpoint at step {latest_step}')
  abstract_model = nnx.eval_shape(lambda: UNet(out_features=32, rngs=nnx.Rngs(0), num_channels=1))
  graphdef, abstract_state = nnx.split(abstract_model)

  restored = ckpt_manager.restore(latest_step, args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_state),
                                                                       extra_metadata=ocp.args.JsonRestore()))
  restored_state = restored.state
  metadata = restored.extra_metadata
  epoch = metadata['epoch'] + 1
  model = nnx.merge(graphdef, restored_state)
  # epoch = restored_ckpt['epoch']
  print(f' loaded from the latest step: {latest_step}')

# Initiate Training
train_ds = get_datasets(BATCH_SIZE)
model = UNet(out_features=32, rngs=nnx.Rngs(0), num_channels=1)
optimizer = nnx.Optimizer(model, optax.adam(1e-4))
trained_state = train(train_ds,
                      model,
                      optimizer,
                      ckpt_manager,
                      epoch)



