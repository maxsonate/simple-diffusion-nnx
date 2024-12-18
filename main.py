"""Main file for running the training."""

import orbax.checkpoint as ocp
from flax import nnx
import optax
from train import train, get_datasets, load_checkpoint
from modules import UNet

BATCH_SIZE = 64

# Enable checkpointing:
checkpoint_dir = '/tmp/checkpoints'
options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)

epoch = 0

if checkpoint_manager.latest_step() is not None:
    model, epoch = load_checkpoint(checkpoint_manager, UNet(out_features=32, rngs=nnx.Rngs(0), num_channels=1))
    print(f'Loaded from the epoch: {epoch}')

# Initiate Training
train_ds = get_datasets(BATCH_SIZE)
model = UNet(out_features=32, rngs=nnx.Rngs(0), num_channels=1)
optimizer = nnx.Optimizer(model, optax.adam(1e-4))
trained_state = train(train_ds, model, optimizer, checkpoint_manager, epoch)
