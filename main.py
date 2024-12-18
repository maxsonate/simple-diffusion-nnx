"""Main file for running the training."""

import orbax.checkpoint as ocp
from flax import nnx
import optax
from train import train, get_datasets
from modules import UNet

BATCH_SIZE = 64


# Enable checkpointing:
ckpt_dir = '/home/siamak/checkpoints'
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

    restored = ckpt_manager.restore(latest_step, 
                                     args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_state),
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
