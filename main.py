"""Main file for running the training."""

import argparse
import orbax.checkpoint as ocp
from flax import nnx
import optax
from train import train, get_datasets, load_checkpoint
from modules import UNet

def main(batch_size: int, checkpoint_dir: str, num_epochs: int):
    """Main function for training the model.

    Args:
        batch_size (int): The batch size for training.
        checkpoint_dir (str): The directory to save checkpoints.
        num_epochs (int): The number of training epochs.

    Returns:
        None
    """

    # Enable checkpointing:
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)

    epoch = 0

    if checkpoint_manager.latest_step() is not None:
        model, epoch = load_checkpoint(
            checkpoint_manager,
            UNet(out_features=32, rngs=nnx.Rngs(0), num_channels=1)
        )
        print(f'Loaded from the epoch: {epoch}')
    else:
        model = UNet(out_features=32, rngs=nnx.Rngs(0), num_channels=1)

    # Initiate Training
    train_ds = get_datasets(batch_size)
    optimizer = nnx.Optimizer(model, optax.adam(1e-4))
    train(train_ds, model, optimizer, checkpoint_manager, epoch, num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='/tmp/checkpoints')
    args = parser.parse_args()
    main(args.batch_size, args.checkpoint_dir, args.num_epochs)
