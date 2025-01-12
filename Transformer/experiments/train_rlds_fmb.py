import datetime
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags, logging
from flax.training import checkpoints
from flax.traverse_util import flatten_dict
from ml_collections import config_flags

from orca.data.rlds.rlds_dataset import RLDSDataset
from orca.data.utils.text_processing import text_processors
from orca.model import create_model_def_fmb
from orca.model.tokenizers import weights_loaders
from orca.train_utils import (
    Timer,
    create_train_state,
    format_name_with_config,
    initialize_compilation_cache,
    shard_batch,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

tf.config.experimental_run_functions_eagerly(True)

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_string("img_obs_key", "image_wrist_1:image_wrist_2:image_side_1:image_side_2", "dict key for image observations")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "train_config.py:transformer_bc"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    os.path.join(config_dir, "data_config.py:all"),
    "File path to the bridgedata configuration.",
    lock_config=False,
)


def main(_):
    # initialize_compilation_cache(cache_dir="/media/nvmep3p/users/xuc/jax_cache")
    initialize_compilation_cache(cache_dir="media/sblee/170d6766-97d9-4917-8fc6-7d6ae84df8961/SSD2/xuc/jax_cache")
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        name=name,
        mode="offline" if FLAGS.debug else "online",
        **FLAGS.config.wandb,
    )
    if FLAGS.config.save_dir is not None:
        save_dir = tf.io.gfile.join(
            FLAGS.config.save_dir,
            FLAGS.config.wandb.project,
            FLAGS.config.wandb.group or "",
            wandb_id,
        )
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        tf.io.gfile.makedirs(save_dir)
        with tf.io.gfile.GFile(
            os.path.join(save_dir, "config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        logging.info("save_dir not passed in, not saving checkpoints")

    # load datasets
    logging.info(f"Loading data from {FLAGS.config.data_path}")
    obs_horizon = FLAGS.config.get("obs_horizon")
    image_keys = FLAGS.img_obs_key.split(":")

    train_data = RLDSDataset(
        dataset_names=FLAGS.config.dataset_name,
        image_obs_key=image_keys,
        tfds_data_dir=FLAGS.config.data_path,
        image_processor="default",
        seed=FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        # batch_size=1,
        obs_horizon=obs_horizon,
        **FLAGS.config.dataset_kwargs
    )
    val_data = RLDSDataset(
        dataset_names=FLAGS.config.dataset_name,
        image_obs_key=image_keys,
        image_processor="default",
        tfds_data_dir=FLAGS.config.data_path,
        seed=FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        obs_horizon=obs_horizon,
        train=False,
        **FLAGS.config.dataset_kwargs
    )
    # get example data without iterator
    # example_train_data = train_data.get_example_data()
    train_data_iter = train_data.get_iterator()

    tf.print("Loading data successful")
    example_batch = next(train_data_iter)
    # try:
    #     example_batch = next(train_data_iter)
    #     tf.print("Batch Retrieved Successfully")
    # except tf.errors.InvalidArgumentError as e:
    #     tf.print("Error:", e)
    #     raise e

    # logging.info(f"Batch keys: {example_batch.keys()}")
    # logging.info(f"Observation shape: {example_batch['observations'].shape}")
    logging.info(f"Batch size: {example_batch['actions'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['actions'].shape[0] // num_devices}"
    )

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    example_batch = shard_batch(example_batch, sharding)

    model_def = create_model_def_fmb(
        action_dim=example_batch["actions"].shape[-1],
        time_sequence_length=obs_horizon,
        **FLAGS.config.model.to_dict(),
    )

    # pretrained weights to load
    pretrained_loaders = [weights_loaders[w] for w in FLAGS.config.pretrained_weights]

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.config.optimizer.learning_rate,
        warmup_steps=FLAGS.config.optimizer.warmup_steps,
        decay_steps=FLAGS.config.optimizer.decay_steps,
        end_value=0.0,
    )
    tx = optax.adam(lr_schedule)
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)

    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=(
            example_batch["observations"],
            example_batch["actions"],
        ),
        pretrained_loaders=pretrained_loaders,
    )
    if FLAGS.config.resume_path is not None:
        train_state = checkpoints.restore_checkpoint(
            FLAGS.config.resume_path, target=train_state
        )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    train_state = jax.device_put(
        jax.tree_map(jnp.array, train_state), sharding.replicate()
    )

    param_count = sum(
        np.prod(x.shape)
        for x in jax.tree_leaves(jax.tree_map(jnp.array, train_state.params))
    )
    print(f"Number of parameters: {param_count:} ({param_count / 1e6:.2f}M)")
    def loss_fn(params, state, batch, rng, train=True):
        info = state.apply_fn(
            {"params": params},
            batch["observations"],
            batch["actions"],
            train=train,
            rngs={"dropout": rng},
        )
        return info["loss"], info

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    @jax.jit
    def eval_step(state, batch):
        loss, info = loss_fn(state.params, state, batch, state.rng, train=False)
        return info

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch = shard_batch(next(train_data_iter), sharding)
        timer.tock("dataset")

        timer.tick("train")
        train_state, update_info = train_step(train_state, batch)
        timer.tock("train")

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            timer.tick("val")
            metrics = []
            for batch in val_data.get_iterator():
                metrics.append(eval_step(train_state, batch))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_log({"validation": metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, train_state, step=i + 1, keep=1e6
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )


if __name__ == "__main__":
    app.run(main)
