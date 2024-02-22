import time
from typing import Iterator, Tuple, Any, Dict, Union, Callable, Iterable

import glob
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tqdm

import itertools
from multiprocessing import Pool
from functools import partial
from tensorflow_datasets.core import download
from tensorflow_datasets.core import split_builder as split_builder_lib
from tensorflow_datasets.core import naming
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import utils
from tensorflow_datasets.core import writer as writer_lib
from tensorflow_datasets.core import example_serializer
from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import file_adapters

Key = Union[str, int]
# The nested example dict passed to `features.encode_example`
Example = Dict[str, Any]
KeyExample = Tuple[Key, Example]

N_WORKERS = 20 # number of parallel workers for data conversion
MAX_PATHS_IN_MEMORY = 400            # number of paths converted & stored in memory before writing to disk
                                    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                                    # note that one path may yield multiple episodes and adjust accordingly


# _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

# OBJECT_IDS = {
#     1: 'rectangle',
#     2: 'round',
#     3: 'oval',
#     4: 'hexagon',
#     5: 'arch',
#     6: 'square-circle',
#     7: 'double-square',
#     8: '3 prong',
#     9: 'star',
# }

# COLOR_IDS = {
#     1: 'brown',
#     2: 'red',
#     3: 'jeans red',
#     4: 'yellow',
#     5: 'green',
#     6: 'jeans blue',
#     7: 'dark blue',
#     8: 'purple',
# }

# SIZE_IDS = {
#     'L': 'large',
#     'M': 'medium',
#     'S': 'small'
# }

# LENGTH_IDS = {
#     'L': 'long',
#     'S': 'short'
# }

# DISTRACTOR_IDS = {
#     'n': False,
#     'y': True
# }

# INSTRUCT_EMBEDS = {}
# for object in tqdm.tqdm(OBJECT_IDS.values()):
#     for color in COLOR_IDS.values():
#         for size in SIZE_IDS.values():
#             for length in LENGTH_IDS.values():
#                 for distractor in DISTRACTOR_IDS.values():
#                     for orientation in ['horizontal', 'vertical']:
#                         instruct = f"Pick up the {size} {length} {color} {object} piece {orientation}ly and insert {'with' if distractor else 'without'} distractors."
#                         instruct_embed = _embed([instruct])[0].numpy()
#                         INSTRUCT_EMBEDS[instruct] = instruct_embed


# def _parse_instruct(episode_path):
#     filename = os.path.basename(episode_path)
#     elems = filename.split('_')
#     object = OBJECT_IDS[int(elems[0])]
#     size = SIZE_IDS[elems[1]]
#     length = LENGTH_IDS[elems[2]]
#     color = COLOR_IDS[int(elems[3])]
#     orientation = elems[4]
#     distractor = DISTRACTOR_IDS[elems[5]]
#     return f"Pick up the {size} {length} {color} {object} piece {orientation}ly and insert {'with' if distractor else 'without'} distractors."


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split."""

    def _parse_example(data):
        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(data['actions'].shape[0]):
            #try:
            if True:
                episode.append({
                    'observation': {
                        'image_side_1': data['obs/side_1'][i].astype(np.uint8),
                        'image_side_2': data['obs/side_2'][i].astype(np.uint8),
                        'image_wrist_1': data['obs/wrist_1'][i].astype(np.uint8),
                        'image_wrist_2': data['obs/wrist_2'][i].astype(np.uint8),
                        # 'image_side_1_depth': data['obs/side_1_depth'][i].astype(np.float32),
                        # 'image_side_2_depth': data['obs/side_2_depth'][i].astype(np.float32),
                        # 'image_wrist_1_depth': data['obs/wrist_1_depth'][i].astype(np.float32),
                        # 'image_wrist_2_depth': data['obs/wrist_2_depth'][i].astype(np.float32),
                        'joint_pos': data['obs/q'][i].astype(np.float32),
                        'joint_vel': data['obs/dq'][i].astype(np.float32),
                        'eef_pose': data['obs/tcp_pose'][i].astype(np.float32),
                        'eef_vel': data['obs/tcp_vel'][i].astype(np.float32),
                        'eef_force': data['obs/tcp_force'][i].astype(np.float32),
                        'eef_torque': data['obs/tcp_torque'][i].astype(np.float32),
                        'state_gripper_pose': data['obs/gripper_pose'][i].astype(np.float32),
                        'primitive': data['primitive'][i],
                        'peg_id': np.int_(data['peg_id'][i]).astype(np.uint8),
                    },
                    'action': data['actions'][i].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (data['actions'].shape[0] - 1)),
                    'is_first': i == 0,
                    'is_last': i == (data['actions'].shape[0] - 1),
                    'is_terminal': i == (data['actions'].shape[0] - 1)
                })
            #except:
            #    return None

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return sample

    for episode_path in paths:
        data = np.load(episode_path, allow_pickle=True).item()  # this is a list of dicts in our case
        # instruct = _parse_instruct(episode_path)
        # instruct_embed = INSTRUCT_EMBEDS[instruct]
        sample = _parse_example(data)
        if sample is None:
            yield None
        else:
            yield episode_path, sample


class FmbDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('12.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '2.0.0': 'Jpeg compression',
      '3.0.0': 'Filtered for ICRA',
      '4.0.0': 'More Data for 3 Pegs',
      '5.0.0': 'Filter out insert 2, 4',
      '6.0.0': 'Change star for hexagon',
      '7.0.0': 'Composition board 100 demos',
      '8.0.0': 'Composition board 30 more demos and 300 more grasp',
      '9.0.0': 'Composition board unfilered',
      '10.0.0': 'Composition board determinstic grasp + reorient + insert',
      '11.0.0': '57 more demos and seperated grasp peg_id',
      '12.0.0': '150 trajectories of new board 1'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_side_1': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Side camera 1 RGB observation.',
                        ),
                        'image_side_2': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Side camera 2 RGB observation.',
                        ),
                        'image_wrist_1': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera 1 RGB observation.',
                        ),
                        'image_wrist_2': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera 2 RGB observation.',
                        ),
                        # 'image_side_1_depth': tfds.features.Tensor(
                        #     shape=(256, 256,),
                        #     dtype=np.float32,
                        #     doc='Side camera 1 depth observation.',
                        # ),
                        # 'image_side_2_depth': tfds.features.Tensor(
                        #     shape=(256, 256,),
                        #     dtype=np.float32,
                        #     doc='Side camera 2 depth observation.',
                        # ),
                        # 'image_wrist_1_depth': tfds.features.Tensor(
                        #     shape=(256, 256,),
                        #     dtype=np.float32,
                        #     doc='Wrist camera 1 depth observation.',
                        # ),
                        # 'image_wrist_2_depth': tfds.features.Tensor(
                        #     shape=(256, 256,),
                        #     dtype=np.float32,
                        #     doc='Wrist camera 2 depth observation.',
                        # ),
                        'joint_pos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint position.',
                        ),
                        'joint_vel': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint velocity.',
                        ),
                        'eef_pose': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot EEF pose.',
                        ),
                        'eef_vel': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Robot EEF velocity.',
                        ),
                        'eef_force': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='Robot EEF force.',
                        ),
                        'eef_torque': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='Robot EEF torque.',
                        ),
                        'state_gripper_pose': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Gripper pose of the robot.'
                        ),
                        'primitive': tfds.features.Text(
                            doc='Primitive Skill Instruction for this time step.'
                        ),
                        'peg_id': tfds.features.Scalar(
                            dtype=np.uint8,
                            doc='Object ID of the object being manipulated.'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot end-effector twists.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define data splits."""
        return {
                'train': glob.glob('/media/nvmep3p/fmb2/processed_demos/board_1/np/*.npy'),
        }

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        split_paths = self._split_paths()
        return {split: _generate_examples(paths=split_paths[split]) for split in split_paths}

    def _generate_examples(self):
        pass  # this is implemented in global method to enable multiprocessing

    def _download_and_prepare(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
            self,
            dl_manager: download.DownloadManager,
            download_config: download.DownloadConfig,
    ) -> None:
        """Generate all splits and returns the computed split infos."""
        split_builder = ParallelSplitBuilder(
            split_dict=self.info.splits,
            features=self.info.features,
            dataset_size=self.info.dataset_size,
            max_examples_per_split=download_config.max_examples_per_split,
            beam_options=download_config.beam_options,
            beam_runner=download_config.beam_runner,
            file_format=self.info.file_format,
            shard_config=download_config.get_shard_config(),
            split_paths=self._split_paths(),
            parse_function=self._generate_examples,
        )
        split_generators = self._split_generators(dl_manager)
        split_generators = split_builder.normalize_legacy_split_generators(
            split_generators=split_generators,
            generator_fn=self._generate_examples,
            is_beam=False,
        )
        dataset_builder._check_split_names(split_generators.keys())

        # Start generating data for all splits
        path_suffix = file_adapters.ADAPTER_FOR_FORMAT[
            self.info.file_format
        ].FILE_SUFFIX

        split_info_futures = []
        for split_name, generator in utils.tqdm(
                split_generators.items(),
                desc="Generating splits...",
                unit=" splits",
                leave=False,
        ):
            filename_template = naming.ShardedFileTemplate(
                split=split_name,
                dataset_name=self.name,
                data_dir=self.data_path,
                filetype_suffix=path_suffix,
            )
            future = split_builder.submit_split_generation(
                split_name=split_name,
                generator=generator,
                filename_template=filename_template,
                disable_shuffling=self.info.disable_shuffling,
            )
            split_info_futures.append(future)

        # Finalize the splits (after apache beam completed, if it was used)
        split_infos = [future.result() for future in split_info_futures]

        # Update the info object with the splits.
        split_dict = splits_lib.SplitDict(split_infos)
        self.info.set_splits(split_dict)


class _SplitInfoFuture:
    """Future containing the `tfds.core.SplitInfo` result."""

    def __init__(self, callback: Callable[[], splits_lib.SplitInfo]):
        self._callback = callback

    def result(self) -> splits_lib.SplitInfo:
        return self._callback()


def parse_examples_from_generator(paths, split_name, total_num_examples, features, serializer):
    generator = _generate_examples(paths)
    outputs = []
    for sample in utils.tqdm(
            generator,
            desc=f'Generating {split_name} examples...',
            unit=' examples',
            total=total_num_examples,
            leave=False,
            mininterval=1.0,
    ):
        if sample is None: continue
        key, example = sample
        try:
            example = features.encode_example(example)
        except Exception as e:  # pylint: disable=broad-except
            utils.reraise(e, prefix=f'Failed to encode example:\n{example}\n')
        outputs.append((key, serializer.serialize_example(example)))
    return outputs


class ParallelSplitBuilder(split_builder_lib.SplitBuilder):
    def __init__(self, *args, split_paths, parse_function, **kwargs):
        super().__init__(*args, **kwargs)
        self._split_paths = split_paths
        self._parse_function = parse_function

    def _build_from_generator(
            self,
            split_name: str,
            generator: Iterable[KeyExample],
            filename_template: naming.ShardedFileTemplate,
            disable_shuffling: bool,
    ) -> _SplitInfoFuture:
        """Split generator for example generators.

        Args:
          split_name: str,
          generator: Iterable[KeyExample],
          filename_template: Template to format the filename for a shard.
          disable_shuffling: Specifies whether to shuffle the examples,

        Returns:
          future: The future containing the `tfds.core.SplitInfo`.
        """
        total_num_examples = None
        serialized_info = self._features.get_serialized_info()
        writer = writer_lib.Writer(
            serializer=example_serializer.ExampleSerializer(serialized_info),
            filename_template=filename_template,
            hash_salt=split_name,
            disable_shuffling=disable_shuffling,
            file_format=self._file_format,
            shard_config=self._shard_config,
        )

        del generator  # use parallel generators instead
        paths = self._split_paths[split_name]
        path_lists = chunk_max(paths, N_WORKERS, MAX_PATHS_IN_MEMORY)  # generate N file lists
        print(f"Generating with {N_WORKERS} workers!")
        pool = Pool(processes=N_WORKERS)
        for i, paths in enumerate(path_lists):
            print(f"Processing chunk {i + 1} of {len(path_lists)}.")
            results = pool.map(
                partial(
                    parse_examples_from_generator,
                    split_name=split_name,
                    total_num_examples=total_num_examples,
                    serializer=writer._serializer,
                    features=self._features
                ),
                paths
            )
            # write results to shuffler --> this will automatically offload to disk if necessary
            print("Writing conversion results...")
            for result in itertools.chain(*results):
                key, serialized_example = result
                writer._shuffler.add(key, serialized_example)
                writer._num_examples += 1
        pool.close()

        print("Finishing split conversion...")
        shard_lengths, total_size = writer.finalize()

        split_info = splits_lib.SplitInfo(
            name=split_name,
            shard_lengths=shard_lengths,
            num_bytes=total_size,
            filename_template=filename_template,
        )
        return _SplitInfoFuture(lambda: split_info)

def dictlist2listdict(DL):
    " Converts a dict of lists to a list of dicts "
    return [dict(zip(DL, t)) for t in zip(*DL.values())]

def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si:si + (d + 1 if i < r else d)]

def chunk_max(l, n, max_chunk_sum):
    out = []
    for _ in range(int(np.ceil(len(l) / max_chunk_sum))):
        out.append(list(chunks(l[:max_chunk_sum], n)))
        l = l[max_chunk_sum:]
    return out
