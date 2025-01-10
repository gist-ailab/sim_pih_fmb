import time
from typing import Iterator, Tuple, Any, Dict, Union, Callable, Iterable

import glob
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tqdm
import pandas as pd
from PIL import Image

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

N_WORKERS = 32 # number of parallel workers for data conversion
MAX_PATHS_IN_MEMORY = 200            # number of paths converted & stored in memory before writing to disk
                                    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                                    # note that one path may yield multiple episodes and adjust accordingly


_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

OBJECT_IDS = {
    1: 'rectangle',
    2: 'round',
    3: 'oval',
    4: 'hexagon',
    5: 'arch',
    6: 'square-circle',
    7: 'double-square',
    8: '3 prong',
    9: 'star',
}

COLOR_IDS = {
    1: 'brown',
    2: 'red',
    3: 'jeans red',
    4: 'yellow',
    5: 'green',
    6: 'jeans blue',
    7: 'dark blue',
    8: 'purple',
}

SIZE_IDS = {
    'L': 'large',
    'M': 'medium',
    'S': 'small'
}

LENGTH_IDS = {
    'L': 'long',
    'S': 'short'
}

DISTRACTOR_IDS = {
    'n': False,
    'y': True
}

INSTRUCT_EMBEDS = {}
for object in tqdm.tqdm(OBJECT_IDS.values()):
    for color in COLOR_IDS.values():
        for size in SIZE_IDS.values():
            for length in LENGTH_IDS.values():
                for distractor in DISTRACTOR_IDS.values():
                    for orientation in ['horizontal', 'vertical']:
                        instruct = f"Pick up the {size} {length} {color} {object} piece lying {orientation}ly and insert it {'with' if distractor else 'without'} distractors."
                        instruct_embed = _embed([instruct])[0].numpy()
                        INSTRUCT_EMBEDS[instruct] = instruct_embed

                        instruct = f"Pick up the {size} {length} {color} {object} piece lying {orientation}ly."
                        instruct_embed = _embed([instruct])[0].numpy()
                        INSTRUCT_EMBEDS[instruct] = instruct_embed

                        instruct = f"Insert the {size} {length} {color} {object} piece."
                        instruct_embed = _embed([instruct])[0].numpy()
                        INSTRUCT_EMBEDS[instruct] = instruct_embed

                        instruct = f"Regrasp the {size} {length} {color} {object} piece."
                        instruct_embed = _embed([instruct])[0].numpy()
                        INSTRUCT_EMBEDS[instruct] = instruct_embed

                        instruct = f"Place the {size} {length} {color} {object} piece on the fixture."
                        instruct_embed = _embed([instruct])[0].numpy()
                        INSTRUCT_EMBEDS[instruct] = instruct_embed

                        instruct = f"Rotate the {size} {length} {color} {object} piece."
                        instruct_embed = _embed([instruct])[0].numpy()
                        INSTRUCT_EMBEDS[instruct] = instruct_embed

instruct = f"Move up."
instruct_embed = _embed([instruct])[0].numpy()
INSTRUCT_EMBEDS[instruct] = instruct_embed

instruct = f"Move to above the board."
instruct_embed = _embed([instruct])[0].numpy()
INSTRUCT_EMBEDS[instruct] = instruct_embed


def _get_embedding(text):
    if text in INSTRUCT_EMBEDS.keys():
        return INSTRUCT_EMBEDS[text]
    else:
        instruct_embed = _embed([instruct])[0].numpy()
        INSTRUCT_EMBEDS[instruct] = instruct_embed
        return instruct_embed

def _parse_instruct(episode_path):
    filename = os.path.basename(episode_path)
    elems = filename.split('_')
    if "insert_only" in filename:
        object = OBJECT_IDS[int(elems[2])]
        size = SIZE_IDS[elems[3]]
        length = LENGTH_IDS[elems[4]]
        color = COLOR_IDS[int(elems[5])]
        return f"Insert the {size} {length} {color} {object} piece."
    else:
        object = OBJECT_IDS[int(elems[0])]
        size = SIZE_IDS[elems[1]]
        length = LENGTH_IDS[elems[2]]
        color = COLOR_IDS[int(elems[3])]
        orientation = elems[4]
        distractor = DISTRACTOR_IDS[elems[5]]
        return f"Pick up the {size} {length} {color} {object} piece lying {orientation}ly and insert it {'with' if distractor else 'without'} distractors."
    
def _parse_primitive_instruct(primitive, object_info):
    size = SIZE_IDS[object_info['size']]
    length = LENGTH_IDS[object_info['length']]
    color = COLOR_IDS[object_info['color']]
    object = OBJECT_IDS[object_info['shape']]
    if primitive == 'grasp':
        return f"Pick up the {size} {length} {color} {object} piece lying {orientation}ly."
    elif primitive == 'insert':
        return f"Insert the {size} {length} {color} {object} piece."
    elif primitive == 'regrasp':
        return f"Regrasp the {size} {length} {color} {object} piece."
    elif primitive == 'place_on_fixture':
        return f"Place the {size} {length} {color} {object} piece on the fixture."
    elif primitive == 'rotate':
        return f"Rotate the {size} {length} {color} {object} piece."
    elif primitive == 'move_up':
        return f"Move up."
    elif primitive == 'go_to_board':
        return f"Move to above the board."
    else:
        raise ValueError(f"Unknown primitive {primitive}.")



def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split by parsing CSV and loading images."""

    def _parse_example(episode_path):
        env_dir = os.path.dirname(episode_path)
        csv_path = episode_path
        data = pd.read_csv(csv_path)
        
        # Assuming images are named consistently
        image_side_1_path = os.path.join(env_dir, 'image_side_1.png')
        image_side_2_path = os.path.join(env_dir, 'image_side_2.png')
        image_wrist_1_path = os.path.join(env_dir, 'image_wrist_1.png')
        image_wrist_2_path = os.path.join(env_dir, 'image_wrist_2.png')
        
        # Load images and convert to numpy arrays
        try:
            image_side_1 = np.array(Image.open(image_side_1_path).resize((256, 256)))
            image_side_2 = np.array(Image.open(image_side_2_path).resize((256, 256)))
            image_wrist_1 = np.array(Image.open(image_wrist_1_path).resize((256, 256)))
            image_wrist_2 = np.array(Image.open(image_wrist_2_path).resize((256, 256)))
        except FileNotFoundError as e:
            print(f"Image not found: {e}")
            return None  # Skip this example if images are missing
        
        # Iterate over each row (timestep) in the CSV
        episode = []
        for i, row in data.iterrows():
            step = {
                'observation': {
                    'image_side_1': image_side_1,  # Replace with actual image for timestep i if different
                    'image_side_2': image_side_2,
                    'image_wrist_1': image_wrist_1,
                    'image_wrist_2': image_wrist_2,
                    # Add other observations from CSV
                    'joint_pos': np.array([row['j1'], row['j2'], row['j3'], row['j4'], row['j5'], row['j6'], row['j7']], dtype=np.float32),
                    'joint_vel': np.array([0.0]*7, dtype=np.float32),  # Placeholder
                    'eef_pose': np.array([row['ee_p_x'], row['ee_p_y'], row['ee_p_z'], row['ee_q_x'], row['ee_q_y'], row['ee_q_z'], row['ee_q_w']], dtype=np.float32),
                    'eef_vel': np.array([0.0]*6, dtype=np.float32),  # Placeholder
                    'eef_force': np.array([0.0]*3, dtype=np.float32),  # Placeholder
                    'eef_torque': np.array([0.0]*3, dtype=np.float32),  # Placeholder
                    'state_gripper_pose': row['hole_p_x'],  # Example, adjust accordingly
                    'primitive': 'insert' if row['image_flag'] == 1 else 'grasp',  # Example
                    'shape_id': 1,  # Example, map appropriately
                    'color_id': 1,  # Example, map appropriately
                    'length': 'L',  # Example, map accordingly
                    'size': 'M',  # Example, map accordingly
                },
                'action': np.array([row['f_x'], row['f_y'], row['f_z'], row['t_x'], row['t_y'], row['t_z'], 0.0], dtype=np.float32),  # Adjust as needed
                'discount': 1.0,
                'reward': float(i == (len(data) - 1)),
                'is_first': i == 0,
                'is_last': i == (len(data) - 1),
                'is_terminal': i == (len(data) - 1),
                'language_instruction': _parse_instruct(episode_path),  # Ensure this function is compatible
                'language_embedding': INSTRUCT_EMBEDS.get(_parse_instruct(episode_path), np.zeros(512, dtype=np.float32)),
            }
            episode.append(step)
        
        # Create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path,
                'episode_language_instruction': _parse_instruct(episode_path),
                'episode_language_embedding': INSTRUCT_EMBEDS.get(_parse_instruct(episode_path), np.zeros(512, dtype=np.float32)),
            }
        }

        return sample

    for episode_path in paths:
        sample = _parse_example(episode_path)
        if sample is None:
            continue
        else:
            yield episode_path, sample


class FmbSingleObjectDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Single Object Manipulation Dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
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
                            encoding_format='png',  # Changed to 'png'
                            doc='Side camera 1 RGB observation.',
                        ),
                        'image_side_2': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',  # Changed to 'png'
                            doc='Side camera 2 RGB observation.',
                        ),
                        'image_wrist_1': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',  # Changed to 'png'
                            doc='Wrist camera 1 RGB observation.',
                        ),
                        'image_wrist_2': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',  # Changed to 'png'
                            doc='Wrist camera 2 RGB observation.',
                        ),
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
                        'shape_id': tfds.features.Scalar(
                            dtype=np.uint8,
                            doc='Object ID of the object being manipulated.'
                        ),
                        'color_id': tfds.features.Scalar(
                            dtype=np.uint8,
                            doc='Color ID of the object being manipulated.'
                        ),
                        'length': tfds.features.Text(
                            doc='Length ID of the object being manipulated.'
                        ),
                        'size': tfds.features.Text(
                            doc='Size ID of the object being manipulated.'
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot end effector pose delta.',
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
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                             'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_language_instruction': tfds.features.Text(
                        doc='Language instruction for the entire episode.'
                    ),
                    'episode_language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding for the entire episode. '
                             'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define data splits by locating all robot_data.csv files."""
        # Adjust the base path as needed
        base_path = '/path/to/1127_with_pins/2pins/shape_000/episode_*/env_*/robot_data.csv'
        all_csv = glob.glob(base_path, recursive=True)
        
        return {
            'train': all_csv,  # Define more splits if necessary
        }

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        split_paths = self._split_paths()
        return {split: self._generate_examples(paths=split_paths[split]) for split in split_paths}

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Override to use the global _generate_examples function."""
        return _generate_examples(paths)

    # Optionally, override _download_and_prepare if needed
    # ...

# Register the dataset
# If you are creating a custom dataset, you might need to register it accordingly
# ...
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

