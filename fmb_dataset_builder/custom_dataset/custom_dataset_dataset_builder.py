"""custom_dataset dataset."""
import time
from typing import Iterator, Tuple, Any, Dict, Union, Callable, Iterable

import glob
import os
import numpy as np
import tqdm
import cv2

import tensorflow_datasets as tfds
import tensorflow_hub as hub

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

def quaternion_to_euler_numpy(quaternions):
    """
    Convert a batch of quaternions to Euler angles.
    Args:
        quaternions (np.ndarray): Array of shape (N, 4) representing quaternions.
    Returns:
        np.ndarray: Array of shape (N, 3) representing Euler angles in degrees.
    """
    # Add axis 0 to quaternions if it is 1D
    quaternions = np.expand_dims(quaternions, axis=0)
    
    # Normalize quaternion
    # quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions.astype(np.float32) / np.linalg.norm(quaternions, axis=1, keepdims=True)

    # Extract components
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    # Euler angles calculation
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    # Convert radians to degrees
    euler_angles = np.stack([roll, pitch, yaw], axis=1) * (180 / np.pi)
    return euler_angles


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split."""

    # def _parse_example(data, instruct, instruct_embed):
    def _parse_example(episode_path):
        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        # import csv file 
        # (Frame,f_x,f_y,f_z,t_x,t_y,t_z,ee_p_x,ee_p_y,ee_p_z,ee_q_x,ee_q_y,ee_q_z,ee_q_w,j1,j2,j3,j4,j5,j6,j7,hole_p_x,hole_p_y,hole_p_z,hole_q_x,hole_q_y,hole_q_z,hole_q_w,peg_p_x,peg_p_y,peg_p_z,peg_q_x,peg_q_y,peg_q_z,peg_q_w,image_flag)
        data_list = np.loadtxt(episode_path, delimiter=',', skiprows=1, dtype=str)

        episode = []
        for i in range(data_list.shape[0]):
            data = data_list[i]
            data_next = data_list[i+1] if i+1 < data_list.shape[0] else data_list[i]

            # No velocity data in the AssemFormer dataset
            frame = data[0]
            f_x = data[1].astype(np.float32)
            f_y = data[2].astype(np.float32)
            f_z = data[3].astype(np.float32)
            t_x = data[4].astype(np.float32)
            t_y = data[5].astype(np.float32)
            t_z = data[6].astype(np.float32)
            ee_p_x = data[7].astype(np.float32)
            ee_p_y = data[8].astype(np.float32)
            ee_p_z = data[9].astype(np.float32)
            ee_q_x = data[10].astype(np.float32)
            ee_q_y = data[11].astype(np.float32)
            ee_q_z = data[12].astype(np.float32)
            ee_q_w = data[13].astype(np.float32)
            j1 = data[14].astype(np.float32)
            j2 = data[15].astype(np.float32)
            j3 = data[16].astype(np.float32)
            j4 = data[17].astype(np.float32)
            j5 = data[18].astype(np.float32)
            j6 = data[19].astype(np.float32)
            j7 = data[20].astype(np.float32)
            hole_p_x = data[21].astype(np.float32)
            hole_p_y = data[22].astype(np.float32)
            hole_p_z = data[23].astype(np.float32)
            hole_q_x = data[24].astype(np.float32)
            hole_q_y = data[25].astype(np.float32)
            hole_q_z = data[26].astype(np.float32)
            hole_q_w = data[27].astype(np.float32)
            peg_p_x = data[28].astype(np.float32)
            peg_p_y = data[29].astype(np.float32)
            peg_p_z = data[30].astype(np.float32)
            peg_q_x = data[31].astype(np.float32)
            peg_q_y = data[32].astype(np.float32)
            peg_q_z = data[33].astype(np.float32)
            peg_q_w = data[34].astype(np.float32)
            image_flag = data[35].astype(np.float32)

            ee_p_x_next = data_next[7].astype(np.float32)
            ee_p_y_next = data_next[8].astype(np.float32)
            ee_p_z_next = data_next[9].astype(np.float32)
            ee_q_x_next = data_next[10].astype(np.float32)
            ee_q_y_next = data_next[11].astype(np.float32)
            ee_q_z_next = data_next[12].astype(np.float32)
            ee_q_w_next = data_next[13].astype(np.float32)

            # change quaternion to euler angle
            ee_q = np.array([ee_q_x, ee_q_y, ee_q_z, ee_q_w])
            ee_euler = quaternion_to_euler_numpy(ee_q)[0]

            ee_q_next = np.array([ee_q_x_next, ee_q_y_next, ee_q_z_next, ee_q_w_next])
            ee_euler_next = quaternion_to_euler_numpy(ee_q_next)[0]

            ee_action = np.array([ee_p_x_next-ee_p_x,
                                    ee_p_y_next-ee_p_y,
                                    ee_p_z_next-ee_p_z,
                                    ee_euler_next[0]-ee_euler[0],
                                    ee_euler_next[1]-ee_euler[1],
                                    ee_euler_next[2]-ee_euler[2]]).astype(np.float32)
                                  
            # if data['primitive'][i] == 'insert':
            image_path = episode_path.replace('robot_data.csv', '{:04d}.png'.format(i))
            image = cv2.imread(image_path)

            # reshape image to 256x256
            image = cv2.resize(image, (256, 256))

            episode.append({
                'observation': {
                    # 'image_side_1': data['obs/side_1'][i].astype(np.uint8),
                    # 'image_side_2': data['obs/side_2'][i].astype(np.uint8),
                    # 'image_wrist_1': data['obs/wrist_1'][i].astype(np.uint8),
                    # 'image_wrist_2': data['obs/wrist_2'][i].astype(np.uint8),
                    # 'image_side_1_depth': data['obs/side_1_depth'][i].astype(np.float32),
                    # 'image_side_2_depth': data['obs/side_2_depth'][i].astype(np.float32),
                    # 'image_wrist_1_depth': data['obs/wrist_1_depth'][i].astype(np.float32),
                    # 'image_wrist_2_depth': data['obs/wrist_2_depth'][i].astype(np.float32),
                    'image_side_1': image.astype(np.uint8),
                    'image_side_2': image.astype(np.uint8),
                    'image_wrist_1': image.astype(np.uint8),
                    'image_wrist_2': image.astype(np.uint8),
                    'image_side_1_depth': image[:,:,0].astype(np.float32),
                    'image_side_2_depth': image[:,:,0].astype(np.float32),
                    'image_wrist_1_depth': image[:,:,0].astype(np.float32),
                    'image_wrist_2_depth': image[:,:,0].astype(np.float32),
                    
                    # 'joint_pos': data['obs/q'][i].astype(np.float32),
                    'joint_pos': np.array([j1, j2, j3, j4, j5, j6, j7]).astype(np.float32),

                    # 'joint_vel': data['obs/dq'][i].astype(np.float32),
                    'joint_vel': np.array([j1, j2, j3, j4, j5, j6, j7]).astype(np.float32),

                    # 'eef_pose': data['obs/tcp_pose'][i].astype(np.float32),
                    'eef_pose': np.array([ee_p_x, ee_p_y, ee_p_z, ee_q_x, ee_q_y, ee_q_z, ee_q_w]).astype(np.float32),

                    # 'eef_vel': data['obs/tcp_vel'][i].astype(np.float32),
                    'eef_vel': np.array([ee_p_x, ee_p_y, ee_p_z, ee_q_x, ee_q_y, ee_q_z, ee_q_w]).astype(np.float32),

                    # 'eef_force': data['obs/tcp_force'][i].astype(np.float32),
                    'eef_force': np.array([f_x, f_y, f_z]).astype(np.float32),

                    # 'eef_torque': data['obs/tcp_torque'][i].astype(np.float32),
                    'eef_torque': np.array([t_x, t_y, t_z]).astype(np.float32),

                    # 'state_gripper_pose': data['obs/gripper_pose'][i].astype(np.float32),
                    'state_gripper_pose': np.float_(0).astype(np.float32),

                    # 'primitive': data['primitive'][i],
                    'primitive': 'insert',

                    # 'shape_id': np.int_(data['object_info']['shape']).astype(np.uint8),
                    'shape_id': np.int_(1).astype(np.uint8),

                    # 'color_id': np.int_(data['object_info']['color']).astype(np.uint8),
                    'color_id': np.int_(1).astype(np.uint8),

                    # 'length': LENGTH_IDS[data['object_info']['length']],
                    'length': LENGTH_IDS['L'],

                    # 'size': SIZE_IDS[data['object_info']['size']],
                    'size': SIZE_IDS['L'],
                },
                # 'action': data['actions'][i].astype(np.float32),
                'action': ee_action,

                'discount': 1.0,

                # 'reward': float(i == (data['actions'].shape[0] - 1)),
                'reward': float(i == (data.shape[0] - 1)),

                'is_first': i == 0,
                # 'is_last': i == (data['actions'].shape[0] - 1),
                'is_last': i == (data.shape[0] - 1),

                # 'is_terminal': i == (data['actions'].shape[0] - 1),
                'is_terminal': i == (data.shape[0] - 1),

                # 'language_instruction': _parse_primitive_instruct(data['primitive'][i], data['object_info']),
                # 'language_embedding': INSTRUCT_EMBEDS[_parse_primitive_instruct(data['primitive'][i], data['object_info'])],
                'language_instruction': _parse_primitive_instruct('insert', {'shape': 1, 'color': 1, 'length': 'L', 'size': 'L'}),
                'language_embedding': INSTRUCT_EMBEDS[_parse_primitive_instruct('insert', {'shape': 1, 'color': 1, 'length': 'L', 'size': 'L'})],
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path,
                'episode_language_instruction': instruct,
                'episode_language_embedding': instruct_embed,
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return sample

    for episode_path in paths:
        # data = np.load(episode_path, allow_pickle=True).item()  # this is a list of dicts in our case
        # instruct = _parse_instruct(episode_path)
        # instruct_embed = INSTRUCT_EMBEDS[instruct]
        sample = _parse_example(episode_path)
        if sample is None:
            yield None
        else:
            yield episode_path, sample

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for custom_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(custom_dataset): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                    'steps': tfds.features.Dataset({
                        'observation': tfds.features.FeaturesDict({
                            'image_side_1': tfds.features.Image(
                                shape=(256, 256, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Side camera 1 RGB observation.',
                            ),
                            'image_side_2': tfds.features.Image(
                                shape=(256, 256, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Side camera 2 RGB observation.',
                            ),
                            'image_wrist_1': tfds.features.Image(
                                shape=(256, 256, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Wrist camera 1 RGB observation.',
                            ),
                            'image_wrist_2': tfds.features.Image(
                                shape=(256, 256, 3),
                                dtype=np.uint8,
                                encoding_format='jpeg',
                                doc='Wrist camera 2 RGB observation.',
                            ),
                            'image_side_1_depth': tfds.features.Tensor(
                                shape=(256, 256,),
                                dtype=np.float32,
                                doc='Side camera 1 depth observation.',
                            ),
                            'image_side_2_depth': tfds.features.Tensor(
                                shape=(256, 256,),
                                dtype=np.float32,
                                doc='Side camera 2 depth observation.',
                            ),
                            'image_wrist_1_depth': tfds.features.Tensor(
                                shape=(256, 256,),
                                dtype=np.float32,
                                doc='Wrist camera 1 depth observation.',
                            ),
                            'image_wrist_2_depth': tfds.features.Tensor(
                                shape=(256, 256,),
                                dtype=np.float32,
                                doc='Wrist camera 2 depth observation.',
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
                                shape=(7,),
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
                            shape=(6,),
                            # shape=(7,),
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
                })
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            # supervised_keys=('image', 'label'),  # Set to `None` to disable
            # homepage='https://dataset-homepage/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(custom_dataset): Downloads the data and defines the splits
        # path = dl_manager.download_and_extract('https://todo-data-url')
        base_path = '/media/sblee/170d6766-97d9-4917-8fc6-7d6ae84df8961/SSD2/workspaces/sim_pih_fmb/fmb_dataset_builder/custom_dataset/1127_with_pins/*_pins/shape_*/generator_tmp/episode_*/env_*/robot_data.csv'
        all_csv = glob.glob(base_path, recursive=True)
        path = {'train': all_csv}
        # TODO(custom_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        # return {
        #     'train': self._generate_examples(path / 'train_imgs'),
        # }
        return {
            'train': _generate_examples(path['train']),
        }

    def _generate_examples(self, path):
        # """Yields examples."""
        # # TODO(custom_dataset): Yields (key, example) tuples from the dataset
        # for f in path.glob('*.jpeg'):
        #     yield 'key', {
        #         'image': f,
        #         'label': 'yes',
        #     }
        pass # this is implemented in global method to enable multiprocessing