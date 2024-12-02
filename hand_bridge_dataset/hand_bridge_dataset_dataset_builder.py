from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from hand_bridge_dataset.conversion_utils import MultiThreadedDatasetBuilder
import json
import os
#import cv2
from PIL import Image
import pickle

data_path_drawer = "/home/oier/Downloads/human_videos1/human_video/drawer"
data_path_spoon = "/home/oier/Downloads/human_videos1/human_video/spoon_cloth"


def get_depth_point(depth_map, x, y, smooth=True):
    height, width = depth_map.shape
    if x >= height:
        # print("x is greater than height: ", x)
        x = height - 1
    elif x < 0:
        x = 0
    if y >= width:
        # print("y is greater than width: ", y)
        y = width - 1
    elif y < 0:
        y = 0
    if smooth:
        # Define the bounds of the neighborhood
        min_y = max(0, y - 1)
        max_y = min(width, y + 2)
        min_x = max(0, x - 1)
        max_x = min(height, x + 2)

        # Extract the neighborhood
        neighborhood = depth_map[min_x:max_x, min_y:max_y]

        # Calculate the average value of the neighborhood
        avg_value = np.mean(neighborhood)
        if np.isnan(avg_value):
            print("nan value found in depth map")
            print("x: ", x, " y: ", y)
        return avg_value
    else:
        return depth_map[x, y]


def compute_visual_trajectory(observation, depth_image, gripper_pos):
    assert len(observation) == depth_image.shape[0]
    assert len(observation) == len(gripper_pos)
    depth_tcp = []
    cumulative_depth_keypoints = []
    cumulative_keypoints = []
    traj_length = len(observation)
    max_img_depth = np.max(depth_image)
    min_img_depth = np.min(depth_image)
    tcp_3d = []
    for i in range(traj_length):
        depth_kp = get_depth_point(depth_image[i], int(gripper_pos[i][0]), int(gripper_pos[i][1]), smooth=True)
        depth_tcp.append(depth_kp)
        tcp_3d.append([gripper_pos[i][0], gripper_pos[i][1], depth_kp])

    for i in range(traj_length):
        pairs = [(gripper_pos[i], gripper_pos[i + 1]) for i in range(i, len(observation) - 1, 1)]
        depth_pairs = [(depth_tcp[i], depth_tcp[i + 1]) for i in range(i, len(observation) - 1, 1)]
        cumulative_depth_keypoints.append(depth_pairs)
        cumulative_keypoints.append(pairs)

    temp_color_list = [int(255 * (i / traj_length)) for i in range(traj_length)]
    count_idx = 0
    list_of_traj_imgs = []
    # print("total length of trajectory: ", traj_length)
    for i in range(traj_length):
        # print("processing image: ", i)
        current_image = observation[i]['images0'].copy()
        trajectory = cumulative_keypoints[i]
        # print("length of trajectory: ", len(trajectory))
        depth_traj = cumulative_depth_keypoints[i]
        for j, keypoints in enumerate(trajectory):
            depth_color = ((depth_traj[j][0] - min_img_depth) / (max_img_depth - min_img_depth) * 255.0)
            cv2.line(current_image, (int(keypoints[0][0]), int(keypoints[0][1])),
                     (int(keypoints[1][0]), int(keypoints[1][1])),
                     color=(0, depth_color, temp_color_list[count_idx:][j]), thickness=2)
        list_of_traj_imgs.append(current_image)
        count_idx += 1

    return list_of_traj_imgs, tcp_3d


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    # _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_examples(episode_path):
        # load raw data --> this should change for your dataset
        print("inside parse examples: ", episode_path)
        # Get all .jpg files and exclude those ending with '_depth.jpg'
        jpg_files = glob.glob(f'{episode_path}/*.jpg')
        depth_files = set(glob.glob(f'{episode_path}/*_depth.jpg'))

        # Create a filtered list maintaining order
        filtered_jpg_files = [jpg_file for jpg_file in sorted(jpg_files) if jpg_file not in depth_files]
        episode = []
        first_idx = -1
        for i, jpg_file in enumerate(filtered_jpg_files):
            print("processing image: ", jpg_file)
            im = Image.open(jpg_file)
            im = im.resize((256, 256), Image.Resampling.LANCZOS)
            im = np.asarray(im).astype(np.uint8)
            hand_keypoint_file_path = jpg_file + "_waypoints.pkl"
            if os.path.exists(hand_keypoint_file_path):
            # Load the pickle file
                with open(hand_keypoint_file_path, 'rb') as file:
                    hand_keypoint = pickle.load(file)
                if not hand_keypoint:
                    print("hand keypoint not found")
                    continue
            else:
                print("hand keypoint file not found")
                continue
                # exit(0)
            depth_file = jpg_file.replace(".jpg", ".npy")
            if os.path.exists(depth_file):
                depth_image = np.load(depth_file)
                print("depth image shape: ", depth_image.shape)
                # list_traj_img, tcp_3d = compute_visual_trajectory(im, depth_image, hand_keypoint)
            else:
                print("depth image not found: ", depth_file)
                continue
                #exit(0)
            if hand_keypoint:
                if first_idx == -1:
                    first_idx = i
                # print(hand_keypoint)
                hand_key = next(iter(hand_keypoint[0].keys())) #retrieve the key (0 or 1 depending on the hand)
                depth_kp1 = get_depth_point(depth_image, int(hand_keypoint[1][hand_key][0][0]), int(hand_keypoint[1][hand_key][0][1]), smooth=True)
                depth_kp2 = get_depth_point(depth_image, int(hand_keypoint[1][hand_key][1][0]),
                                            int(hand_keypoint[1][hand_key][1][1]), smooth=True)
                depth_kp_avg = (depth_kp1 + depth_kp2) / 2
                # print(depth_kp_avg)
                tcp_3d = np.array([hand_keypoint[0][hand_key][0], hand_keypoint[0][hand_key][1], depth_kp_avg], dtype=np.float32)

                directory = os.path.basename(os.path.dirname(jpg_file))
                annotation = ""
                if "drawer" in directory:
                    annotation = "Put bread inside drawer"
                elif "spoon" in directory:
                    annotation = "Put green spoon on blue towel"
                # language_embedding = _embed([annotation])[0].numpy()

                episode.append({
                                   'observation': {
                                        'image_0': im,
                                        'tcp_point_2d': np.array(hand_keypoint[0][hand_key], dtype=np.int32),
                                        'tcp_point_3d': tcp_3d,
                                    },
                                   # 'action': example['actions'][i].astype(np.float32),
                                   'discount': 1.0,
                                   'reward': float(i == (len(filtered_jpg_files) - 1)),
                                   'is_first': i == first_idx,
                                   'is_last': i == (len(filtered_jpg_files) - 1),
                                   'is_terminal': i == (len(filtered_jpg_files) - 1),
                                   'language_instruction': annotation,
                               })
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        # data = np.load(episode_path, allow_pickle=True)  # this is a list of dicts in our case
        # count_not_found=0
        # for k, example in enumerate(data):
        #     # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        #     episode = []
        #     # episode_path_kun = episode_path.replace("/home/oier/", "/nfs/kun2/users/homer/datasets/bridge_data_all/")
        #     found = True
        #     if episode_path in gripper_pos_lookup:
        #         if str(k) in gripper_pos_lookup[episode_path]:
        #             gripper_pos = gripper_pos_lookup[episode_path][str(k)]['features']['gripper_position']
        #             if gripper_pos is None:
        #                 print("gripper position not found", episode_path, k)
        #                 # continue
        #                 found = False
        #             else:
        #                 # retrieve depth image
        #                 meta_id = f'{k}__{episode_path}'
        #                 meta_id = meta_id.replace('/', '\\')
        #                 depth_file = os.path.join(depth_path, meta_id)
        #                 if os.path.exists(depth_file):
        #                     depth_image = np.load(depth_file)
        #                     # print("loaded depth image shape: ", depth_image.shape)
        #                     list_traj_img, tcp_3d = compute_visual_trajectory(example['observations'], depth_image,
        #                                                                       gripper_pos)
        #                 else:
        #                     print("depth image not found")
        #                     print("depth file: ", depth_file)
        #                     found = False
        #         else:
        #             print("traj lookup not found")
        #             print(str(k))
        #             print(gripper_pos_lookup[episode_path].keys())
        #             found = False
        #
        #     else:
        #         print("gripper lookup not found")
        #         found = False
        #         # continue
        #     instruction = example['language'][0]
        #     if instruction:
        #         language_embedding = _embed([instruction])[0].numpy()
        #     else:
        #         language_embedding = np.zeros(512, dtype=np.float32)
        #
        #     if found:
        #         for i in range(len(example['observations'])):
        #             observation = {
        #                 'state': example['observations'][i]['state'].astype(np.float32),
        #             }
        #             for image_idx in range(4):
        #                 orig_key = f'images{image_idx}'
        #                 new_key = f'image_{image_idx}'
        #                 if orig_key in example['observations'][i]:
        #                     observation[new_key] = example['observations'][i][orig_key]
        #                 else:
        #                     observation[new_key] = np.zeros_like(example['observations'][i]['images0'])
        #
        #                 observation['visual_trajectory'] = list_traj_img[i]
        #                 observation['tcp_point_2d'] = np.array(gripper_pos[i], dtype=np.int32)
        #                 observation['tcp_point_3d'] = np.array(tcp_3d[i], dtype=np.float32)
        #                 observation['tcp_point_3d_trajectory'] = np.array(tcp_3d[i:], dtype=np.float32)
        #
        #             episode.append({
        #                 'observation': observation,
        #                 'action': example['actions'][i].astype(np.float32),
        #                 'discount': 1.0,
        #                 'reward': float(i == (len(example['observations']) - 1)),
        #                 'is_first': i == 0,
        #                 'is_last': i == (len(example['observations']) - 1),
        #                 'is_terminal': i == (len(example['observations']) - 1),
        #                 'language_instruction': instruction,
        #                 'language_embedding': language_embedding,
        #             })
        #     else:
        #         count_not_found += 1
        #         print("visual trajectory not found, counter: ", count_not_found)
        #
        #
        #     if len(episode) > 0 and found:
        #         # create output data sample
        #         sample = {
        #             'steps': episode,
        #             'episode_metadata': {
        #                 'file_path': episode_path,
        #                 'episode_id': k,
        #             }
        #         }
        #         # mark dummy values
        #         for image_idx in range(4):
        #             orig_key = f'images{image_idx}'
        #             new_key = f'image_{image_idx}'
        #             sample['episode_metadata'][f'has_{new_key}'] = orig_key in example['observations']
        #         sample['episode_metadata']['has_language'] = bool(instruction)
        #
        #         # if you want to skip an example for whatever reason, simply return None
        #         yield episode_path + str(k), sample
        #     else:
        #         print("episode is empty")
        #         return None




    # for smallish datasets, use single-thread parsing
    for sample in paths:
        for id, sample in _parse_examples(sample):
            yield id, sample


class HandBridgeDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    N_WORKERS = 1  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 1  # number of paths converted & stored in memory before writing to disk
    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        ),
                        'visual_trajectory': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Visual trajectory observation.',
                        ),
                        # 'depth': tfds.features.Tensor(
                        #     shape=(256, 256),
                        #     dtype=np.float32,
                        #     # encoding_format='jpeg',  # check of this is correct
                        #     doc='Main camera Depth observation.',
                        # ),
                        'tcp_point_2d': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.int32,
                            doc='TCP 2d point.',
                        ),
                        'tcp_point_3d': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='TCP 3d point.',
                        ),
                        'tcp_point_3d_trajectory': tfds.features.Sequence(
                            feature=tfds.features.Tensor(shape=(3,), dtype=np.float32),
                        ),
                        # 'trajectory_found': tfds.features.Scalar(
                        #     dtype=np.bool_,
                        #     doc='True on first step of the episode.'
                        # ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
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

                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='ID of episode in file_path.'
                    ),
                    'has_image_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image0 exists in observation, otherwise dummy value.'
                    ),
                    'has_image_1': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image1 exists in observation, otherwise dummy value.'
                    ),
                    'has_image_2': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image2 exists in observation, otherwise dummy value.'
                    ),
                    'has_image_3': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image3 exists in observation, otherwise dummy value.'
                    ),
                    'has_language': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if language exists in observation, otherwise empty string.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        base_paths = [data_path_drawer, data_path_spoon]
        train_filenames = [filename for path in base_paths for filename in glob.glob(f'{path}/*/')]
        print(f"Converting {len(train_filenames)} training episodes.")
        return {
            'train': train_filenames,
        }
