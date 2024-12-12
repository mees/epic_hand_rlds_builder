from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from hand_epic_dataset.conversion_utils import MultiThreadedDatasetBuilder
import json
import os
#import cv2
from PIL import Image
import pickle

data_path_hand_depth_epic = "/scratch/partial_datasets/oiermees/epickitchens/frames"
data_path_rgb_epic = "/datasets/epic100_2024-01-04_1913/frames"

def list_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def path_to_id(path_str, part_to_remove):
    desired_part = path_str.replace(part_to_remove, "").replace(".npy", "").replace(".jpg","").replace(".pkl","")
    return desired_part

def chunk_file_paths(file_paths, chunk_size=1000):
  """
  Chunks a list of file paths into smaller lists, grouping files from the same directory.

  Args:
    file_paths: A list of file paths.
    chunk_size: The desired size of each chunk.

  Returns:
    A list of chunks, each containing at most `chunk_size` file paths from the same directory.
  """

  chunks = []
  current_chunk = []
  current_dir = None

  for file_path in file_paths:
    file_dir = os.path.dirname(file_path)
    if current_dir is None or current_dir == file_dir:
      current_chunk.append(file_path)
      current_dir = file_dir
    else:
      # If we encounter a different directory, create a new chunk
      chunks.append(current_chunk)
      current_chunk = [file_path]
      current_dir = file_dir

  # Add the remaining files to the last chunk
  if current_chunk:
    chunks.append(current_chunk)

  # Now, chunk each directory's files into smaller chunks of size `chunk_size`
  all_chunks = []
  for i, chunk in enumerate(chunks):
    for j in range(0, len(chunk), chunk_size):
      all_chunks.append(chunk[j:j + chunk_size])

  return all_chunks

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    # _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_examples(episode_path):
        # load raw data --> this should change for your dataset
        print("inside parse examples: ", episode_path)

        for i, filepath in enumerate(episode_path):
            print("processing image: ", filepath)
            epic_id = path_to_id(episode_path, data_path_rgb_epic)
            im = Image.open(filepath)
            im = im.resize((224, 224), Image.Resampling.LANCZOS)
            im = np.asarray(im).astype(np.uint8)
            hand_keypoint_file_path = data_path_hand_depth_epic+epic_id+"_waypoints.pkl"
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
            depth_file = data_path_hand_depth_epic+filepath+".npy"
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
                        # 'state': tfds.features.Tensor(
                        #     shape=(7,),
                        #     dtype=np.float32,
                        #     doc='Robot state, consists of [7x robot joint angles, '
                        #         '2x gripper position, 1x door opening angle].',
                        # ),
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
        with open('/scratch/partial_datasets/oiermees/epic_annotated_file_paths.pkl', 'rb') as f:
            list_of_annotated_images = pickle.load(f)
        list_of_annotated_images_sorted = sorted(list_of_annotated_images)
        all_chunks = chunk_file_paths(list_of_annotated_images_sorted, chunk_size=1000)
        print(f"Converting {len(list_of_annotated_images)} training episodes.")
        return {
            'train': all_chunks,
        }
