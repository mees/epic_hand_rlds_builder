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
from utils import get_depth_point, path_to_id, list_files_in_directory

data_path_hand_depth_epic = "/scratch/partial_datasets/oiermees/epickitchens/frames"
data_path_rgb_epic = "/datasets/epic100_2024-01-04_1913/frames"

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    # _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_examples(demo_dict):
        # load raw data --> this should change for your dataset
        annotation = demo_dict["annotation"]
        episode_paths = demo_dict["file_paths"]
        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i, filepath in enumerate(episode_paths):
            print("processing image: ", filepath)
            epic_id = path_to_id(episode_paths, data_path_rgb_epic)
            im = Image.open(filepath)
            original_width, original_height = im.size
            new_height, new_width = 224, 224
            im = im.resize((new_height, new_width), Image.Resampling.LANCZOS)
            # compute the resizing scale to later scale the hand coordinates
            scale_x = new_width / original_width
            scale_y = new_height / original_height
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
                exit()
                # continue
            depth_file = data_path_hand_depth_epic+filepath+".npy"
            if os.path.exists(depth_file):
                depth_image = np.load(depth_file)
                print("depth image shape: ", depth_image.shape)
            else:
                print("depth image not found: ", depth_file)
                exit()
                #continue
            tcp_point_3d_left = np.array([0], dtype=np.float32)
            tcp_point_3d_right = np.array([0], dtype=np.float32)
            has_hand_left = False
            has_hand_right = False
            if hand_keypoint:
                for hand, kp2d in hand_keypoint[1].items():
                    # we can't query the actual TCP for depth, because that's in the air, so use thumb points
                    depth_kp1 = get_depth_point(depth_image, int(kp2d[0][0]), int(kp2d[0][1]), smooth=True)
                    depth_kp2 = get_depth_point(depth_image, int(kp2d[1][0]), int(kp2d[1][1]), smooth=True)
                    depth_kp_avg = (depth_kp1 + depth_kp2) / 2

                    if hand == 0:
                        tcp_point_2d_left = hand_keypoint[0][0]
                        has_hand_left = True
                        #scale the 2d point to the new image size
                        tcp_point_2d_left = (tcp_point_2d_left * np.array([scale_x, scale_y])).astype(float)
                        tcp_point_3d_left = np.hstack((tcp_point_2d_left, depth_kp_avg))
                    elif hand == 1:
                        tcp_point_2d_right = hand_keypoint[0][1]
                        has_hand_right = True
                        tcp_point_2d_right = (tcp_point_2d_right * np.array([scale_x, scale_y])).astype(float)
                        tcp_point_3d_right = np.hstack((tcp_point_2d_right, depth_kp_avg))

            episode.append({
                               'observation': {
                                    'image_0': im,
                                    'tcp_point_3d_left': tcp_point_3d_left,
                                    'tcp_point_3d_right': tcp_point_3d_right,
                                    'has_hand_left': has_hand_left,
                                    'has_hand_right': has_hand_right,
                                },
                               # 'action': example['actions'][i].astype(np.float32),
                               'discount': 1.0,
                               'reward': float(i == (len(episode_paths) - 1)),
                               'is_first': i == 0,
                               'is_last': i == (len(episode_paths) - 1),
                               'is_terminal': i == (len(episode_paths) - 1),
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
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'tcp_point_3d_left': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='TCP 3d point of the left hand.',
                        ),
                        'tcp_point_3d_right': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float32,
                            doc='TCP 3d point of the right hand.',
                        ),
                        'has_hand_left': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='True if left hand was detected.'
                        ),
                        'has_hand_right': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='True if right hand was detected.'
                        ),
                    }),
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
        with open('/scratch/partial_datasets/oiermees/epic_annotations.pkl', 'rb') as f:
            list_of_annotated_images = pickle.load(f)

        print(f"Converting {len(list_of_annotated_images)} training episodes.")
        return {
            'train': list_of_annotated_images,
        }