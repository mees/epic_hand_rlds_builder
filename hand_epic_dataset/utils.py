import numpy as np
import os

def list_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def path_to_id(path_str, part_to_remove):
    desired_part = path_str.replace(part_to_remove, "").replace(".npy", "").replace(".jpg","").replace(".pkl","")
    return desired_part

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