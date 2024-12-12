import pickle
import os

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

with open('epic_annotated_file_paths.pkl', 'rb') as f:
  list_of_annotated_images = pickle.load(f)

list_of_annotated_images_sorted = sorted(list_of_annotated_images)
chunk_file_paths(list_of_annotated_images_sorted, chunk_size=1000)