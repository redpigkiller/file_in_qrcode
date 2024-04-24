import argparse
import glob
from pathlib import Path
import numpy as np

def load_file_pattern(file_pattern :str) -> list[str]:
    file_paths = sorted(glob.glob(file_pattern))

    if not file_paths:
        print(f"No files found matching the pattern: {file_pattern}")
    
    return file_paths


class FilePatternAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        pattern = values
        files = sorted(glob.glob(pattern))
        print(files)
        if not files:
            parser.error(f"No files found matching the pattern: {pattern}")
        
        # Split the files into separate lists based on the prefix
        file_groups = {}
        for file in files:
            prefix, num_ext = Path(file).stem.rsplit('_', 1)
            num = int(num_ext)  # Convert the number part to an integer
            file_groups.setdefault(prefix, []).append((num, file))
        
        # Sort the files within each group by the number
        sorted_files = []
        for group in file_groups.values():
            sorted_files.extend(sorted(file for num, file in group))
        
        setattr(namespace, self.dest, sorted_files)

def shuffle_color(arr :np.ndarray, seed :int=-1) -> np.ndarray:
    """
    Shuffle a 3-d numpy array with given seed (for invertibility).

    [Input]
        - arr   : 3-d numpy array
        - seed  : random seed (no shuffle if seed is negative)

    [Output]
        - shuffled arr  : 3-d numpy array
    """

    # Get the shape of the array
    h, w, c = arr.shape

    if seed >= 0:
        # Instantiate the random generator with seed set
        rng = np.random.default_rng(seed)

        # Generate the shuffled index array
        shuffle_index = np.linspace(0, c-1, c, dtype=np.int32)
        rng.shuffle(shuffle_index)

        # Shuffle the input array
        shuffle_arr = np.zeros((h, w, c), dtype=np.int32)
        shuffle_arr[:, :, shuffle_index] = arr

    else:
        shuffle_arr = arr

    return shuffle_arr

def unshuffle_color(arr :np.ndarray, seed :int=-1) -> np.ndarray:
    """
    Unshuffle a 3-d numpy array with given seed (for invertibility).

    [Input]
        - arr   : 3-d numpy array
        - seed  : random seed (no shuffle if seed is negative)

    [Output]
        - unshuffled arr  : 3-d numpy array
    """

    # Get the shape of the array
    h, w, c = arr.shape

    if seed >= 0:
        # Instantiate the random generator with seed set
        rng = np.random.default_rng(seed)

        # Generate the shuffled index array
        index = np.linspace(0, c-1, c, dtype=np.int32)
        shuffle_index = np.copy(index)

        # Generate the shuffled index array
        rng.shuffle(shuffle_index)

        # Generate the unshuffled index array
        unshuffle_index = np.zeros((c,), dtype=np.int32)
        unshuffle_index[shuffle_index] = index

        # Unshuffle the input array
        unshuffle_arr = np.zeros((h, w, c), dtype=np.int32)
        unshuffle_arr[:, :, unshuffle_index] = arr

    else:
        unshuffle_arr = arr

    return unshuffle_arr