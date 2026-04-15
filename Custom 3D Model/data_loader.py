import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
from config import CFG
from utils import frames_from_video_file

class VideoDataset:
    def __init__(self):
        self.file_paths = []
        self.targets = []

    def load_files(self):
        for i, cls in enumerate(CFG.classes):
            sub_file_paths = glob.glob(f"{CFG.dataset_path}/{cls}/**.mp4")
            self.file_paths += sub_file_paths
            self.targets += [i] * len(sub_file_paths)

    def extract_features(self):
        features = []
        for file_path in tqdm(self.file_paths):
            features.append(frames_from_video_file(file_path, CFG.n_frames, CFG.output_size, CFG.frame_step))
        return np.array(features)

    def get_arrays(self):
        """
        Convenience method that loads file paths, extracts features for all videos,
        and returns (features, targets) as numpy arrays.

        Warning: This loads all processed frames into memory. For large datasets
        consider streaming or on-the-fly decoding instead of using this method.
        """
        # ensure file lists are populated
        if not self.file_paths or not self.targets:
            self.load_files()

        features = self.extract_features()
        targets = np.array(self.targets)

        # allow the caller to manage memory
        return features, targets

    def get_datasets(self):
        self.load_files()
        features = self.extract_features()

        train_features, val_features, train_targets, val_targets = train_test_split(
            features, self.targets, test_size=CFG.test_size, random_state=CFG.random_state
        )
        
        print(f"Train samples: {len(train_targets)} | Validation samples: {len(val_targets)}")

        # Build train dataset
        train_ds = tf.data.Dataset.from_tensor_slices((train_features, train_targets)) \
            .shuffle(CFG.batch_size * 4) \
            .batch(CFG.batch_size) \
            .cache() \
            .prefetch(1)   

        # Build validation dataset
        valid_ds = tf.data.Dataset.from_tensor_slices((val_features, val_targets)) \
            .batch(CFG.batch_size) \
            .cache() \
            .prefetch(1)  

        # Apply HPC-safe threading options
        options = tf.data.Options()
        options.threading.max_intra_op_parallelism = 1
        options.threading.private_threadpool_size = 1
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        train_ds = train_ds.with_options(options)
        valid_ds = valid_ds.with_options(options)

        # Clean up memory
        del train_features, val_features
        gc.collect()

        return train_ds, valid_ds

