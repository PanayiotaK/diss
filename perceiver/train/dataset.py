# @title dataset.py
import enum
from typing import Any, Generator, Mapping, Optional, Sequence, Text, Tuple
import os
import jax
import random
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from train import autoaugment


Batch = Mapping[Text, np.ndarray]
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)
AUTOTUNE = tf.data.experimental.AUTOTUNE

INPUT_DIM = 384 //16  # The number of pixels in the image resize.


class Split(enum.Enum):
  """ImageNet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name: Text) -> 'Split':
    return {'TRAIN': Split.TRAIN, 'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
            'VALID': Split.VALID, 'VALIDATION': Split.VALID,
            'TEST': Split.TEST}[name.upper()]

  @property
  def num_examples(self):
    return {Split.TRAIN_AND_VALID: 40, Split.TRAIN: 40,
            Split.VALID: 40, Split.TEST: 40}[self]


def load(
    split: Split,
    *,
    is_training: bool,
    # batch_dims should be:
    # [device_count, per_device_batch_size] or [total_batch_size]
    batch_dims: Sequence[int],
    augmentation_settings: Mapping[str, Any],
    # The shape to which images are resized.
    im_dim: int = INPUT_DIM,
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[Batch, None, None]:
  """Loads the given split of the dataset."""
  # start, end = _shard(split, jax.host_id(), jax.process_count())
  
  im_size = (im_dim, im_dim)

  total_batch_size = np.prod(batch_dims)
  full_list = []
  if (split == Split.TRAIN or split == Split.TRAIN_AND_VALID ):
    path = '/content/dataset/train'
    filenames_list = os.listdir(path)    
  elif (split == Split.VALID):
      path = '/content/dataset/valid'
      filenames_list  = os.listdir(path)      
  else:
      path = '/content/dataset/test'
      filenames_list  = os.listdir(path)

  for file in filenames_list:
    full_list.append(os.path.join(path,file))

      
  ds = dataset_numpy(full_list)
  options = tf.data.Options()
  options.threading.private_threadpool_size = threadpool_size
  options.threading.max_intra_op_parallelism = (
      max_intra_op_parallelism)
  options.experimental_optimization.map_parallelization = True
  if is_training:
    options.experimental_deterministic = False
  ds = ds.with_options(options)

  if is_training:
    if jax.process_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/vallabel must be divisible by {total_batch_size}')

  # def add_random_labels(example):
  #   image = example #_preprocess_image(example['image'], is_training, im_size, augmentation_settings)
  #   label_init = random.randint(0, 600)
  #   label = tf.cast(label_init, tf.int32)
  #   out = {'images': image, 'labels': label}
  #   return out

  # ds = ds.map(add_random_labels, num_parallel_calls=AUTOTUNE)

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size)

  ds = ds.prefetch(AUTOTUNE)

  yield from tfds.as_numpy(ds)


def dataset_numpy(filenames_list):
    ''' Function that gets a list of paths of npz files and creates a tf.dataset. 
    Npz files contain the compressed video and label. 
    One npz = one video '''    
    if filenames_list :
        # initialize train dataset
        tensor_init = np.load(filenames_list[0])
        image_init = tensor_init['arr_0']
        class_init = np.expand_dims(tensor_init['arr_1'], axis=0)        
        video = np.expand_dims(image_init,axis=0 )        
        ds = tf.data.Dataset.from_tensor_slices({'images':video, 'labels' : class_init})   
                
        for file in filenames_list[1:]: 
            data = np.load(file)
            video =  np.expand_dims(data['arr_0'], axis=0)
            class_no = np.expand_dims(data['arr_1'], axis=0)
            add_ds = tf.data.Dataset.from_tensor_slices({'images':video, 'labels' : class_no})
            ds = ds.concatenate(add_ds)
        return ds 
    else:
        print("empty list")
    
