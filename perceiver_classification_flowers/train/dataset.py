# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet dataset with pre-processing and augmentation.

Deng, et al CVPR 2009 - ImageNet: A large-scale hierarchical image database.
https://image-net.org/
"""

import enum
from typing import Any, Generator, Mapping, Optional, Sequence, Text, Tuple

import jax
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from train import autoaugment


Batch = Mapping[Text, np.ndarray]
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)
AUTOTUNE = tf.data.experimental.AUTOTUNE

INPUT_DIM = 224  # The number of pixels in the image resize.


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
    return {Split.TRAIN_AND_VALID: 1020, Split.TRAIN: 1020,
            Split.VALID: 1020, Split.TEST: 6149}[self]


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
  start, end = _shard(split, jax.host_id(), jax.host_count())

  im_size = (im_dim, im_dim)

  total_batch_size = np.prod(batch_dims)

  tfds_split = tfds.core.ReadInstruction(_to_tfds_split(split),
                                         from_=start, to=end, unit='abs')

  ds = tfds.load('oxford_flowers102', split=tfds_split,
                 decoders={'image': tfds.decode.SkipDecoding()})

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = threadpool_size
  options.experimental_threading.max_intra_op_parallelism = (
      max_intra_op_parallelism)
  options.experimental_optimization.map_parallelization = True
  if is_training:
    options.experimental_deterministic = False
  ds = ds.with_options(options)

  if is_training:
    if jax.host_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/vallabel must be divisible by {total_batch_size}')

  def crop_augment_preprocess(example):
    image, _ = _preprocess_image(
        example['image'], is_training, im_size, augmentation_settings)

    label = tf.cast(example['label'], tf.int32)

    out = {'images': image, 'labels': label}

    if is_training:
    
      if augmentation_settings['mixup_alpha'] is not None:
        beta = tfp.distributions.Beta(
            augmentation_settings['mixup_alpha'],
            augmentation_settings['mixup_alpha'])
        out['mixup_ratio'] = beta.sample()
    return out

  ds = ds.map(crop_augment_preprocess, num_parallel_calls=AUTOTUNE)

  # Mixup/cutmix by temporarily batching (using the per-device batch size):
  use_cutmix = augmentation_settings['cutmix']
  use_mixup = augmentation_settings['mixup_alpha'] is not None
  if is_training and (use_cutmix or use_mixup):
    inner_batch_size = batch_dims[-1]
    # Apply mixup, cutmix, or mixup + cutmix on batched data.
    # We use data from 2 batches to produce 1 mixed batch.
    ds = ds.batch(inner_batch_size * 2)
    if not use_cutmix and use_mixup:
      ds = ds.map(my_mixup, num_parallel_calls=AUTOTUNE)
    # Unbatch for further processing.
    ds = ds.unbatch()

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size)

  ds = ds.prefetch(AUTOTUNE)

  yield from tfds.as_numpy(ds)


def my_mixup(batch):
  """Apply mixup: https://arxiv.org/abs/1710.09412."""
  batch = dict(**batch)
  bs = tf.shape(batch['images'])[0] // 2
  ratio = batch['mixup_ratio'][:bs, None, None, None]
  images = (ratio * batch['images'][:bs] + (1.0 - ratio) * batch['images'][bs:])
  mix_labels = batch['labels'][bs:]
  labels = batch['labels'][:bs]
  ratio = ratio[..., 0, 0, 0]  # Unsqueeze
  return {'images': images, 'labels': labels,
          'mix_labels': mix_labels, 'ratio': ratio}


def _to_tfds_split(split: Split) -> tfds.Split:
  """Returns the TFDS split appropriately sharded."""
  # NOTE: Imagenet dlabel not release labels for the test split used in the
  # competition, so it has been typical at DeepMind to conslabeler the VALID
  # split the TEST split and to reserve 10k images from TRAIN for VALID.
  if split in (
      Split.TRAIN, Split.TRAIN_AND_VALID, Split.VALID):
    return tfds.Split.TRAIN
  else:
    assert split == Split.TEST
    return tfds.Split.VALIDATION


def _shard(
    split: Split, shard_index: int, num_shards: int) -> Tuple[int, int]:
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  arange = np.arange(split.num_examples)
  shard_range = np.array_split(arange, num_shards)[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  if split == Split.TRAIN:
    # Note that our TRAIN=TFDS_TRAIN[1020:] and VALID=TFDS_TRAIN[:1020].
    offset = Split.VALID.num_examples
    start += offset
    end += offset
  return start, end


def _preprocess_image(
    image_bytes: tf.Tensor,
    is_training: bool,
    image_size: Sequence[int],
    augmentation_settings: Mapping[str, Any],
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Returns processed and resized images."""

  # Get the image crop.
  if is_training:
    image, im_shape = _decode_whole_image(image_bytes)
    image = tf.image.random_flip_left_right(image)
  else:
    image, im_shape = _decode_whole_image(image_bytes)
  assert image.dtype == tf.uint8

  # # Optionally apply RandAugment: https://arxiv.org/abs/1909.13719

  # Resize and normalize the image crop.
  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outslabele the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  image = tf.image.resize(
      image, image_size, tf.image.ResizeMethod.BICUBIC)
  image = _normalize_image(image)

  return image, im_shape


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
  """Normalize the image to zero mean and unit variance."""
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image



def _decode_whole_image(image_bytes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  image = tf.io.decode_jpeg(image_bytes, channels=3)
  im_shape = tf.io.extract_jpeg_shape(image_bytes, output_type=tf.int32)
  return image, im_shape



