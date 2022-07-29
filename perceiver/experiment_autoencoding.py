# @title Experiment.py 

import functools
from typing import Generator, Mapping, Text, Tuple

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
from ml_collections import config_dict
import numpy as np
import optax
import jax.tools.colab_tpu

import io_processors
import perceiver
from train import dataset
from train import utils

logging.get_absl_handler().use_absl_log_file('logging','./' )

FLAGS = flags.FLAGS

OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[Text, jnp.ndarray]


N_TRAIN_EXAMPLES = dataset.Split.TRAIN_AND_VALID.num_examples
N_CLASSES = 600
# Only local/debug parameters are supported out of the box.
# To use the scaled-up hyperparameters, please adapt this script to your
# training setup and set this flag to False
IS_LOCAL = False#True
NUM_FRAMES = 16
# SAMPLES_PER_PATCH = 16
NUM_CLASSES = 600
IMG_SZ = 24 # 24  #56


jax.tools.colab_tpu.setup_tpu()
# import jax
# logging.info("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
# logging.info("devices: [%s] ",jax.devices())

def get_training_steps(batch_size, n_epochs):
  return (N_TRAIN_EXAMPLES * n_epochs) // batch_size


def get_config():
  """Return config object for training."""
  use_debug_settings = IS_LOCAL
  config = base_config.get_base_config()

  # Experiment config.
  local_batch_size = 2
  # Modify this to adapt to your custom distributed learning setup
  num_devices = 1
  config.train_batch_size = local_batch_size * num_devices
  config.n_epochs = 110

  def _default_or_debug(default_value, debug_value):
    return debug_value if use_debug_settings else default_value

  n_train_examples = N_TRAIN_EXAMPLES
  num_classes = N_CLASSES

  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              optimizer=dict(
                  base_lr=5e-4,
                  max_norm=10.0,  # < 0 to turn off.
                  schedule_type='constant_cosine',
                  weight_decay=1e-1,
                  decay_pos_embs=True,
                  scale_by_batch=True,
                  cosine_decay_kwargs=dict(
                      init_value=0.0,
                      warmup_epochs=0,
                      end_value=0.0,
                  ),
                  step_decay_kwargs=dict(
                      decay_boundaries=[0.5, 0.8, 0.95],
                      decay_rate=0.1,
                  ),
                  constant_cosine_decay_kwargs=dict(
                      constant_fraction=0.5,
                      end_value=0.0,
                  ),
                  optimizer='lamb',
                  # Optimizer-specific kwargs:
                  adam_kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                      eps=1e-8,
                  ),
                  lamb_kwargs=dict(
                      b1=0.9,
                      b2=0.999,
                      eps=1e-6,
                  ),
              ),
          

              model=dict(
                  perceiver_kwargs=dict(
                      encoder=dict(
                        num_self_attends_per_block=8,
                        # Weights won't be shared if num_blocks is set to 1.
                        num_blocks=1,
                        z_index_dim=28*28*1,
                        num_z_channels=512,
                        num_cross_attend_heads=1,
                        num_self_attend_heads=8,
                        cross_attend_widening_factor=1,
                        self_attend_widening_factor=1,
                        dropout_prob=0.0,
                        z_pos_enc_init_scale=0.02,
                        cross_attention_shape_for_attn='kv',
                        name='encoder'
                        ),

                      decoder=dict(
                        # Autoencoding, don't pass inputs to the queries.
                        concat_preprocessed_input=False,                        
                        # Modality specific decoders are used ONLY to generate queries.
                        # All modalties are decoded together using a unified decoder.                       
                        num_outputs=None,
                        output_num_channels=512,
                        use_query_residual=False,
                      ),
                  ),
              ),
            
              training=dict(
                  inputs_per_epoch=n_train_examples,
                  label_smoothing=0.1,
                  n_epochs=config.get_oneway_ref('n_epochs'),
                  batch_size=config.get_oneway_ref('train_batch_size')
              ),
              data=dict(
                  num_classes=num_classes,
                  # Run on smaller inputs to debug.
                  im_dim = _default_or_debug(24, 24),
                  augmentation=dict(
                     
                  ),
                  ),
              evaluation=dict(
                  subset='test',
                  batch_size=2,
              ),
          )
      )
  )

  # Training loop config.
  config.training_steps = get_training_steps(
      config.get_oneway_ref('train_batch_size'),
      config.get_oneway_ref('n_epochs'))
  config.log_train_data_interval = 10
  config.log_tensors_interval = 10
  config.save_checkpoint_interval = 20
  config.eval_specific_checkpoint_dir = ''
  config.best_model_eval_metric = 'eval_top_1_acc'
  config.checkpoint_dir = ''
  config.train_checkpoint_all_hosts = False

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config


class Experiment(experiment.AbstractExperiment):
  """ImageNet experiment."""

  # A map from object properties that will be checkpointed to their name
  # in a checkpoint. Currently we assume that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)

    self.mode = mode
    self.init_rng = init_rng
    self.config = config

    # Checkpointed experiment state.
    self._params = None
    self._state = None
    self._opt_state = None

    # Input pipelines.
    self._train_input = None
    self._eval_input = None

    self.forward = hk.transform_with_state(self._forward_fn)

    # NOTE: We "donate" the `params, state, opt_state` arguments which allows
    # JAX (on some backends) to reuse the device memory associated with these
    # inputs to store the outputs of our function (which also start with
    # `params, state, opt_state`).
    self._update_func = jax.pmap(self._update_func, axis_name='i',
                                 donate_argnums=(0, 1, 2))
    self._eval_batch = jax.jit(self._eval_batch)

  def _forward_fn(
      self,
      inputs: dataset.Batch,
      is_training: bool,
      subsampling
  ) -> jnp.ndarray:

    inputs = inputs['images']
    
    subsampled_index_dims = {      
        'image': subsampling['image'].shape[0],
        'label': 1,
    }
    
      
    
    perceiver_kwargs = self.config.model.perceiver_kwargs
    output_postprocessor = io_processors.MultimodalPostprocessor(  modalities={                            
            'image': io_processors.ProjectionPostprocessor(
                num_outputs=1),
            'label': io_processors.ClassificationPostprocessor(
                num_classes=NUM_CLASSES),
        })
    input_preprocessor = io_processors.MultimodalPreprocessor(
        min_padding_size=4,        
        modalities={      
            'image': io_processors.ImagePreprocessor(
                position_encoding_type= 'fourier',
                fourier_position_encoding_kwargs=dict(
                    num_bands=32,
                    max_resolution=(NUM_FRAMES, IMG_SZ , IMG_SZ),
                    sine_only=False,
                    concat_pos=True,
                ),
                n_extra_pos_mlp=0,
                prep_type='pixels',
                spatial_downsample=4,
                temporal_downsample=1),
            'label': io_processors.OneHotPreprocessor(),
        },
        mask_probs={'image': 0.0, 'label': 1.0},
        )
    encoder = perceiver.PerceiverEncoder(**perceiver_kwargs['encoder'])
    decoder = perceiver.MultimodalDecoder(
        subsampled_index_dims=subsampled_index_dims,
        modalities={
                'image': perceiver.BasicVideoAutoencodingDecoder(
                    concat_preprocessed_input=False,
                    subsampled_index_dims=subsampling['image'],
                    output_shape=inputs.shape[:4],
                    num_z_channels=1024,
                    output_num_channels=512,
                    use_query_residual=False,
                    position_encoding_type='fourier',
                    fourier_position_encoding_kwargs=dict(
                      num_bands=32,
                      max_resolution=(NUM_FRAMES, IMG_SZ , IMG_SZ),
                      sine_only=False,
                      concat_pos=True,
                    ),   
                ),
                'label': perceiver.ClassificationDecoder(
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    num_classes=NUM_CLASSES,
                    num_z_channels=1024,
                    use_query_residual=False,
                    position_encoding_type='trainable',
                    trainable_position_encoding_kwargs=dict(
                        num_channels=1024,
                        init_scale=0.02,
                    ),
                ),
            },
        **perceiver_kwargs['decoder'])
    
    model = perceiver.Perceiver(
      input_preprocessor=input_preprocessor,
      encoder=encoder,
      decoder=decoder,
      output_postprocessor=output_postprocessor)

    return model({'image': inputs,                
                'label': np.zeros((inputs.shape[0], 600))},
               is_training=is_training, subsampled_output_points=subsampling)

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step: int, rng: jnp.ndarray,
           *unused_args, **unused_kwargs):
    """See base class."""

    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)

    self._params, self._state, self._opt_state, scalars = (
        self._update_func(
            self._params, self._state, self._opt_state, inputs, rng, global_step
            ))

    scalars = jl_utils.get_first(scalars)
    return scalars

  def _initialize_train(self):
    self._train_input = jl_utils.py_prefetch(self._build_train_input)

    total_batch_size = self.config.training.batch_size
    steps_per_epoch = (
        self.config.training.inputs_per_epoch / self.config.training.batch_size)
    total_steps = self.config.training.n_epochs * steps_per_epoch
    # Scale by the (negative) learning rate.
    self._lr_schedule = utils.get_learning_rate_schedule(
        total_batch_size, steps_per_epoch, total_steps, self.config.optimizer)

    self._optimizer = utils.make_optimizer(
        self.config.optimizer,
        self._lr_schedule)

    # Check we haven't already restored params
    if self._params is None:
      logging.info('Initializing parameters.')

      inputs = next(self._train_input)
      nchunks = 128
      image_chunk_size = np.prod(inputs['images'].shape[1:-1]) // nchunks
      subsampling = {
            'image': jnp.arange(
                image_chunk_size * 0 , image_chunk_size * ( 1)),
            
            'label': None,
        }
    
      init_net = jax.pmap(lambda *a: self.forward.init(*a,  subsampling=subsampling, is_training=True))
      init_opt = jax.pmap(self._optimizer.init)

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state.
      init_rng = jl_utils.bcast_local_devices(self.init_rng)

      self._params, self._state = init_net(init_rng, inputs)
      self._opt_state = init_opt(self._params)

  def _load_data(self, split, is_training, batch_dims):
    """Wrapper for dataset loading."""

    return dataset.load(
        split=split,
        is_training=is_training,
        batch_dims=batch_dims,
        im_dim=self.config.data.im_dim,
        augmentation_settings=self.config.data.augmentation,
        )

  def _build_train_input(self) -> Generator[dataset.Batch, None, None]:
    """See base class."""
    num_devices = jax.device_count()
    logging.info('No devices: %d ', num_devices)

    global_batch_size = self.config.training.batch_size
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

    if ragged:
      raise ValueError(
          f'Global batch size {global_batch_size} must be divisible by '
          f'num devices {num_devices}')

    split = dataset.Split.TRAIN_AND_VALID

    return self._load_data(
        split=split,
        is_training=True,
        batch_dims=[jax.local_device_count(), per_device_batch_size])

  def _one_hot(self, value):
    """One-hot encoding potentially over a sequence of labels."""
    y = jax.nn.one_hot(value, self.config.data.num_classes)
    return y

  def _loss_fn(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: dataset.Batch,
      rng: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Tuple[Scalars, hk.State]]:
    nchunks = 128
    reconstruction = {}
    
    for chunk_idx in range(nchunks):
        image_chunk_size = np.prod(inputs['images'].shape[1:-1]) // nchunks
        
        subsampling = {
            'image': jnp.arange(
                image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
           
            'label': None,
        }
        
        output, state = self.forward.apply(
            params, state, rng, inputs, subsampling=subsampling, is_training=True)
    
        reconstruction['label'] = output['label']
        if 'image' not in reconstruction:
            reconstruction['image'] = output['image']
            
        else:
            reconstruction['image'] = jnp.concatenate(
                [reconstruction['image'], output['image']], axis=1)
          
            
    reconstruction['image'] = jnp.reshape(reconstruction['image'], inputs['images'].shape)

    label = self._one_hot(inputs['labels'])
    
    # # Apply label-smoothing to one-hot labels.
    label_smoothing = self.config.training.label_smoothing
    if not (label_smoothing >= 0. and label_smoothing < 1.):
      raise ValueError(
          f"'label_smoothing is {label_smoothing} and should be in [0, 1)")
    if label_smoothing > 0:
      smooth_positives = 1. - label_smoothing
      smooth_negatives = label_smoothing / self.config.data.num_classes
      label = smooth_positives * label + smooth_negatives


######## New stuff here ###################
    loss_w_batch_class = utils.softmax_cross_entropy(reconstruction['label'], label)
    loss_w_batch_images = utils.l1_loss(reconstruction['image'], inputs['images'])
    
    loss_images = jnp.mean(loss_w_batch_images, dtype=loss_w_batch_images.dtype)
    loss_class = jnp.mean(loss_w_batch_class, dtype=loss_w_batch_class.dtype)
    
    loss = 0.03*loss_images + 1.0* loss_class
    
    scaled_loss = loss / jax.device_count()

    metrics = utils.topk_correct(reconstruction['label'], inputs['labels'], prefix='')
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)

    top_1_acc = metrics['top_1_acc']
    top_5_acc = metrics['top_5_acc']

    loss_scalars = dict(
        loss=loss,
        top_1_acc=top_1_acc,
        top_5_acc=top_5_acc,
    )

    return scaled_loss, (loss_scalars, state)

  def _update_func(
      self,
      params: hk.Params,
      state: hk.State,
      opt_state: OptState,
      inputs: dataset.Batch,
      rng: jnp.ndarray,
      global_step: int,
  ) -> Tuple[hk.Params, hk.State, OptState, Scalars]:
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
    scaled_grads, (loss_scalars, state) = grad_loss_fn(
        params, state, inputs, rng)
    grads = jax.lax.psum(scaled_grads, axis_name='i')

    # Grab the learning rate to log before performing the step.
    learning_rate = self._lr_schedule(global_step)

    # Compute and apply updates via our optimizer.
    updates, opt_state = self._optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    n_params = 0
    for k in params.keys():
      for l in params[k]:
        n_params = n_params + np.prod(params[k][l].shape)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {'learning_rate': learning_rate,
               'n_params (M)': float(n_params/1e6),
               'global_gradient_norm': optax.global_norm(grads)}
    loss_scalars = {f'train_{k}': v for k, v in loss_scalars.items()}
    scalars.update(loss_scalars)
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, state, opt_state, scalars

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_args):
    """See base class."""
    global_step = np.array(jl_utils.get_first(global_step))
    scalars = jax.device_get(self._eval_epoch(jl_utils.get_first(rng)))

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

  def _eval_batch(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: dataset.Batch,
      rng: jnp.ndarray,
  ) -> Scalars:
    """Evaluates a batch."""
    nchunks = 128
    reconstruction = {}
    for chunk_idx in range(nchunks):
        image_chunk_size = np.prod(inputs['images'].shape[1:-1]) // nchunks
        
        subsampling = {
            'image': jnp.arange(
                image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
            
            'label': None,
        }
        output, _ = self.forward.apply(
            params, state, rng, inputs, subsampling, is_training=False)
        reconstruction['label'] = output['label']
        if 'image' not in reconstruction:
            reconstruction['image'] = output['image']
            # reconstruction['audio'] = output['audio']
        else:
            reconstruction['image'] = jnp.concatenate(
                [reconstruction['image'], output['image']], axis=1)
            # reconstruction['audio'] = jnp.concatenate(
            
        
    reconstruction['image'] = jnp.reshape(reconstruction['image'], inputs['images'].shape)
    
    

    
    ### New stuff here ##############

    labels = self._one_hot(inputs['labels'])
    loss_class = utils.softmax_cross_entropy(reconstruction['label'], labels)
    loss_images = utils.l1_loss(reconstruction['image'], inputs['images'])
    
    loss = 0.03*loss_images + 1.0* loss_class

    metrics = utils.topk_correct(reconstruction['label'], inputs['labels'], prefix='')
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    top_1_acc = metrics['top_1_acc']
    top_5_acc = metrics['top_5_acc']

    bs = reconstruction['label'].shape[0]

    top_1_acc = jnp.expand_dims(top_1_acc, axis=0) * bs
    top_5_acc = jnp.expand_dims(top_5_acc, axis=0) * bs

    # NOTE: Returned values will be summed and finally divided by num_samples.
    return {
        'eval_loss': loss,
        'eval_top_1_acc': top_1_acc, 'eval_top_5_acc': top_5_acc}

  def _build_eval_input(self) -> Generator[dataset.Batch, None, None]:
    split = dataset.Split.from_string(self.config.evaluation.subset)

    return self._load_data(
        split=split,
        is_training=False,
        batch_dims=[self.config.evaluation.batch_size])

  def _eval_epoch(self, rng):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_scalars = None

    params = jl_utils.get_first(self._params)
    state = jl_utils.get_first(self._state)

    for inputs in self._build_eval_input():
      num_samples += inputs['labels'].shape[0]
      scalars = self._eval_batch(params, state, inputs, rng)

      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)

    mean_scalars = jax.tree_util.tree_map(lambda x: x / num_samples, summed_scalars)
    return mean_scalars


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))