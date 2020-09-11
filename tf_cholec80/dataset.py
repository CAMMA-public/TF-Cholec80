'''
Author: Tong Yu
Copyright (c) University of Strasbourg. All Rights Reserved.
'''

from glob import glob
import tensorflow as tf
import json
from pkg_resources import resource_stream


_MODES = [
  "FRAME",
  "VIDEO",
  "INFER"
]

_H = 480  # frame height
_W = 854  # frame width
_C = 3    # channels

_N_VIDEOS = 80


def make_cholec80(
  n_minibatch,
  config_path=None,
  video_ids=None,
  mode="FRAME"
):
  """Builds and returns Cholec80 as a tf.data.Dataset object

  Args:
      n_minibatch (int): number of samples in the minibatch
      config_path (str, optional): Path to the configuration file
      video_ids ([int], optional): ids of selected videos
      mode (str, optional): mode of operation

  Returns:
      a tf.data.Dataset feeding batches of annotated frames from Cholec80
  """
  dispatch = {
    "FRAME": FrameModeBuilder,
    "VIDEO": VideoModeBuilder,
    "INFER": InferenceModeBuilder
  }
  assert(mode in _MODES)
  builder = dispatch[mode](
    n_minibatch,
    config_path,
    video_ids
  )
  return builder.build()


class Cholec80Builder:
  """Base class for Cholec80 dataset builders 
  """
  def __init__(self, n_minibatch, config_path, video_ids):
    self._config = self.get_config(config_path)
    self._video_ids = video_ids
    self._n_minibatch = n_minibatch

  def prebuild(self):
    """Early build stage for the Cholec80 dataset.
    """
    filelist = self.grab_files(
      self._config["cholec80_dir"],
      self._video_ids
    )
    ds = tf.data.Dataset.from_tensor_slices(filelist)
    return ds
  
  def build(self):
    """Builds and returns the Cholec80 dataset
    """
    return NotImplementedError

  def expand(self, filename):
    """Expands a filename into a nested dataset.
    Meant to be used in conjunction with flat_map or interleave.
    """
    return NotImplementedError

  def parse_example(self, example_proto):
    """Decodes protos read from TFRecord files.
    Meant to be used in conjunction with map.
    """
    return NotImplementedError

  def grab_files(self, cholec80_dir, video_ids=None):
    """Gets the list of TFRecord files
    """
    all_files = sorted(glob("{}/*".format(cholec80_dir)))
    if video_ids is None:
      video_ids = range(_N_VIDEOS)
    return [all_files[i] for i in video_ids]

  def get_config(self, config_path):
    """Fetches and parses the config file
    """
    if config_path is None:
      try:
        config_dict = json.load(
          resource_stream("tf_cholec80", "configs/config.json")
        )
      except ModuleNotFoundError:
        with open("tf_cholec80/configs/config.json") as f:
          config_dict = json.loads(f.read())
    else:
      with open(config_path) as f:
        config_dict = json.loads(f.read())
    return config_dict

  def mk_template_dict(self):
    """Provides the template for parsing TFRecord protos
    """
    template_dict = {
      "frame"       : tf.io.FixedLenFeature([], tf.string),
      "video_id"    : tf.io.FixedLenFeature([], tf.string),
      "frame_id"    : tf.io.FixedLenFeature([], tf.int64),
      "total_frames": tf.io.FixedLenFeature([], tf.int64),
      "instruments" : tf.io.FixedLenFeature(7, tf.int64),
      "phase"       : tf.io.FixedLenFeature([], tf.int64)
    }
    return template_dict

  def parse_img(self, img_string):
    """Single frame parser
    """
    return tf.io.decode_png(img_string)

  def parse_img_batch(self, img_string_batch):
    """Batch frame parser
    """
    return tf.map_fn(self.parse_img, img_string_batch, dtype=tf.uint8)


class FrameModeBuilder(Cholec80Builder):
  """Builds the Cholec80 dataset for frame-based tasks
  """
  def __init__(self, n_minibatch, config_path, video_ids):
    super().__init__(n_minibatch, config_path, video_ids)

  def build(self):
    ds = self.prebuild()
    ds = ds.shuffle(self._config["n_file_shuffle"])
    ds = ds.interleave(
      self.expand,
      num_parallel_calls=self._config["n_parallel_interleave_calls"],
      cycle_length=self._config["n_interleave_cycle"],
      block_length=self._config["n_interleave_block"],
    )
    ds = ds.batch(self._n_minibatch)
    ds = ds.shuffle(self._config["n_batch_shuffle"])
    ds = ds.map(
      self.parse_example,
      num_parallel_calls=self._config["n_parallel_parse_calls"]
    )
    ds = ds.prefetch(self._config["n_prefetch"])
    return ds

  def expand(self, filename):
    ds = tf.data.TFRecordDataset(filename)
    return ds

  def parse_example(self, example_proto):
    parsed = tf.io.parse_example(example_proto, self.mk_template_dict())
    parsed["frame"] = tf.reshape(
      self.parse_img_batch(parsed["frame"]),
      [-1, _H, _W, _C]
    )
    parsed["end_flag"] = tf.constant(False)
    return parsed


class VideoModeBuilder(Cholec80Builder):
  """Builds the Cholec80 dataset for video-based tasks
  """
  def __init__(self, n_minibatch, config_path, video_ids):
    super().__init__(n_minibatch, config_path, video_ids)

  def build(self):
    ds = self.prebuild()
    ds = ds.shuffle(self._config["n_file_shuffle"])
    ds = ds.flat_map(self.expand)
    ds = ds.prefetch(self._config["n_prefetch"])
    return ds

  def expand(self, filename):
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.batch(self._n_minibatch)
    ds = ds.map(self.parse_example)
    return ds

  def parse_example(self, example_proto):
    parsed = tf.io.parse_example(example_proto, self.mk_template_dict())
    parsed["frame"] = tf.reshape(
      self.parse_img_batch(parsed["frame"]),
      [-1, _H, _W, _C]
    )
    max_frame = tf.reduce_max(parsed["frame_id"])
    parsed["end_flag"] = (tf.equal(max_frame, parsed["total_frames"][-1]))
    return parsed


class InferenceModeBuilder(Cholec80Builder):
  """Builds the Cholec80 dataset for inference tasks
  """
  def __init__(self, n_minibatch, config_path, video_ids):
    super().__init__(n_minibatch, config_path, video_ids)

  def build(self):
    ds = self.prebuild()
    ds = ds.flat_map(self.expand)
    ds = ds.prefetch(self._config["n_prefetch"])
    return ds

  def expand(self, filename):
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.batch(self._n_minibatch)
    ds = ds.map(self.parse_example)
    return ds

  def parse_example(self, example_proto):
    parsed = tf.io.parse_example(example_proto, self.mk_template_dict())
    parsed["frame"] = tf.reshape(
      self.parse_img_batch(parsed["frame"]),
      [-1, _H, _W, _C]
    )
    max_frame = tf.reduce_max(parsed["frame_id"])
    parsed["end_flag"] = (tf.equal(max_frame, parsed["total_frames"][-1]))
    return parsed
