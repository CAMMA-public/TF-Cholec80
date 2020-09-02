'''
Author: Tong Yu
Copyright (c) University of Strasbourg. All Rights Reserved.
'''

from glob import glob
import tensorflow as tf
import json
from collections import namedtuple


_MODES = [
  "FRAME",
  "VIDEO",
  "INFER"
]

_H = 480
_W = 854
_C = 3

_N_VIDEOS = 80

def make_cholec80(
  n_minibatch,
  config_path="config.json",
  video_ids=None,
  mode="FRAMES"
):
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
  def __init__(self, n_minibatch, config_path, video_ids):
    self._config = self.get_config(config_path)
    self._video_ids = video_ids
    self._n_minibatch = n_minibatch

  def prebuild(self):
    filelist = self.grab_files(
      self._config["cholec80_dir"],
      self._video_ids
    )
    ds = tf.data.Dataset.from_tensor_slices(filelist)
    return ds
  
  def build(self):
    return NotImplementedError

  def grab_files(self, cholec80_dir, video_ids=None):
    all_files = sorted(glob("{}/*".format(cholec80_dir)))
    if video_ids is None:
      video_ids = range(_N_VIDEOS)
    return [all_files[i] for i in video_ids]

  def get_config(self, config_path):
    with open(config_path) as f:
      config_dict = json.loads(f.read())
    return config_dict

  def mk_template_dict(self):
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
    return tf.io.decode_png(img_string)

  def parse_img_batch(self, img_string_batch):
    return tf.map_fn(self.parse_img, img_string_batch, dtype=tf.uint8)


class FrameModeBuilder(Cholec80Builder):
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
      [self._n_minibatch, _H, _W, _C]
    )
    parsed["end_flag"] = tf.constant(False)
    return parsed


class VideoModeBuilder(Cholec80Builder):
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
      [self._n_minibatch, _H, _W, _C]
    )
    max_frame = tf.reduce_max(parsed["frame_id"])
    parsed["end_flag"] = (tf.equal(max_frame, parsed["total_frames"][-1]))
    return parsed


class InferenceModeBuilder(Cholec80Builder):
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
      [self._n_minibatch, _H, _W, _C]
    )
    max_frame = tf.reduce_max(parsed["frame_id"])
    parsed["end_flag"] = (tf.equal(max_frame, parsed["total_frames"][-1]))
    return parsed
