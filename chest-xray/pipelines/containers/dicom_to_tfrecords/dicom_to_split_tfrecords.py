# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Beam pipeline to create TFRecord files from JPEG files stored on GCS.
These are the TFRecord format expected by  the resnet and amoebanet models.
Example usage:
python -m preprocess.py \
       --train_csv gs://cloud-ml-data/img/flower_photos/train_set.csv \
       --validation_csv gs://cloud-ml-data/img/flower_photos/eval_set.csv \
       --labels_file labels.txt \
       --project_id $PROJECT \
       --output_dir gs://${BUCKET}/output
The format of the CSV files is:
    URL-of-image,label
And the format of the labels_file is simply a list of strings one-per-line.
"""

from __future__ import print_function

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import apache_beam as beam
import tensorflow as tf
import pydicom
import logging
import numpy as np
from PIL import Image
import random

from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, SetupOptions
from skimage.transform import resize

_METRICS_NAMESPACE = 'radiology'

class ExtractFn(beam.DoFn):
  """DoFn for grouping dicoms by their SeriesInstanceUID (stack id)."""

  def __init__(self, output_dir):
    self._good_dicom = beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                                    'good_dicom')
    self._bad_dicom = beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                                   'bad_dicom')
    self._output_dir = output_dir

  def process(self, element):
    try:
      with tf.gfile.Open(element, 'r') as dicom_file:
        dicom_metadata = pydicom.read_file(dicom_file, stop_before_pixels=True)
      self._good_dicom.inc()
      series_uid = dicom_metadata.SeriesInstanceUID
      patients_age = dicom_metadata.PatientAge
      patients_sex = dicom_metadata.PatientSex
      study_description = dicom_metadata.StudyDescription
      if "|" not in study_description:
        logging.getLogger().info("extract %s, %s, %s, %s", series_uid, patients_age, patients_sex, study_description)
        yield series_uid, patients_age, patients_sex, study_description, element
    except Exception as e:  # pylint: disable=bare-except
      logging.getLogger().error(e)
      self._bad_dicom.inc()

class ConvertFn(beam.DoFn):
  """DoFn for grouping dicoms by their SeriesInstanceUID (stack id)."""

  def __init__(self, output_dir, LABEL_MAP):
    self._good_dicom = beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                                    'good_dicom')
    self._bad_dicom = beam.metrics.Metrics.counter(_METRICS_NAMESPACE,
                                                   'bad_dicom')
    self._output_dir = output_dir
    self.LABEL_MAP = LABEL_MAP

  def process(self, element):
    try:
      dicom_filename = element[4]
      
      with tf.gfile.Open(dicom_filename, 'r') as dicom_file:
        ds = pydicom.dcmread(dicom_file)
      self._good_dicom.inc()
      
      logging.getLogger().info("convert %s, %s", element[0], self.LABEL_MAP[element[3]])
      example = _convert_to_example(ds.pixel_array, self.LABEL_MAP[element[3]], 299, 299)
      #print(series_uid, patients_age, patients_sex, study_description)
      yield example.SerializeToString()
    except Exception as e:  # pylint: disable=bare-except
      logging.getLogger().error(e)
      self._bad_dicom.inc()


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(pixel_array, label_int, height, width):
                        
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label_int: integer, identifier for ground truth (0-based)
    label_str: string, identifier for ground truth, e.g., 'daisy'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  num_channels = 1
  if num_channels == 1:
      colorspace = 'grayscale'
  if num_channels == 3:
      colorspace = 'RGB'
  image_format = 'JPEG'

  #resized_array = resize(pixel_array, (299, 299))
  im = Image.fromarray(np.uint8(pixel_array))
  im = im.resize((299, 299), Image.NEAREST)
  #print("after image resize: ", np.asarray(im))
  #print("after scikit resize: ", resized_array)
  #print(resized_array)
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height': _int64_feature(height),
              'image/width': _int64_feature(width),
              'image/colorspace': _bytes_feature(colorspace),
              'image/channels': _int64_feature(num_channels),
              'image/class/label': _int64_feature(label_int),
              'image/format': _bytes_feature(image_format),
              #'image/filename': _bytes_feature(os.path.basename(filename)),
              'image/encoded': _bytes_feature(im.tobytes())
          }))
  return example


class DicomConverterOptions(PipelineOptions):
    """
    Runtime Parameters given during template execution
    path and organization parameters are necessary for execution of pipeline
    campaign is optional for committing to bigquery
    """
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            '--input_dir',
            type=str,
            help='Path of the file to read from')
        parser.add_argument(
            '--output_dir',
            type=str,
            help='Source name')
        parser.add_argument(
            '--labels_file',
            type=str,
            help='Source name')

def run(app_args ):
  logging.getLogger().info("app_args: ", app_args)
  test_mode = app_args.mode == 'local'
  JOBNAME = (
      'preprocess-images-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
  OUTPUT_DIR = app_args.output_dir
  INPUT_DIR = app_args.input_dir
  LABELS_FILE = app_args.labels_file
  PROJECT = app_args.project # PipelineOptions().view_as(GoogleCloudOptions).project

  if test_mode:
    logging.getLogger().info('Launching local job ... hang on')
    OUTPUT_DIR = './preproc'
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)
  else:
    logging.getLogger().info('Launching Dataflow job {} ... hang on'.format(JOBNAME))
    try:
      subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
    except subprocess.CalledProcessError:
      pass


  options = {
      'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
      'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
      'job_name': JOBNAME,
      'project': PROJECT,
      'teardown_policy': 'TEARDOWN_ALWAYS',
      'save_main_session': True
  }
  pipeline_options = PipelineOptions(flags=[], **options)
  #pipeline_options = PipelineOptions(flags=pipeline_args)
  #pipeline_options.view_as(SetupOptions).save_main_session = True
  #dicom_converter_options = PipelineOptions().view_as(DicomConverterOptions)
  if test_mode:
    RUNNER = 'DirectRunner'
  else:
    RUNNER = 'DataflowRunner'



  # clean-up output directory since Beam will name files 0000-of-0004 etc.
  # and this could cause confusion if earlier run has 0000-of-0005, for eg
  #if on_cloud:
    #try:
      #subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
    #except subprocess.CalledProcessError:
      #pass
  #else:
    #shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    #os.makedirs(OUTPUT_DIR)

  # read list of labels
  LABEL_MAP = {}
  with tf.gfile.FastGFile(LABELS_FILE, 'r') as f:
    LABELS = [line.rstrip() for line in f]
  
  for i, label in enumerate(LABELS):
    LABEL_MAP[label] = i + 1
  print('Read in {} labels, from {} to {}'.format(
      len(LABELS), LABELS[0], LABELS[-1]))
  if len(LABELS) < 2:
    print('Require at least two labels')
    sys.exit(-1)
  # set up Beam pipeline to convert images to TF Records

  dicom_file_pattern = os.path.join(INPUT_DIR, '*.dcm')
  dicom_files = tf.gfile.Glob(dicom_file_pattern)

  eval_percent = 15
  with beam.Pipeline(RUNNER, options=pipeline_options) as p:
    # BEAM tasks
    dataset = (
          p
          | 'read_files' >> beam.Create(dicom_files)
    )

    train_dataset, eval_dataset = (
        dataset
        | 'Split dataset' >> beam.Partition(
            lambda elem, _: int(random.uniform(0, 100) < eval_percent), 2)
    )

    step = "train"
    _ = (
         train_dataset 
          | '{}_extract'.format(step) >> beam.ParDo(ExtractFn(OUTPUT_DIR))
          | '{}_convert'.format(step) >>  beam.ParDo(ConvertFn(OUTPUT_DIR, LABEL_MAP))
          | '{}_write_tfr'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
               os.path.join(OUTPUT_DIR, "train"), file_name_suffix=".tfrecord", num_shards=50)
    )

    step = "eval"
    _ = (
         eval_dataset 
          | '{}_extract'.format(step) >> beam.ParDo(ExtractFn(OUTPUT_DIR))
          | '{}_convert'.format(step) >>  beam.ParDo(ConvertFn(OUTPUT_DIR, LABEL_MAP))
          | '{}_write_tfr'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
               os.path.join(OUTPUT_DIR, "val"), file_name_suffix=".tfrecord", num_shards=50)
    )


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_dir", type=str, required=True)
  parser.add_argument("--output_dir", type=str, required=True)
  parser.add_argument("--labels_file", type=str, required=True)
  parser.add_argument("--project", type=str, required=True)
  parser.add_argument("--mode", choices=['local', 'cloud'])
  #parser.add_argument("--pattern", dest="pattern", required=True)
  app_args,_ = parser.parse_known_args()

  run(app_args)
