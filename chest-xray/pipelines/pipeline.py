#!/usr/bin/env python3
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import kfp.dsl as dsl

class ObjectDict(dict):
  def __getattr__(self, name):
    if name in self:
      return self[name]
    else:
      raise AttributeError("No such attribute: " + name)


@dsl.pipeline(
  name='my-pipeline',
  description='preprocess pipeline'
)
def run(
    project,
    bucket,
    input_dir,
    output_dir,
    labels_file,
    mode='cloud'
):
  """Pipeline to train babyweight model"""
  start_step = 1

  # Step 1: create training dataset using Apache Beam on Cloud Dataflow
  if start_step <= 1:
    preprocess = dsl.ContainerOp(
      name='preprocess',
      # image needs to be a compile-time string
      image='gcr.io/{}/dicom-preprocess:latest'.format(project),
      arguments=[
        '--input_dir', input_dir,
        '--output_dir', output_dir,
        '--labels_file', labels_file,
        '--project', project,
        '--mode', mode,
      ],
      file_outputs={'bucket': '/output.txt'}
    )
  else:
    preprocess = ObjectDict({
      'outputs': {
        'bucket': bucket
      }
    })



if __name__ == '__main__':
  import kfp.compiler as compiler
  import sys
  if len(sys.argv) != 2:
    print("Usage: pipeline pipeline-output-name")
    sys.exit(-1)
  
  filename = sys.argv[1]
  compiler.Compiler().compile(run, filename)
