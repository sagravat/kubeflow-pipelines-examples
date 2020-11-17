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
  name='train-pipeline',
  description='train pipeline'
)
def run(
    project,
    bucket,
    train_file,
    eval_file
):
  """Pipeline to train babyweight model"""
  start_step = 1

  # Step 1: create training dataset using Apache Beam on Cloud Dataflow
  if start_step <= 1:
    train = dsl.ContainerOp(
      name='train',
      # image needs to be a compile-time string
      image='gcr.io/{}/chest-xray-xfer-learning-train:latest'.format(project),
      arguments=[
        '--train_file', train_file,
        '--eval_file', eval_file,
      ],
      file_outputs={'bucket': '/output.txt'}
    )
    train.set_gpu_limit(4)
  else:
    train = ObjectDict({
      'outputs': {
        'bucket': bucket
      }
    })



if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(run, __file__ + '.tar.gz')
