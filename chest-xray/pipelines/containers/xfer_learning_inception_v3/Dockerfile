FROM gcr.io/ml-pipeline/ml-pipeline-kubeflow-trainer:latest

RUN mkdir /chest-xray

RUN apt-get update -y && apt-get install --no-install-recommends -y -q ca-certificates python-dev python-setuptools wget unzip git 

RUN easy_install pip

RUN pip install tensorflow-hub

COPY train.py /chest-xray
COPY test_data/*.tfrecord /chest-xray/

ENTRYPOINT ["python", "/chest-xray/train.py"]
