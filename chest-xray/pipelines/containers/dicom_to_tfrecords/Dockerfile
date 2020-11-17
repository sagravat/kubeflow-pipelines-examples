FROM gcr.io/ml-pipeline/ml-pipeline-dataflow-tft:latest

RUN mkdir /dicom-tf

RUN apt-get update -y && apt-get install --no-install-recommends -y -q ca-certificates python-dev python-setuptools wget unzip git python-skimage

RUN easy_install pip

RUN pip install pydicom && pip install Pillow && pip install scikit-image


COPY dicom_to_split_tfrecords.py /dicom-tf

ENTRYPOINT ["python", "/dicom-tf/dicom_to_split_tfrecords.py"]
