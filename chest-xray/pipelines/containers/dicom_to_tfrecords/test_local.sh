./build.sh ; docker run -t dicom-preprocess:latest --input_dir gs://$1/dicom --output_dir gs://$1/output/split --labels_file gs://$1/chest_xray_labels.txt --project $1 --mode local --bucket gs://$1/kubeflow