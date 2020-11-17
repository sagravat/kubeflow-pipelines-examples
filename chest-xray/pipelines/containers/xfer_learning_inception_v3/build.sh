if [ -z "$1" ]; then
  PROJECT_ID=$(gcloud config config-helper --format "value(configuration.properties.core.project)")
else
  PROJECT_ID=$1
fi

if [ -z "$2" ]; then
  TAG_NAME="latest"
else
  TAG_NAME="$2"
fi


CONTAINER_NAME=chest-xray-xfer-learning-train

docker build -t ${CONTAINER_NAME} .
docker tag ${CONTAINER_NAME} gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
docker push gcr.io/${PROJECT_ID}/${CONTAINER_NAME}:${TAG_NAME}
