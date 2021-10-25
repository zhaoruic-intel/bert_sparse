#!/bin/bash

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

docker run -it --rm \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e LD_LIBRARY_PATH='/workspace/install/lib/' \
  -v $PWD:/workspace/bert \
  -v $PWD/results:/results \
  --env https_proxy=http://proxy-us.intel.com:912 \
  --env http_proxy=http://proxy-us.intel.com:911 \
  bert $CMD
