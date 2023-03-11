#!/bin/bash -eux

# Run this script from the repo's root folder
#
# $ ./docker/build-and-push.sh

# 1. Build Docker images for CPU and GPU

image="us-docker.pkg.dev/replicate/odegeasslbc/fastgan"
cpu_tag="$image:cpu"
gpu_tag="$image:gpu"

docker build -f docker/Dockerfile.cpu --tag "$cpu_tag" .
docker build -f docker/Dockerfile.gpu --tag "$gpu_tag" .

# 2. Test Docker images

test_output_folder=/tmp/test-chromagan/output

docker run -it --rm \
       -v $test_output_folder/cpu:/outputs \
       $cpu_tag \
       art --n_sample=20

[ -f $test_output_folder/cpu/0.png ] || exit 1
[ -f $test_output_folder/cpu/9.png ] || exit 1

docker run --gpus all -it --rm \
       -v $test_output_folder/gpu:/outputs \
       $gpu_tag \
       art --n_sample=20

[ -f $test_output_folder/gpu/0.png ] || exit 1
[ -f $test_output_folder/gpu/9.png ] || exit 1

sudo rm -rf "$test_output_folder"

# 3. Push Docker images

docker push $cpu_tag
docker push $gpu_tag
