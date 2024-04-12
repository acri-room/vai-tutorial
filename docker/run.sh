#!/bin/bash
# Copyright 2022 Xilinx Inc.

confirm() {
  echo -en "\n\n$1 [y/n]? "
  read REPLY
  case $REPLY in
    [Yy]) ;;
    [Nn]) exit 0 ;;
    *) confirm ;;
  esac
    REPLY=''
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <image>"
    exit 2
fi

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

case $(hostname) in
  as006|as106)
    IMAGE_NAME=xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306
    ROCM=0
    ;;
  as005|as105)
    IMAGE_NAME=xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306
    ROCM=0
    ;;
  *)
    echo This host is not supported
    exit
    ;;
esac

image_exists=`(docker inspect --type=image $IMAGE_NAME 2> /dev/null || true) | jq 'length'`
if [[ $image_exists -eq 0 ]] ; then
  echo "Couldn't find Docker image: $IMAGE_NAME"
  image_tar=/tools/docker/${IMAGE_NAME/://}.tar
  if [[ -e $image_tar ]] ; then
    echo "Loading Docker image from $image_tar..."
    docker load < $image_tar
  fi
fi

xclmgmt_driver="$(find /dev -name xclmgmt\* 2> /dev/null)"
docker_devices=""
for i in ${xclmgmt_driver} ;
do
  docker_devices+="--device=$i "
done

render_driver="$(find /dev/dri -name renderD\* 2> /dev/null)"
for i in ${render_driver} ;
do
  docker_devices+="--device=$i "
done

kfd_driver="$(find /dev -name kfd\* 2> /dev/null)"
for i in ${kfd_driver} ;
do
    docker_devices+="--device=$i "
done

if [[ $ROCM -eq 1 ]] ; then
  volumes+="-v /opt/rocm-5.7.1:/opt/rocm-5.7.1 "
fi

docker_run_params=$(cat <<-END
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -v $PRJ_DIR:/workspace \
    -v /tools:/tools \
    $volumes \
    -e PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST \
    -e PIP_INDEX_URL=$PIP_INDEX_URL \
    -e PIP_NO_CACHE_DIR=$PIP_NO_CACHE_DIR \
    -w /workspace \
    --network=host \
    --shm-size=16g
    --detach
    --name vitisai
    ${RUN_MODE} \
    $IMAGE_NAME \
    tail -f /dev/null
END
)

##############################

if [[ ! -f "$PRJ_DIR/.confirm" ]]; then

    if [[ $IMAGE_NAME == *"gpu"* ]]; then
        arch="gpu"
    elif [[ $IMAGE_NAME == *"rocm"* ]]; then
        arch='rocm'
    else
        arch='cpu'
    fi

  prompt_file="$PRJ_DIR/docker/PROMPT_${arch}.txt"

  sed -n '1, 5p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '5, 15p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '15, 28p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '28, 61p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '62, 224p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '224, 308p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key

  sed -n '309, 520p' $prompt_file
  read -n 1 -s -r -p "Press any key to continue..." key
  
  confirm "Do you agree to the terms and wish to proceed"
fi

touch .confirm 
#docker pull $IMAGE_NAME 

# Check existing container and stop
for id in $(docker ps -a --filter "name=vitisai" --format "{{.ID}}") ; do
  name=$(docker ps -a --filter "id=$id" --format "{{.Names}}")
  if [[ $name == vitisai ]] ; then
    confirm "The Docker container named 'vitisai' already exists.\nDo you wish to delete it and proceed?"
    docker rm -f vitisai > /dev/null
  fi
done

set -ex

if [[ $IMAGE_NAME == *"gpu"* ]]; then
  docker run \
    $docker_devices \
    --gpus all \
    $docker_run_params
elif [[ $IMAGE_NAME == *"rocm"* ]]; then
  docker run \
    $docker_devices \
    --group-add=render --group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    $docker_run_params
else
  docker run \
    $docker_devices \
    $docker_run_params
fi

docker cp $PRJ_DIR/docker/bashrc vitisai:/etc/bash.bashrc

if [[ $ROCM -eq 1 ]] ; then
  docker exec vitisai rm /etc/alternatives/rocm
  docker exec vitisai ln -s /opt/rocm-5.7.1/ /etc/alternatives/rocm
fi

if [[ $# -gt 0 ]]; then
  docker exec -it vitisai "$@"
else
  docker exec -it vitisai bash
fi

